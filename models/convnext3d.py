from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W, D)
    (2) DwConv -> Permute to (N, H, W, D, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        kernel_size (int): Kernel size of downsampling convolutional layer.
        drop_path (float): Stochastic depth rate. Default: 0.0
        use_v2 (bool): Whether to use ConvNeXt version 1 or version 2.
        layer_scale_init (float): Initial scaling factor of gamma parameter.
        eps (float): Epsilon to stabilize training. Default: 1e-6.
    """
    def __init__(
            self, 
            dim: int, 
            kernel_size: int = 3, 
            drop_path: float = 0.0, 
            use_v2: bool = False, 
            layer_scale_init: float = 1e-6,
            eps: float = 1e-6
        ) -> None:

        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, stride=1) # depthwise conv
        self.norm = LayerNorm(dim, eps=eps)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        if use_v2:
            self.grn = GRN(4 * dim, eps=eps)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        if not use_v2:
            self.gamma = nn.Parameter(layer_scale_init * torch.ones((dim)), requires_grad=True) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_v2 = use_v2

    def forward(
            self, 
            x: torch.Tensor
        ) -> torch.Tensor:

        inputs = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, H, D, W) -> (N, H, W, D, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        if self.use_v2:
            x = self.grn(x)
        x = self.pwconv2(x)
        if not self.use_v2:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3) # (N, H, W, D, C) -> (N, C, H, W, D)

        x = inputs + self.drop_path(x)
        return x

class ChannelAgnosticStem(nn.Module):
    r"""Channel-agnostic stem (paper Sec. 3.4).

    Applies a SHARED single-channel stem convolution to each input channel
    independently and aggregates the resulting feature maps, so a stem
    pre-trained on single-channel DINO views transfers to a multi-channel input
    (T1 IOP + T2, or fused modalities) WITHOUT re-initialising the first conv. The
    conv weight shape is independent of the number of input channels, so the
    DINO-pretrained stem loads directly at fine-tuning time.

    For a single-channel input this is identical to the original stem (mean over
    one channel is the identity), so DINO pre-training is unchanged.

    Args:
        out_dim (int): Output (stage-0) channels.
        kernel_size (int): Stem conv kernel size.
        stride (int): Stem conv stride.
        eps (float): LayerNorm epsilon.
        agg (str): Channel aggregation. 'mean' (default) keeps the feature scale
            stable across different channel counts; 'sum' is the alternative.
    """
    def __init__(self, out_dim: int, kernel_size: int, stride: int, eps: float = 1e-6, agg: str = 'mean') -> None:
        super().__init__()
        self.conv = nn.Conv3d(1, out_dim, kernel_size=kernel_size, stride=stride)
        self.norm = LayerNorm(out_dim, eps=eps, data_format="channels_first")
        # ===== CHANGE-ME (aggregation op): 'mean' vs 'sum'. Default 'mean'. =====
        self.agg = agg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[0], x.shape[1]
        x = x.reshape(b * c, 1, *x.shape[2:])   # shared stem per channel
        x = self.conv(x)
        x = x.reshape(b, c, *x.shape[1:])
        # ===== CHANGE-ME (aggregation op): switch to x.sum(dim=1) for 'sum'. =====
        x = x.mean(dim=1) if self.agg == 'mean' else x.sum(dim=1)
        x = self.norm(x)
        return x


class ConvNeXt3d(nn.Module):
    r""" ConvNeXt
        A 3D adaptation of the offical pytorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        kernel_size (int): Kernel size of downsampling convolutional layer. Defaults to 3.
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        use_v2 (bool): Whether to use ConvNeXt version 1 or version 2.
        layer_scale_init (float): Initial scaling factor of gamma parameter.
        eps (float): Epsilon to stabilize training. Default: 1e-6.
    """
    def __init__(
            self, 
            in_chans: int = 3, 
            num_classes: int = 1000, 
            kernel_size: int = 3,
            depths: list[int] = [3, 3, 9, 3], 
            dims: list[int] = [96, 192, 384, 768], 
            drop_path_rate: float = 0.0,
            use_v2: bool = False,
            layer_scale_init: float = 1e-6,
            eps: float = 1e-6,
            stem_mode: str = 'agnostic'
        ) -> None:

        super().__init__()
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        # stem_mode selects the stem. 'agnostic' (paper Sec 3.4): a shared single-channel
        # stem applied per input channel and aggregated, so the DINO-pretrained stem
        # transfers to multi-channel inputs without re-initialisation (in_chans is then only
        # informational). 'original' (Nolte's code): the standard multi-channel ConvNeXt
        # stem Conv3d(in_chans, ...); the 1-channel DINO stem is expanded to in_chans by a
        # repeat at load time (see train.py/pretrain.py load_weights). The two use different
        # state-dict keys, so a checkpoint must be trained and loaded under the same mode.
        if stem_mode == 'agnostic':
            stem = ChannelAgnosticStem(dims[0], kernel_size=3, stride=3, eps=eps)
        else:
            stem = nn.Sequential(
                nn.Conv3d(in_chans, dims[0], kernel_size=3, stride=3),
                LayerNorm(dims[0], eps=eps, data_format="channels_first"))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=eps, data_format="channels_first"),
                    nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(
                    dim=dims[i], kernel_size=kernel_size, drop_path=dp_rates[cur + j], use_v2=use_v2, layer_scale_init=layer_scale_init, eps=eps
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=eps) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    def _init_weights(
            self, 
            m: nn.Module
        ) -> None:

        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(
            self, 
            x: torch.Tensor
        ) -> torch.Tensor:

        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-3, -2, -1])) # global average pooling, (N, C, H, W, D) -> (N, C)

    def forward(
            self, 
            x: torch.Tensor
        ) -> torch.Tensor:

        x = self.forward_features(x)
        x = self.head(x)
        return x
    
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, depth, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width, depth).
    """
    def __init__(
            self, 
            normalized_shape: int, 
            eps: float = 1e-6, 
            data_format: str = "channels_last"
        ) -> None:

        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(
            self, 
            x: torch.Tensor
        ) -> torch.Tensor:

        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x
        
class GRN(nn.Module):
    """ 
    GRN (Global Response Normalization) layer
    """
    def __init__(
            self, 
            dim: int, 
            eps: float = 1e-6
        ) -> None:

        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        self.eps = eps

    def forward(
            self, 
            x: torch.Tensor
        ) -> torch.Tensor:
        
        Gx = torch.norm(x, p=2, dim=(1,2,3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x
        
def convnext3d_atto(**kwargs):
    model = ConvNeXt3d(depths=[2, 2, 6, 2], dims=[32, 64, 128, 256], **kwargs)
    return model

def convnext3d_femto(**kwargs):
    model = ConvNeXt3d(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnext3d_compact(**kwargs):
    # HCCNet-C: custom intermediate variant (C=56 -> H=448), ~0.75x HCCNet-Pico.
    # Uses 4 attention heads (head dim 112) in the Transformer; see MedNet(nhead=4).
    model = ConvNeXt3d(depths=[2, 2, 6, 2], dims=[56, 112, 224, 448], **kwargs)
    return model

def convnext3d_pico(**kwargs):
    model = ConvNeXt3d(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnext3d_nano(**kwargs):
    model = ConvNeXt3d(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnext3d_tiny(**kwargs):
    model = ConvNeXt3d(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnext3d_base(**kwargs):
    model = ConvNeXt3d(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnext3d_large(**kwargs):
    model = ConvNeXt3d(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnext3d_huge(**kwargs):
    model = ConvNeXt3d(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    return model