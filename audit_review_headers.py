"""Read acquisition header metadata for the three updated-outcome review cases."""
import gzip
import json
import os
from pathlib import Path
import struct

root = Path(os.environ.get('HCCNET_NIFTI_DIR', '/deepstore/datasets/bms/hcc_study/nifti'))
rows = []
for pid in ['PT_042', 'PT_200', 'PT_214']:
    for folder in sorted(root.glob(pid+'_*')):
        for file in sorted(folder.glob('*.nii.gz')):
            with gzip.open(file, 'rb') as stream:
                header = stream.read(348)
            endian = '<' if struct.unpack('<i', header[:4])[0] == 348 else '>'
            rows.append({'visit': folder.name, 'file': file.name, 'description': header[148:228].split(b'\0')[0].decode('utf-8', errors='replace'), 'dimensions': struct.unpack(endian+'8h', header[40:56]), 'spacing': struct.unpack(endian+'8f', header[76:108])})
print(json.dumps(rows, indent=2))
