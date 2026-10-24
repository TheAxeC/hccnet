"""Geometry checks for the private radiologist atlas."""
import sys
import tempfile
import unittest
from pathlib import Path
import nibabel as nib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from patient_figures import plane_geometry, sample


class PatientFigureGeometry(unittest.TestCase):
    def test_physical_plane_survives_axis_permutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data=np.zeros((11,13,5),np.float32)
            # Asymmetric marker: patient's right and posterior, visible at image bottom-left.
            data[8:10,1:3,2]=10
            affine=np.diag([2.,2.,3.,1.])
            original=nib.Nifti1Image(data,affine)
            nib.save(original,root/'T1WI.nii.gz')
            world,z,_=plane_geometry(root)
            self.assertEqual(z,6.)
            self.assertGreater(world[0,0,0],world[0,-1,0])
            self.assertLess(world[0,0,1],world[-1,0,1])
            rotated=original.as_reoriented(np.array([[1,-1],[0,1],[2,1]]))
            nib.save(rotated,root/'T1A.nii.gz')
            np.testing.assert_allclose(sample(root/'T1WI.nii.gz',world),sample(root/'T1A.nii.gz',world),atol=1e-5,equal_nan=True)
            marker=np.nanargmax(sample(root/'T1WI.nii.gz',world))
            row,col=np.unravel_index(marker,world.shape[:2])
            self.assertLess(row,world.shape[0]//2)
            self.assertLess(col,world.shape[1]//2)


if __name__=='__main__':unittest.main()
