"""
EXECUTION NOT IN UNIX ENV

Rigidly nudge a tractogram so that it passes through voxels
with the highest QA values.

Input
-----
    qa_img_path : 3-D NIfTI with QA scalars in the range 0...1
    vtk_in      : input tractogram (RAS-mm)
Output
------
    <vtk_in_basename>_qaaligned.vtk   (RAS-mm)
"""

import numpy as np
import nibabel as nib
from dipy.io.streamline import load_vtk_streamlines, save_vtk_streamlines
from dipy.tracking.streamline import transform_streamlines
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------- paths -----
qa_img_path = r"C:\Users\2710i\OneDrive\Work\BachelorThesis\DSI_TRACTS\warp\HCP1065_qa.nii.gz"
vtk_in      = r"C:\Users\2710i\OneDrive\Work\BachelorThesis\DSI_TRACTS\warp\whole_brain_mni_downsampled.vtk"
vtk_out     = vtk_in.replace(".vtk", "_qaaligned.vtk")

# --------------------------------------------------------- 1.  Load data ----
qa_img   = nib.load(qa_img_path)
qa_data  = qa_img.get_fdata(dtype=np.float32)
qa_aff   = qa_img.affine
qa_inv   = np.linalg.inv(qa_aff)

streams  = load_vtk_streamlines(vtk_in, to_lps=False)   # tractogram is RAS (mm)

# Speed up
points = np.vstack([sl[::1] for sl in streams])
ones   = np.ones((points.shape[0], 1))
homog  = np.hstack([points, ones])   # N x 4 homogeneous coords

# ---------------------------------------------------- 2.  Objective func ----
def neg_score(trans_xyz):
    """Return minus the QA-sum (we minimise)."""
    # build 4x4 rigid shift
    T = np.eye(4)
    T[:3, 3] = trans_xyz                # translation in mm

    # transform all sampled points
    pts_mm = (T @ homog.T).T[:, :3]     # Nx3

    # map them into QA - image voxel space
    stacked   = np.hstack([pts_mm, np.ones((pts_mm.shape[0],1))])  # Nx4
    vox_homog = (qa_inv @ stacked.T).T   # Nx4
    vox       = vox_homog[:, :3]         # drop the 4th row -> Nx3

    # map_coordinates wants shape (ndim, num_points)
    coords = vox.T                       # 3xN

    # sample QA
    vals = map_coordinates(
        qa_data,
        coords,
        order=1,
        mode="constant",
        cval=0.0
    )
    return -vals.sum()

def neg_score_rigid(params):
    rot = R.from_euler('xyz', params[:3]).as_matrix()   # 3x3
    T   = np.eye(4)
    T[:3,:3] = rot
    T[:3,3]  = params[3:]

    # transform sampled points
    pts_mm = (T @ homog.T).T[:, :3]     # Nx3

    # map them into QA - image voxel space
    stacked   = np.hstack([pts_mm, np.ones((pts_mm.shape[0],1))])  # Nx4
    vox_homog = (qa_inv @ stacked.T).T   # Nx4
    vox       = vox_homog[:, :3]         # drop the 4th row -> Nx3

    # map_coordinates wants shape (ndim, num_points)
    coords = vox.T                       # 3xN

    # sample QA
    vals = map_coordinates(
        qa_data,
        coords,
        order=1,
        mode="constant",
        cval=0.0
    )
    return -vals.sum()


# ---------------------------------------------- 3a.  3-DOF translation only -------
opt_trans = minimize(
    neg_score,
    x0=np.zeros(3),            # [tx,ty,tz]
    method="Powell",
    options=dict(maxiter=100, disp=True))
best_shift = opt_trans.x
print("Best pure translation (mm):", best_shift.round(2))

# ----------------------------------------------------- 3b.  6-DOF rigid-body -------
# initial guess: the translation 
init6 = np.hstack((np.zeros(3), best_shift))

opt_rigid = minimize(
    neg_score_rigid,
    x0=init6,                  # [alpha,beta,gama, tx,ty,tz]
    method="Powell",
    options=dict(maxiter=200, disp=True))
alpha, beta, gamma, tx, ty, tz = opt_rigid.x
print("Best rigid params (rad,mm):", opt_rigid.x.round(4))

# ----------------------------------------------------- 4.  Save new VTK -----
# final 4x4 using the optimized rotation and translatio
rot_mat = R.from_euler('xyz', [alpha, beta, gamma]).as_matrix()
Tbest   = np.eye(4)
Tbest[:3,:3] = rot_mat
Tbest[:3, 3] = [tx, ty, tz]

aligned_streams = transform_streamlines(streams, Tbest)
save_vtk_streamlines(aligned_streams, vtk_out, to_lps=False, binary=True)
print("Written:", vtk_out)
