"""
NOT IN WSL

Align an RAS-mm tractogram to an LPS T1 volume for 3D Slicer.

Assumptions
-----------
* whole_brain_mni.trk is already in MNI RAS world-mm (Which )

Result
------
whole_brain_mni_downsampled.vtk :
    - compressed (downsampled)
    - geometry sparsified with approx_polygon_track(tol=0.2 mm)
    - correctly CENTERED on the T1 volume in Slicer (not fully alligned) !!!
"""

import numpy as np
import nibabel as nib
from random import sample
from dipy.io.streamline import load_trk, save_vtk_streamlines
from dipy.tracking.streamline import Streamlines, transform_streamlines
from dipy.tracking.distances import approx_polygon_track

# ------------------------------------------------------------------ paths ----
# ------------------------------------------------------------------ paths ----
# Use a common base directory for outputs if desired
vtk_out = r"C:\Users\2710i\OneDrive\Work\BachelorThesis\DSI_TRACTS\warp\whole_brain_mni_downsampled.vtk"

# Path to the MNI T1 template (target space)
t1_path = r"C:\Users\2710i\OneDrive\Work\BachelorThesis\DSI_TRACTS\mni_icbm152_nlin_asym_09a\mni_icbm152_t1_tal_nlin_asym_09a.nii"

# Path to the input tractogram
trk_path    = r"C:\Users\2710i\OneDrive\Work\BachelorThesis\DSI_TRACTS\warp\whole_brain_mni.trk"

keep_fraction = 0.05       # 5 % random subset for quick display
tol           = 0.2        # mm, geometric simplification tolerance
# --------------------------------------------------------------------------- #

# ---------------------------------------------------------------- 1)  T1 info
t1_img   = nib.load(t1_path)
t1_aff   = t1_img.affine           # 4×4 vox→world (LPS) matrix
t1_shape = t1_img.shape[:3]
voxel_sizes = np.linalg.norm(t1_aff[:3, :3], axis=0)  # mm per voxel

corner0 = t1_aff @ [0, 0, 0, 1]
corner1 = t1_aff @ [t1_shape[0]-1, t1_shape[1]-1, t1_shape[2]-1, 1]
t1_center = (corner0[:3] + corner1[:3]) / 2

print("\nT1 volume")
print("  dims (vox) :", t1_shape)
print("  voxel size :", voxel_sizes.round(3), "mm")
print("  world box  :", corner0[:3].round(1), "→", corner1[:3].round(1), "mm")
print("  world 0,0,0:", corner0[:3].round(1), "mm")
print("  world centre:", t1_center.round(1), "mm")

# ---------------------------------------------------------------- 2)  fibres
sft = load_trk(trk_path, "same", bbox_valid_check=False)
streamlines = sft.streamlines
print(f"\nLoaded tractogram : {len(streamlines):,} streamlines")

# optional speed/size subset
n_keep = int(len(streamlines) * keep_fraction)
subset = Streamlines(sample(list(streamlines), n_keep))
print(f"Using {n_keep:,} streamlines for bbox / compression")

# -------------------------------------------------------------- 3)  fibre box
all_pts = np.vstack([pts for sl in subset for pts in sl])
fib_min, fib_max = all_pts.min(axis=0), all_pts.max(axis=0)
fib_center       = (fib_min + fib_max) / 2
print("\nFibres (original RAS)")
print("  world box  :", fib_min.round(1), "→", fib_max.round(1), "mm")
print("  world centre:", fib_center.round(1), "mm")

# ------------------------------------------------ 4)  orientation + translate
lps2ras = np.diag([-1, -1, 1, 1])           # flip X,Y (LPS to RAS)
fib_center_lps = (lps2ras @ np.append(fib_center, 1))[:3]

translation = t1_center - fib_center_lps
shift = np.eye(4)
shift[:3, 3] = translation

aff_total = shift @ lps2ras

print("\nApplied transform")
print("  ras→lps flip matrix :\n", lps2ras)
print("  translation (mm)    :", translation.round(2))
print("  combined 4×4 affine :\n", aff_total)

aligned = transform_streamlines(subset, aff_total) # subset streamlines 

# ----------------------------------------------------------- 5)  compression
print("\nGeometry simplification …")
compressed = Streamlines(approx_polygon_track(s, tol) for s in aligned)
print(f"After compression : {len(compressed):,} streamlines")

# ----------------------------------------------------------- 6)  save VTK XML
print("Saving to VTK …")
save_vtk_streamlines(compressed,
                     vtk_out,
                     to_lps=False,
                     binary=True)
print("Saved :", vtk_out)
