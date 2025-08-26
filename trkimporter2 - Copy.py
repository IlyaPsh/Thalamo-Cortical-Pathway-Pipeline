import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from dipy.io.streamline import load_trk, save_vtk_streamlines
from dipy.tracking.streamline import Streamlines, transform_streamlines
from dipy.tracking.distances import approx_polygon_track
import os

WHOLE_BRAIN_VTK = r"C:\Users\2710i\OneDrive\Work\BachelorThesis\DSI_TRACTS\warp\whole_brain_mni_downsampled.vtk"
INPUT_TRKS = [
    r"C:\Users\2710i\OneDrive\Work\BachelorThesis\DSI_TRACTS\warp\ATLAS_TRACTS\N20_P20\ProjectionBrainstem_MedialLemniscusL.trk",
    r"C:\Users\2710i\OneDrive\Work\BachelorThesis\DSI_TRACTS\warp\ATLAS_TRACTS\N30\ProjectionBrainstem_DentatorubrothalamicTract-lr.trk",
    r"C:\Users\2710i\OneDrive\Work\BachelorThesis\DSI_TRACTS\warp\ATLAS_TRACTS\N30\ProjectionBrainstem_NonDecussatingDentatorubrothalamicTractL.trk",
    r"C:\Users\2710i\OneDrive\Work\BachelorThesis\DSI_TRACTS\warp\ATLAS_TRACTS\N20_P20\ProjectionBasalGanglia_ThalamicRadiationL_Superior.trk",
]

# 1) Safer polydata helper (adds verts)
def poly_from_points(pts_np):
    poly = vtk.vtkPolyData()
    vtk_pts = vtk.vtkPoints()
    arr = numpy_to_vtk(pts_np.astype('float32'), deep=1)
    arr.SetNumberOfComponents(3)
    vtk_pts.SetData(arr)
    poly.SetPoints(vtk_pts)

    # Add verts so VTK treats every point as an actual vertex
    verts = vtk.vtkCellArray()
    n = pts_np.shape[0]
    verts.Allocate(n)
    for i in range(n):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(i)
    poly.SetVerts(verts)
    return poly

# 2) Robust ICP wrapper
def icp_rigid_affine(moving_poly, fixed_poly,
                     *, max_iter=800, n_landmarks=None,
                     start_by_centroids=True, max_mean_dist=1e-4):
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(moving_poly)
    icp.SetTarget(fixed_poly)
    icp.GetLandmarkTransform().SetModeToRigidBody()

    icp.SetMaximumNumberOfIterations(int(max_iter))
    icp.SetMaximumMeanDistance(float(max_mean_dist))
    icp.CheckMeanDistanceOn()

    if n_landmarks is not None:
        icp.SetMaximumNumberOfLandmarks(int(n_landmarks))

    if start_by_centroids:
        icp.StartByMatchingCentroidsOn()
    else:
        icp.StartByMatchingCentroidsOff()

    icp.Modified(); icp.Update()

    M = icp.GetMatrix()
    A = np.eye(4)
    for i in range(4):
        for j in range(4):
            A[i, j] = M.GetElement(i, j)
    return A, float(icp.GetMeanDistance())


def restrict_wb_points(wb_pts, ctr, *, hemi_lock=True, radius_mm=60):
    """Return a subset of whole-brain points near ctr (and same hemisphere if desired)."""
    mask = np.all(np.abs(wb_pts - ctr) <= radius_mm, axis=1)
    if hemi_lock and abs(ctr[0]) > 3:
        mask &= (np.sign(wb_pts[:,0]) == np.sign(ctr[0]))
    sub = wb_pts[mask]
    if sub.shape[0] < 5000:
        sub = wb_pts[np.all(np.abs(wb_pts - ctr) <= radius_mm*1.5, axis=1)]
        if hemi_lock and abs(ctr[0]) > 3:
            sub = sub[np.sign(sub[:,0]) == np.sign(ctr[0])]
    return sub

# Load whole-brain reference once
wb_reader = vtk.vtkPolyDataReader(); wb_reader.SetFileName(WHOLE_BRAIN_VTK); wb_reader.Update()
wb_pts = vtk_to_numpy(wb_reader.GetOutput().GetPoints().GetData())

for trk_path in INPUT_TRKS:
    print("\n=== Processing:", trk_path)
    sft = load_trk(trk_path, "same", bbox_valid_check=False)
    streamlines = sft.streamlines
    pts_all = np.vstack(streamlines)
    ctr = pts_all.mean(axis=0)

    # 1. Tighter region per bundle
    filename = os.path.basename(trk_path).lower()
    if 'mediallemniscus' in filename:
        radius_mm = 50
    elif 'thalamicradiation' in filename:
        radius_mm = 85
    else:
        radius_mm = 70
    print(f"  Using region radius: {radius_mm} mm")
    
    wb_sub = restrict_wb_points(wb_pts, ctr, hemi_lock=True, radius_mm=radius_mm)
    if wb_sub.shape[0] > 100_000:
        sel = np.random.choice(wb_sub.shape[0], 100_000, replace=False)
        wb_sub = wb_sub[sel]
    print(f"  WB subset points: {wb_sub.shape[0]:,}")
    
    pts_coarse = pts_all if pts_all.shape[0] <= 150_000 else pts_all[np.random.choice(pts_all.shape[0], 150_000, replace=False)]
    pts_fine   = pts_all if pts_all.shape[0] <= 300_000 else pts_all[np.random.choice(pts_all.shape[0], 300_000, replace=False)]

    # 2. Outlier-robust source subset
    d = np.linalg.norm(pts_coarse - ctr, axis=1)
    keep = d <= np.quantile(d, 0.95)
    pts_coarse = pts_coarse[keep]
    print(f"  Moving points (coarse/fine): {pts_coarse.shape[0]:,} / {pts_fine.shape[0]:,}")

    # --- Coarse ICP ---
    A0, d0 = icp_rigid_affine(
        poly_from_points(pts_coarse), poly_from_points(wb_sub),
        max_iter=1500, n_landmarks=50_000, start_by_centroids=True, max_mean_dist=1e-3)
    
    # --- Fine ICP (pre-apply A0) ---
    ones = np.ones((pts_fine.shape[0], 1), dtype=np.float64)
    pts_fine_A0 = (A0 @ np.hstack([pts_fine, ones]).T).T[:, :3]
    A1, d1 = icp_rigid_affine(
        poly_from_points(pts_fine_A0), poly_from_points(wb_sub),
        max_iter=3000, n_landmarks=100_000, start_by_centroids=False, max_mean_dist=5e-4)
    A_total = A1 @ A0

    # 3. Two-pass fine ICP (settling pass)
    wb_relax = restrict_wb_points(wb_pts, ctr, hemi_lock=True, radius_mm=radius_mm + 10)
    if wb_relax.shape[0] > 120_000:
        wb_relax = wb_relax[np.random.choice(wb_relax.shape[0], 120_000, replace=False)]
    pts_fine_A_total = (A_total @ np.hstack([pts_fine, np.ones((pts_fine.shape[0],1))]).T).T[:, :3]
    A2, d2 = icp_rigid_affine(poly_from_points(pts_fine_A_total), poly_from_points(wb_relax),
                              max_iter=1500, n_landmarks=100_000, start_by_centroids=False, max_mean_dist=5e-4)
    A_total = A2 @ A_total
    
    # 4. Log transforms
    print("  Final mean dist (pass 2):", round(d2, 4))
    print("  Final A_total:\n", np.round(A_total, 4))
    
    # Hemisphere guard uses final A_total
    print(f"  Original centroid: {np.round(ctr, 2)}")
    ctr_new = (A_total @ np.r_[ctr, 1])[:3]
    print(f"  Moved centroid: {np.round(ctr_new, 2)}")
    if np.sign(ctr[0]) != 0 and np.sign(ctr_new[0]) != np.sign(ctr[0]):
        print("  [warn] hemisphere flip — retrying stricter region")
        wb_sub_strict = restrict_wb_points(wb_pts, ctr, hemi_lock=True, radius_mm=40)
        if wb_sub_strict.shape[0] > 80_000:
            sel = np.random.choice(wb_sub_strict.shape[0], 80_000, replace=False)
            wb_sub_strict = wb_sub_strict[sel]

        A0, _ = icp_rigid_affine(poly_from_points(pts_coarse), poly_from_points(wb_sub_strict),
                                 max_iter=1200, n_landmarks=40_000, start_by_centroids=True, max_mean_dist=1e-3)
        pts_fine_A0 = (A0 @ np.hstack([pts_fine, np.ones((pts_fine.shape[0],1))]).T).T[:, :3]
        A1, _ = icp_rigid_affine(poly_from_points(pts_fine_A0), poly_from_points(wb_sub_strict),
                                 max_iter=2500, n_landmarks=80_000, start_by_centroids=False, max_mean_dist=5e-4)
        A_total = A1 @ A0
        print("  Retry A_total:\n", np.round(A_total, 4))

    # Apply + simplify
    aligned = transform_streamlines(streamlines, A_total)
    tol = 0.2
    compressed = Streamlines(approx_polygon_track(s, tol) for s in aligned)

    vtk_out = trk_path.replace(".trk", ".vtk")
    save_vtk_streamlines(compressed, vtk_out, to_lps=False, binary=True)
    print("saved:", vtk_out)