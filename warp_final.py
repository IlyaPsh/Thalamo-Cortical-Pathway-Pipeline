#!/usr/bin/env python3
from pathlib import Path
import subprocess, shlex
import vtk, numpy as np
import os
from vtk.util.numpy_support import numpy_to_vtk
from concurrent.futures import ProcessPoolExecutor
import tempfile, gc, shutil
import nibabel as nb
from scipy.ndimage import map_coordinates

nthreads = str(os.cpu_count())
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = nthreads  # already str()


"""
Requirements:

conda env with: nibabel, ants, FSL, antspyx
Usage:
cd /mnt/c/Users/2710i/OneDrive/Work/BachelorThesis/DSI_TRACTS
conda activate tract_env
python -u warp_final.py
Warp streamlines from native DWI space → ICBM-152 (2009c).
"""

overwrite = True  # Force overwrite for all steps

def _all_exist(paths):
    if isinstance(paths, (str, Path)): paths = [paths]
    return all(Path(p).is_file() for p in paths)

def run(cmd, outputs=None, *, quiet=False, **kw):
    """
    Run a shell command.
    Set quiet=True to hide command line + ANTs chatter.
    """
    if outputs and _all_exist(outputs) and not overwrite:
        name = Path(outputs[0] if isinstance(outputs, (list, tuple)) else outputs).name
        print(f"[skip] {name} exists — skipping {' '.join(str(c) for c in cmd)}")
        return

    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    cmd = [str(c) for c in cmd]

    # --------------- noisy vs. quiet -------------------
    if quiet:
        kw.setdefault("stdout", subprocess.DEVNULL)
        kw.setdefault("stderr", subprocess.DEVNULL)
    else:
        print("\n$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
        kw.setdefault("stdout", subprocess.PIPE)
        kw.setdefault("stderr", subprocess.STDOUT)

    # ---------------------------------------------------
    proc = subprocess.Popen(cmd, text=True, **kw)
    if not quiet:                    # stream live output only in verbose mode
        for line in proc.stdout:
            print(line, end="")
    proc.wait()

    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    if not quiet:
        print(f"[✓] {cmd[0]} completed", flush=True)


tmp = Path("/mnt/c/Users/2710i/OneDrive/Work/TestV6/NIFTI/warp_files")
tmp.mkdir(parents=True, exist_ok=True)
print(f"[info] working dir: {tmp}")

RAW_DWI     = Path("/mnt/c/Users/2710i/OneDrive/Work/TestV6/SLICER/processed_dwi/dwi.nii")
RAW_T1      = Path("/mnt/c/Users/2710i/OneDrive/AMU_20141001_MRI/Nifti/MRI_001)_Brain_20130516164213_5.nii")
TRACT_DIR   = Path("/mnt/c/Users/2710i/OneDrive/Work/TestV6/SLICER/masks")
RAW_MNI_T1  = Path("/mnt/c/Users/2710i/OneDrive/Work/BachelorThesis/DSI_TRACTS/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii")

bvec = Path("/mnt/c/Users/2710i/OneDrive/Work/TestV6/SLICER/processed_dwi/bvecs.bvec")
bval = Path("/mnt/c/Users/2710i/OneDrive/Work/TestV6/SLICER/processed_dwi/bvals.bval")

OUT_DIR    = Path("/mnt/c/Users/2710i/OneDrive/Work/TestV6")
VOLUME_DIR = Path("/mnt/c/Users/2710i/OneDrive/Work/TestV6/NIFTI/warped_volumes")
OUT_DIR.mkdir(exist_ok=True);  VOLUME_DIR.mkdir(exist_ok=True)

# Extract the first b0 volume (assuming it's volume 0) from the 4D DWI
b0 = tmp / "b0.nii.gz"
run(["fslroi", str(RAW_DWI), str(b0), "0", "1"], outputs=[b0])

# Brain extract T1
T1_brain = tmp / "T1_brain.nii.gz"
run(["bet", str(RAW_T1), str(T1_brain), "-R", "-f", "0.3", "-g", "0", "-m"], 
     outputs=[T1_brain])

# Brain extract b0
b0_brain = tmp / "b0_brain.nii.gz"
run(["bet", str(b0), str(b0_brain), "-m", "-f", "0.3"], 
     outputs=[b0_brain])


# Register b0 to T1
ants_prefix_str = str(tmp / "b0_to_T1_")
b0_to_t1_img = Path(f"{ants_prefix_str}Warped.nii.gz")
b0_to_t1_fwd_warp = Path(f"{ants_prefix_str}1Warp.nii.gz")
b0_to_t1_inv_warp = Path(f"{ants_prefix_str}1InverseWarp.nii.gz")
b0_to_t1_affine_mat = Path(f"{ants_prefix_str}0GenericAffine.mat")
metric_mi = f"MI[{b0_brain},{T1_brain},1,32,Regular,0.25]"
metric_cc = f"CC[{b0_brain},{T1_brain},1,4]"
run([
    "antsRegistration", "-d", "3",
    "-m", metric_mi,
    "-t", "Rigid[0.1]", "-c", "[1000x500x250,1e-6]", "-s", "4x2x1vox", "-f", "4x2x1",
    "-m", metric_mi,
    "-t", "Affine[0.1]", "-c", "[1000x500x250,1e-7]", "-s", "4x2x1vox", "-f", "4x2x1",
    "-m", metric_cc,
    "-t", "SyN[0.1,3,0]", "-c", "[100x70x50,1e-6]", "-s", "2x1x0vox", "-f", "4x2x1",
    "-v", "1",
    "-o", ants_prefix_str],
    outputs=[b0_to_t1_affine_mat, b0_to_t1_fwd_warp, b0_to_t1_inv_warp])

# b0 to T1 image
print("\n--- 1. Apply transforms to b0 into T1 space using ANTs ---")
run(["antsApplyTransforms", "-d", "3",
        "-i", str(b0_brain), "-o", str(b0_to_t1_img),
        "-t", str(b0_to_t1_affine_mat), "-t", str(b0_to_t1_inv_warp),
        "-r", str(T1_brain),
        "-v", "1"],
    outputs=[b0_to_t1_img])

# Register T1 to MNI using SyN
ants_prefix_str = str(tmp / "T1_to_MNI_")
t1_to_mni_img = Path(f"{ants_prefix_str}Warped.nii.gz")
t1_to_mni_fwd_warp = Path(f"{ants_prefix_str}1Warp.nii.gz")
t1_to_mni_inv_warp = Path(f"{ants_prefix_str}1InverseWarp.nii.gz")
t1_to_mni_affine_mat = Path(f"{ants_prefix_str}0GenericAffine.mat")
run(["antsRegistrationSyN.sh", "-d", "3",
     "-f", str(RAW_MNI_T1), "-m", str(T1_brain),
     "-o", ants_prefix_str, "-n", nthreads],
    outputs=[t1_to_mni_affine_mat, t1_to_mni_img, t1_to_mni_fwd_warp, t1_to_mni_inv_warp])

# b0 to MNI image
print("\n--- 1. Apply transforms to b0 into MNI space using ANTs ---")
b0_to_mni_img = tmp / "b0_to_MNI.nii.gz"
run(["antsApplyTransforms", "-d", "3",
        "-i", str(b0_brain), "-o", str(b0_to_mni_img),
        "-t", str(t1_to_mni_affine_mat), "-t", str(t1_to_mni_inv_warp),
        "-r", str(RAW_MNI_T1),
        "-v", "1"],
    outputs=[b0_to_mni_img])


print("\n--- 3. Apply transforms to streamlines ---")
print(f"[info] Applying transforms (DWI → MNI):")
print(f"       {t1_to_mni_affine_mat} (inv)")
print(f"       {t1_to_mni_inv_warp}")
print(f"       {b0_to_t1_inv_warp}")
print(f"       {b0_to_t1_affine_mat} (inv)")

max_workers = os.cpu_count()        # how many chunks / processes
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"   # avoid over-subscription
# -----------------------------------------------------------------------------

def warp_chunk(csv_in, csv_out):
    run([
        "antsApplyTransformsToPoints", "-d", "3",
        "-i", csv_in, "-o", csv_out,
        "-t", f"[{b0_to_t1_affine_mat},1]",
        "-t", f"{b0_to_t1_inv_warp}",
        "-t", f"[{t1_to_mni_affine_mat},1]",
        "-t", f"{t1_to_mni_inv_warp}"],
        quiet=True)          
    return csv_out  # for collection


for vtk_path_original in TRACT_DIR.rglob("*_fibers_trimmed.vtk"):
    print(f"\nProcessing tract: {vtk_path_original.name}")
    vtk_out_mni = vtk_path_original.parent / f"{vtk_path_original.stem}_icbm152_2009c.vtk"

    # ---------- load once, slice, and fan-out --------------------------------
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(vtk_path_original))
    reader.Update()
    pts_ras = np.asarray(reader.GetOutput().GetPoints().GetData())

    # RAS ➜ LPS, since antsApplyTransformsToPoints expects points in LPS coordinates
    pts_ras[:, :2] *= -1
    pts_lps_chunks = np.array_split(pts_ras, max_workers)     # ≈ N/cores each

    tmpdir = Path(tempfile.mkdtemp(prefix="warp_chunks_"))
    csv_in_list  = []
    csv_out_list = []

    for idx, chunk in enumerate(pts_lps_chunks):
        cin  = tmpdir / f"chunk_{idx}.csv"
        cout = tmpdir / f"chunk_{idx}_out.csv"
        csv_in_list.append(str(cin));  csv_out_list.append(str(cout))

        # write header once per chunk
        np.savetxt(cin, np.c_[chunk, np.zeros(len(chunk)), np.arange(len(chunk))],
                   delimiter=",", fmt="%.6f", header="x,y,z,t,label", comments="")

    # ---------- warp each chunk in a separate process -------------------------
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for _ in ex.map(warp_chunk, csv_in_list, csv_out_list):
            pass

    # ---------- merge, convert back to RAS, build VTK -------------------------
    chunks_warped = [np.loadtxt(cout, delimiter=",", skiprows=1, usecols=(0,1,2))
                     for cout in csv_out_list]
    pts_lps_new = np.vstack(chunks_warped)

    # LPS ➜ RAS
    pts_lps_new[:, :2] *= -1

    vtk_arr = numpy_to_vtk(pts_lps_new, deep=1)
    vtk_arr.SetNumberOfComponents(3)
    vtk_pts = vtk.vtkPoints();  vtk_pts.SetData(vtk_arr)
    poly = reader.GetOutput();  poly.SetPoints(vtk_pts);  poly.Modified()

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(vtk_out_mni));  writer.SetInputData(poly);  writer.Write()
    print(f"[✓] Streamlines saved to {vtk_out_mni}")

    # clean up temp files & be nice to RAM
    shutil.rmtree(tmpdir, ignore_errors=True)
    del pts_ras, pts_lps_new, pts_lps_chunks, chunks_warped
    gc.collect()

print("\nAll streamlines processed successfully!")

os.environ["OMP_NUM_THREADS"] = "1"   # for FNIRT + img2imgcoord

# Optionally cross-check with FSL results
#  EPI (b0) ➜ T1  Affine        ──>  epi_reg.mat
epi_out      = tmp / "epi_reg_dwi_to_t1"
epi_mat_fsl  = epi_out.with_suffix(".mat")      # b0 ➜ T1 (FLIRT 4×4)
run(["epi_reg",
     f"--epi={b0}", f"--t1={RAW_T1}", f"--t1brain={T1_brain}",
     f"--out={epi_out}"],
    outputs=[epi_mat_fsl])

"""

# The following section attempts to replicate the results using FSL tools
# THIS DOES NOT WORK

# -------------------------------------------------------------------------
# 3.  T1 ➜ MNI  Affine (FLIRT)     ──>  t1_to_mni_affine.mat
# -------------------------------------------------------------------------
t1_to_mni_affine = tmp / "t1_to_mni_affine.mat"
run(["flirt", "-in", str(T1_brain), "-ref", str(RAW_MNI_T1),
     "-omat", str(t1_to_mni_affine), "-dof", "12", "-out",
     str(tmp / "T1_in_MNI_lin.nii.gz")],
    outputs=[t1_to_mni_affine])

# -------------------------------------------------------------------------
# 4.  T1 ➜ MNI  Non-linear (FNIRT) ──>  t1_to_mni_warp.nii.gz
# -------------------------------------------------------------------------
t1_to_mni_warp = tmp / "t1_to_mni_warp.nii.gz"
run(["fnirt", "--in=" + str(RAW_T1),
     "--aff=" + str(t1_to_mni_affine),
     "--cout=" + str(t1_to_mni_warp),
     "--config=T1_2_MNI152_2mm"],                    # typical config
    outputs=[t1_to_mni_warp])

# -------------------------------------------------------------------------
# 5.  Compose to one  b0 ➜ MNI  warp ──>  b0_to_mni_warp.nii.gz
# -------------------------------------------------------------------------
b0_to_mni_warp = tmp / "b0_to_mni_warp.nii.gz"
run(["convertwarp",
     "--ref="   + str(RAW_MNI_T1),
     "--premat="+ str(epi_mat_fsl),      # b0 ➜ T1
     "--warp1=" + str(t1_to_mni_warp),   # T1 ➜ MNI
        "--relout",   
     "--out="   + str(b0_to_mni_warp)],
    outputs=[b0_to_mni_warp])             # final field: b0 ➜ MNI

print("\n--- 6. Apply FSL composite warp to streamlines ---")
print(f"[info] Using composite warp: {b0_to_mni_warp}")

# -------------------------------------------------------------------------
# 6.  Load the composite warp ONCE (4-D ΔxΔyΔz, LPI, relative)
# -------------------------------------------------------------------------

warp_img   = nb.load(b0_to_mni_warp)                # (X,Y,Z,3)
warp_data  = warp_img.get_fdata(dtype=np.float32)   # shrink RAM
A_warp_inv = np.linalg.inv(warp_img.affine)         # world-mm → voxel

def warp_points_lps_mm(points_mm):
    
    #points_mm : (N,3) float32, LPI mm in b0 space
    #returns   : (N,3) float32, LPI mm in MNI space
    
    # 1. affine → voxel coordinates of the warp image
    vox = nb.affines.apply_affine(A_warp_inv, points_mm).T  # (3, N)

    # 2. trilinear interpolation of the displacement field
    disp = np.vstack([map_coordinates(warp_data[..., c], vox,
                                      order=1, mode="nearest")
                      for c in range(3)]).T          # (N,3) Δmm

    # 3. add displacement (pure NumPy, broadcast)
    return points_mm + disp.astype(np.float32)

from concurrent.futures import ProcessPoolExecutor

max_workers = os.cpu_count()

def warp_chunk_numpy(chunk):
    return warp_points_lps_mm(chunk)

for vtk_path_original in TRACT_DIR.rglob("*_fibers_trimmed.vtk"):
    print(f"\nProcessing tract: {vtk_path_original.name}")
    vtk_out_mni = vtk_path_original.parent / (
        f"{vtk_path_original.stem}_fsl_icbm152_2009c.vtk")

    # -------- load tract once ----------------------------------------------
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(vtk_path_original))
    reader.Update()
    pts_ras = np.asarray(reader.GetOutput().GetPoints().GetData(), dtype=np.float32)

    # RAS → LPS
    pts_ras[:, :2] *= -1
    pts_lps_chunks = np.array_split(pts_ras, max_workers)

    # -------- parallel warp -------------------------------------------------
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        warped_chunks = list(ex.map(warp_chunk_numpy, pts_lps_chunks))

    pts_lps_new = np.vstack(warped_chunks)

    # LPS → RAS
    pts_lps_new[:, :2] *= -1

    # -------- write VTK -----------------------------------------------------
    vtk_arr = numpy_to_vtk(pts_lps_new, deep=1)
    vtk_arr.SetNumberOfComponents(3)
    vtk_pts = vtk.vtkPoints(); vtk_pts.SetData(vtk_arr)
    poly = reader.GetOutput(); poly.SetPoints(vtk_pts); poly.Modified()

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(vtk_out_mni))
    writer.SetInputData(poly)
    writer.Write()
    print(f"[✓] Streamlines saved to {vtk_out_mni}")

    # tidy
    del pts_ras, pts_lps_new, pts_lps_chunks, warped_chunks
    gc.collect()

print("\nAll streamlines processed successfully!")


# old no parallel fallback version

for vtk_path_original in TRACT_DIR.rglob("*_fibers_trimmed.vtk"):
    print(f"\nProcessing tract: {vtk_path_original.name}")
    vtk_out_mni = vtk_path_original.parent / f"{vtk_path_original.stem}_icbm152_2009c.vtk"

    csv_in_dwi = tmp / f"{vtk_path_original.stem}.csv"
    csv_out_mni = tmp / f"{vtk_path_original.stem}_icbm152_2009c.csv"

    # Load the VTK fiber tract file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(vtk_path_original))
    reader.Update()
    polydata = reader.GetOutput()
    points = polydata.GetPoints()
    print(f"Number of points in tract: {points.GetNumberOfPoints()}")

    # Convert points to a NumPy array (RAS coordinates as stored by Slicer)
    pts_ras = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])

    # Convert RAS -> LPS by flipping x and y
    pts_lps = pts_ras.copy()
    pts_lps[:,0] *= -1
    pts_lps[:,1] *= -1

    # Save to CSV (with header x,y,z,t,label as required by ANTs)
    with open(csv_in_dwi, "w") as f:
        f.write("x,y,z,t,label\n")
        for idx, (x,y,z) in enumerate(pts_lps):
            # use t=0 (no time dimension) and label as point index or 0
            f.write(f"{x:.6f},{y:.6f},{z:.6f},0,{idx}\n")

    # Warp the streamlines from native DWI space to MNI space in reverse order:
    run(["antsApplyTransformsToPoints", "-d", "3", "-i", f"{csv_in_dwi}", "-o", f"{csv_out_mni}",
        "-t", f"[{b0_to_t1_affine_mat},1]",
        "-t", f"{b0_to_t1_inv_warp}",
        "-t", f"[{t1_to_mni_affine_mat},1]",
        "-t", f"{t1_to_mni_inv_warp}"])

    print(f"[info] Transformed points saved to {csv_out_mni}")

    # Load transformed points (LPS) from CSV (skip header)
    pts_lps_new = np.loadtxt(csv_out_mni, delimiter=',', skiprows=1, usecols=(0,1,2))

    # Convert LPS -> RAS
    pts_ras_new = pts_lps_new.copy()
    pts_ras_new[:,0] *= -1
    pts_ras_new[:,1] *= -1

    # Update the polydata points with new RAS coordinates
    vtk_arr = numpy_to_vtk(
    pts_ras_new,      # copies only if deep=1
    deep=1,           # keep a real copy; avoids GC woes
    )
    vtk_arr.SetNumberOfComponents(3)

    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(vtk_arr)

    polydata.SetPoints(vtk_pts)
    polydata.Modified()

    # Save to a new VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(vtk_out_mni))
    writer.SetInputData(polydata)
    writer.Write()
    print(f"[info] Streamlines saved to {vtk_out_mni}")

print("\nAll streamlines processed successfully!")
"""