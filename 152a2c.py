#!/usr/bin/env python3
from pathlib import Path
import subprocess, shlex
import vtk, numpy as np
import os
import tempfile, gc, shutil
from vtk.util.numpy_support import numpy_to_vtk
from concurrent.futures import ProcessPoolExecutor

"""

FINAL ATLAS FIBRES COREGISTRATION TO 2009c MNI

"""

overwrite = True  # Force overwrite for all steps !!!!!!

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

# ---------- top-level helpers ------------------------------------------------
a2c_affine = None        # will be filled inside main()
a2c_iwarp  = None

def warp_chunk(csv_pair):
    """Run antsApplyTransformsToPoints on one <in, out> CSV pair."""
    cin, cout = csv_pair
    run(["antsApplyTransformsToPoints", "-d", "3",
         "-i", cin, "-o", cout,
         "-t", f"[{a2c_affine},1]",
         "-t", a2c_iwarp],
        quiet=True)
    return cout

    
def main():

    nthreads = str(os.cpu_count())
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = nthreads  # already str()

    # -------- TEMP DIR PATH ----------
    tmp = Path("/mnt/c/Users/2710i/OneDrive/Work/BachelorThesis/DSI_TRACTS/warp")
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"[info] working dir: {tmp}")
    #  ---------------------------------

    # ----------- PATHS --------------
    VTK_IN = Path("/mnt/c/Users/2710i/OneDrive/Work/BachelorThesis/DSI_TRACTS/warp/whole_brain_mni_downsampled_qaaligned.vtk")
    VTK_OUT = Path("/mnt/c/Users/2710i/OneDrive/Work/BachelorThesis/DSI_TRACTS/warp/ground_truth_mni_c.vtk")
    MNI_A = Path("/mnt/c/Users/2710i/OneDrive/Work/BachelorThesis/DSI_TRACTS/mni_icbm152_nlin_asym_09a/mni_icbm152_t1_tal_nlin_asym_09a.nii")
    MNI_C = Path("/mnt/c/Users/2710i/OneDrive/Work/BachelorThesis/DSI_TRACTS/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii")
    #  -------------------------------
    # Register b0 to T1 using SyN
    ants_prefix_str = str(tmp / "MNI_A_to_MNI_C")
    a_to_c_img = Path(f"{ants_prefix_str}Warped.nii.gz")
    a_to_c_fwd_warp = Path(f"{ants_prefix_str}1Warp.nii.gz")
    a_to_c_inv_warp = Path(f"{ants_prefix_str}1InverseWarp.nii.gz")
    a_to_c_affine_mat = Path(f"{ants_prefix_str}0GenericAffine.mat")
    # PS those are names ants automatically gives them

    run(["antsRegistrationSyN.sh", "-d", "3",
        "-f", str(MNI_C), "-m", str(MNI_A),
        "-o", ants_prefix_str, "-n", nthreads],
        outputs=[a_to_c_affine_mat, a_to_c_img, a_to_c_fwd_warp, a_to_c_inv_warp])

    global a2c_affine, a2c_iwarp
    a2c_affine = str(a_to_c_affine_mat)
    a2c_iwarp  = str(a_to_c_inv_warp)

    print("\n--- 3. Apply transforms to streamlines ---")
    print(f"[info] Applying transforms (ICBM 152 2009a → ICBM 152 2009c):")
    print(f"       {a_to_c_affine_mat} (inv)")
    print(f"       {a_to_c_inv_warp}")


    max_workers = os.cpu_count()        # how many chunks / processes
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"   # avoid over-subscription
    #improve multi - threading later
    
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(VTK_IN))
    reader.Update()
    pts_ras = np.asarray(reader.GetOutput().GetPoints().GetData())

    # RAS ➜ LPS
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

    csv_pairs = list(zip(csv_in_list, csv_out_list))
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for _ in ex.map(warp_chunk, csv_pairs):
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
    writer.SetFileName(str(VTK_OUT));  writer.SetInputData(poly);  writer.Write()
    print(f"[✓] Streamlines saved to {VTK_OUT}")

    # ---------- clean up temp files & be nice to RAM --------------------------
    shutil.rmtree(tmpdir, ignore_errors=True)
    del pts_ras, pts_lps_new, pts_lps_chunks, chunks_warped
    gc.collect()

if __name__ == "__main__":
    main()
