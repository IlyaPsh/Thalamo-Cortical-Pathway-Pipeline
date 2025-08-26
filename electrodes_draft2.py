import slicer
import numpy as np
import vtk
import os
from pathlib import Path
import subprocess, shlex
import pandas as pd 
import scipy.io as sio

def suppress_vtk_warnings():
    """Suppress VTK warnings globally."""
    warning_output = vtk.vtkFileOutputWindow()
    warning_output.SetFileName("NUL" if os.name == "nt" else "/dev/null")  # Redirect to null device
    vtk.vtkObject.SetGlobalWarningDisplay(0)
    vtk.vtkOutputWindow.SetInstance(warning_output)

def enable_vtk_warnings():
    """Re-enable VTK warnings globally."""
    vtk.vtkObject.SetGlobalWarningDisplay(1)

suppress_vtk_warnings() # Otherwise they spam the console output

# ============================================================================
# ## 0. SETUP AND CONFIGURATION ##
# ============================================================================

# -----------------------------------------------------------
#   0_A.  GLOBAL TOGGLE
# -----------------------------------------------------------
COMBINE_EVEN_ODD = True          # flip to False for even/odd files

# Bridge step - set to False since it isnt required
DO_BRIDGE = False

#   (same flag name that source_locate_v6.py already uses)

# -----------------------------------------------------------
#   0_B.  Switch on the flag - define EEG_STUFF & GROUPS
# -----------------------------------------------------------

if COMBINE_EVEN_ODD:   #combined "average" run
    EEG_STUFF = Path(r"C:\Users\2710i\OneDrive\Work\BachelorThesis\ssep_analysis_average")

    GROUPS = {
        # tag            .mat lives inside the session folder          components
        "S1_N20P20": (r"{session}/evokedNP20_{session}.mat", ("N20", "P20")),
        "S1_P30N30": (r"{session}/evokedNP30_{session}.mat", ("P30", "N30")),
        "S2_N20P20": (r"{session}/evokedNP20_{session}.mat", ("N20", "P20")),
        "S2_P30N30": (r"{session}/evokedNP30_{session}.mat", ("P30", "N30")),
    }

else: 
    EEG_STUFF = Path(r"C:\Users\2710i\OneDrive\Work\BachelorThesis\ssep_analysis_v2")

    GROUPS = {
        "S1_N20P20": (r"{session}/evokedNP20_{session}_even.mat", ("N20", "P20")),
        "S1_P30N30": (r"{session}/evokedNP30_{session}_odd.mat", ("P30", "N30")),
        "S2_N20P20": (r"{session}/evokedNP20_{session}_even.mat", ("N20", "P20")),
        "S2_P30N30": (r"{session}/evokedNP30_{session}_odd.mat", ("P30", "N30")),
    }

# Map "S1"/"S2" to full session folder names derived from EEG file stems
SESSION_MAP = {
    "S1": "RecordSession_1_2025.06.24_10.35.00",
    "S2": "RecordSession_2_2025.06.24_10.38.43",
}

# --- tuneable parameters ---
# For bridging gaps between fiber bundles
MAX_GAP_MM = 4.0      # Stitch only if endpoints are closer than this
MAX_ANGLE_DEG = 25.0     # and their outgoing tangents are within this angle

# For ROI definition
ROI_DILATE_MM = 0.5      # 0 = exact label, >0 = more

# base radius (mm) added to the |amplitude| scaling coming from radius_* arrays in the .mat
# ---- cylinder size andd colour master switches ----------------------
BASE_CYL_RADIUS  = 5.0       # mm added to every cylinder
RADIUS_RESCALE   = 0.50      # 1.0 = keep .mat radii, 0.7 = shrink all 30%
# ---------------------------------------------------------------------

# For electrode geometry
HEIGHT_CYL = 45.0       # Height of the cylindrical electrodes

# For the EP source sphere
# UNUSED
#RADIUS_SPHERE = 6.0      # Radius of the sphere representing the EP source

# --- File Paths ---

ATLAS_FIBERS_PATH = r"C:\Users\2710i\OneDrive\Work\BachelorThesis\DSI_TRACTS\warp\ground_truth_mni_c.vtk"

THALAMUS_ROI_PATH = r"C:\Users\2710i\OneDrive\Work\BachelorThesis\electrode_stuff\volumes\Thalamus.nii.gz"
MOTOR_CORTEX_ROI_PATH = r"C:\Users\2710i\OneDrive\Work\BachelorThesis\electrode_stuff\volumes\Motor.nii.gz"
# This function requires an internal capsule labelmap. 
INTERNAL_CAPSULE_ROI_PATH = r"C:\Users\2710i\OneDrive\Work\BachelorThesis\electrode_stuff\volumes\internal_capsule.nii.gz" #<-- ADD PATH
                                # already imported above

ELEC_DIR = Path(r"C:\Users\2710i\OneDrive\Work\BachelorThesis\electrode_stuff")

TRACT_DIR = Path(r"C:\Users\2710i\OneDrive\Work\TestV6\SLICER\masks")

def read_elec_table(fname: str, has_name: bool = True):
    """
    Returns three parallel lists: names (or None), positions, normals.
    The .txt files are white-space separated and have RAS + normal columns.
    """
    df = pd.read_csv(ELEC_DIR / fname, sep=r"\s+", engine="python")
    pos   = df[['R(mm)', 'A(mm)', 'S(mm)']].to_numpy().tolist()
    norms = df[['n_R',  'n_A',  'n_S']].to_numpy().tolist()
    names = df['name'].astype(str).tolist() if has_name else None
    return names, pos, norms

def load_polydata(filepath: str, nodename: str):
    """Loads a .vtk or .vtp file as a model node."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find file: {filepath}")
    node = slicer.util.loadFiberBundle(filepath)
    if not node:
        raise RuntimeError(f"Could not load model from: {filepath}")
    node.SetName(nodename)
    slicer.app.processEvents()
    print(f"Loaded PolyData: {nodename}")
    node.GetDisplayNode().SetVisibility(False)   # hide

    return node

def amplitude_to_radius(radius_scalar):
    """
    The .mat files already give you a 5-25 mm range (radius_* arrays).
    only rescale it globally here and add the constant base.
    """
    return BASE_CYL_RADIUS + RADIUS_RESCALE * radius_scalar

# --- 1. coloring helper (check again later) -------------------------------
def component_colour(amplitude):
    """Red for >=0 uV, blue for <0 uV."""
    amp = float(np.asarray(amplitude).ravel()[0])   #  unwrap array to scalar
    return (1.0, 0.0, 0.0) if amp >= 0 else (0.0, 0.0, 1.0)


def create_oriented_cylinder(electrodeRAS, normalRAS, radius_mm, height_mm):
    """Creates a vtkPolyData for a cylinder at a specific position and orientation."""
    cylinderSource = vtk.vtkCylinderSource()
    radius_mm = float(np.asarray(radius_mm).ravel()[0])
    cylinderSource.SetRadius(radius_mm)
    cylinderSource.SetHeight(height_mm)
    cylinderSource.SetResolution(36)

    transform = vtk.vtkTransform()
    transform.Translate(electrodeRAS) # Move to position first

    # Calculate rotation from default Y-axis to the target normal
    normal = np.array(normalRAS, dtype=float)
    if np.linalg.norm(normal) < 1e-6:
        normal = np.array([0, 1, 0]) # Default to Y-axis if normal is zero
    else:
        normal = normal / np.linalg.norm(normal)

    y_axis = np.array([0, 1, 0])
    rotation_axis = np.cross(y_axis, normal)
    if np.linalg.norm(rotation_axis) > 1e-6:
        angle_rad = np.arccos(np.clip(np.dot(y_axis, normal), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        transform.RotateWXYZ(angle_deg, rotation_axis)
    elif np.dot(y_axis, normal) < 0: # Aligned but opposite direction
        transform.RotateWXYZ(180, 1, 0, 0)

    # Translate along new axis to center the cylinder base at the electrode position halway
    transform.Translate(0, height_mm / 2, 0)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputConnection(cylinderSource.GetOutputPort())
    transformFilter.Update()
    return transformFilter.GetOutput()

def create_segmentation_from_cylinders(polydata_list, labels, colours, referenceVolume):
    segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    segNode.SetReferenceImageGeometryParameterFromVolumeNode(referenceVolume)

    for i, (pd, lbl, col) in enumerate(zip(polydata_list, labels, colours)):
        # 1. helper model (needed for the import call)
        m = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        m.SetAndObservePolyData(pd)

        # 2. import into the segmentation
        slicer.modules.segmentations.logic().ImportModelToSegmentationNode(m, segNode)

        # 3. colourise the segment (this is what the 3-D viewers show)
        segID = segNode.GetSegmentation().GetNthSegmentID(i)
        segNode.GetSegmentation().GetSegment(segID).SetName(str(lbl))
        segNode.GetSegmentation().GetSegment(segID).SetColor(*col)
        slicer.app.processEvents()

        slicer.mrmlScene.RemoveNode(m)            # tidy-up helper model

    segNode.GetDisplayNode().SetVisibility(False)
    print(f" {len(polydata_list)} segments imported into {segNode.GetName()}")
    return segNode

def export_segment_to_labelmap(segmentationNode, segmentId, referenceVolume, outputName):
    """Exports a single segment into a new binary labelmap volume."""
    outputLabelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", outputName)
    segmentIds = vtk.vtkStringArray()
    segmentIds.InsertNextValue(segmentId)
    slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(segmentationNode, segmentIds, outputLabelmap, referenceVolume)
    slicer.app.processEvents()
    return outputLabelmap

def dilate_labelmap(labelmapNode, radius_mm):
    if radius_mm <= 0:
        return labelmapNode

    # 1. label-map - segmentation (single segment)
    segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
        labelmapNode, segNode)

    # 2. headless Segment Editor
    paramNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segEditor = slicer.qMRMLSegmentEditorWidget()
    segEditor.setMRMLScene(slicer.mrmlScene)
    segEditor.setMRMLSegmentEditorNode(paramNode) 
    segEditor.setSegmentationNode(segNode)      
    segEditor.setSourceVolumeNode(labelmapNode)    

    segEditor.setActiveEffectByName("Margin")
    segEditor.activeEffect().setParameter("MarginSizeMm", float(radius_mm))
    segEditor.activeEffect().self().onApply()

    # 3. export back and clean-up
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
        segNode, labelmapNode)
    for n in (segEditor, paramNode, segNode):
        n = None  # drop refs / delete scene nodes later
    print(f"Dilated {labelmapNode.GetName()} by {radius_mm} mm.")
    return labelmapNode

def _endpoints_and_tangents_with_ids(polydata):
    """Extracts endpoints, tangents, and original point IDs for ref from a fiber bundle."""
    pts = vtk.vtkPoints()
    tans = []
    original_ids = []
    lines = polydata.GetLines()
    all_points = polydata.GetPoints()
    
    idl = vtk.vtkIdList()
    lines.InitTraversal()
    while lines.GetNextCell(idl):
        n = idl.GetNumberOfIds()
        if n < 2: continue
        
        # Tail endpoint
        p0_id, p1_id = idl.GetId(0), idl.GetId(1)
        p0, p1 = np.asarray(all_points.GetPoint(p0_id)), np.asarray(all_points.GetPoint(p1_id))
        t0 = p1 - p0; t0 /= (np.linalg.norm(t0) + 1e-6)
        pts.InsertNextPoint(p0); tans.append(t0); original_ids.append(p0_id)
        
        # Head endpoint
        q0_id, q1_id = idl.GetId(n-2), idl.GetId(n-1)
        q0, q1 = np.asarray(all_points.GetPoint(q0_id)), np.asarray(all_points.GetPoint(q1_id))
        t1 = q1 - q0; t1 /= (np.linalg.norm(t1) + 1e-6)
        pts.InsertNextPoint(q1); tans.append(t1); original_ids.append(q1_id)
        
    return pts, np.vstack(tans), original_ids

def bridge_between_bundles(bundleA_node, bundleB_node, max_gap_mm, max_ang_deg, out_name="Bridged_Fibers"):
    """
    IGNORE
    Combines two fiber bundles and creates plausible straight-line connections
    between the endpoints of A and B that are close and aligned.
    """
    pdA, pdB = bundleA_node.GetPolyData(), bundleB_node.GetPolyData()
    ptsA, tanA, idsA = _endpoints_and_tangents_with_ids(pdA)
    ptsB, tanB, idsB = _endpoints_and_tangents_with_ids(pdB)

    b_endpoint_polydata = vtk.vtkPolyData(); b_endpoint_polydata.SetPoints(ptsB)
    kdB = vtk.vtkKdTreePointLocator(); kdB.SetDataSet(b_endpoint_polydata); kdB.BuildLocator()

    appender = vtk.vtkAppendPolyData(); appender.AddInputData(pdA); appender.AddInputData(pdB); appender.Update()
    merged_pd = appender.GetOutput()
    
    cos_thr = np.cos(np.deg2rad(max_ang_deg))
    offsetB = pdA.GetNumberOfPoints()

    bridge_lines = vtk.vtkCellArray()
    for i in range(ptsA.GetNumberOfPoints()):
        p = np.asarray(ptsA.GetPoint(i))
        neighbors_vtk = vtk.vtkIdList()
        kdB.FindPointsWithinRadius(max_gap_mm, p, neighbors_vtk)

        for k in range(neighbors_vtk.GetNumberOfIds()):
            j = neighbors_vtk.GetId(k)
            q = np.asarray(ptsB.GetPoint(j))
            
            v_pq = q - p
            if np.dot(v_pq, v_pq) > max_gap_mm**2: continue
            
            v_norm = np.linalg.norm(v_pq)
            if v_norm < 1e-6: continue
            v_unit = v_pq / v_norm
            
            if np.dot(v_unit, tanA[i]) < cos_thr: continue
            if np.dot(-v_unit, tanB[j]) < cos_thr: continue
            
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, idsA[i])
            line.GetPointIds().SetId(1, idsB[j] + offsetB)
            bridge_lines.InsertNextCell(line)

    final_lines = vtk.vtkCellArray()
    final_lines.DeepCopy(merged_pd.GetLines())

    idl = vtk.vtkIdList()
    bridge_lines.InitTraversal()
    while bridge_lines.GetNextCell(idl):
        final_lines.InsertNextCell(idl) 

    merged_pd.SetLines(final_lines)

        
    out_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", out_name)
    out_node.SetAndObservePolyData(merged_pd)
    out_node.CreateDefaultDisplayNodes()
    slicer.app.processEvents()
    print(f"Bridged {bundleA_node.GetName()} and {bundleB_node.GetName()} with {bridge_lines.GetNumberOfCells()} new connections.")
    return out_node

def build_union_labelmap(base_lm, add_lm, name):

"""
Combine two thalamic segments into a labelmap
"""
    seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    seg.SetReferenceImageGeometryParameterFromVolumeNode(base_lm)
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(base_lm,  seg)
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(add_lm,   seg)

    out = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", name)
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(seg, out)
    return out

def select_through_tracks(fiber_node, roi_labelmap, labels="1,2"):

# check PAPER
    out_node = slicer.mrmlScene.AddNewNodeByClass(
                  "vtkMRMLFiberBundleNode",
                  f"{fiber_node.GetName()}_through")
    params = {
        "InputLabel_A":   roi_labelmap.GetID(),
        "InputFibers":    fiber_node.GetID(),
        "OutputFibers":   out_node.GetID(),
        "PassLabel":      labels,        # <- list!
        "PassOperation":  "AND",
        "SamplingDistance": "0.3"
    }
    slicer.cli.runSync(slicer.modules.fiberbundlelabelselect, None, params)
    return out_node

def select_through_tracks_by_pairs(fiber_node,
                                   roi_labelmap,
                                   label_pairs,
                                   out_name):
    """
    1) Runs "select_through_tracks()" for each (a,b) in "label_pairs"
       (with AND-logic -> fibres must hit both labels)
    2) Appends all resulting sub-bundles into a single bundle
    3) Removes the per-pair temporary nodes.
    """
    appender   = vtk.vtkAppendPolyData()
    temp_nodes = []

    for a, b in label_pairs:
        pair_lbls = f"{a},{b}"
        sub       = select_through_tracks(fiber_node,
                                          roi_labelmap,
                                          labels=pair_lbls)
        temp_nodes.append(sub)
        appender.AddInputData(sub.GetPolyData())

    appender.Update()

    merged = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLFiberBundleNode",
                                                out_name)
    merged.SetAndObservePolyData(appender.GetOutput())
    merged.CreateDefaultDisplayNodes()
    slicer.app.processEvents()


    for n in temp_nodes:
        slicer.mrmlScene.RemoveNode(n)


    merged.GetDisplayNode().SetVisibility(True)
    print(f"{out_name}: kept {merged.GetPolyData().GetNumberOfLines()} fibres")
    return merged


# --- Analysis and Metrics ---

def _label_at_ras(ras_xyz, labelmap_node, ijk_to_ras_matrix, arr):
    """Fast lookup of a label value at a given RAS coordinate."""
    ras_h = list(ras_xyz) + [1]
    ijk_h = ijk_to_ras_matrix.MultiplyPoint(ras_h)
    ijk = [int(round(c)) for c in ijk_h[:3]]
    
    dims = arr.shape
    if 0 <= ijk[2] < dims[0] and 0 <= ijk[1] < dims[1] and 0 <= ijk[0] < dims[2]:
        return int(arr[ijk[2], ijk[1], ijk[0]])
    return 0

def summarise_nuclei(fibers_node, thal_lm, capsule_lm):
    """Counts fiber points within thalamic nuclei and checks for internal capsule intersection."""
    counts, capsule_fibers = {}, 0
    pd = fibers_node.GetPolyData()
    if pd.GetNumberOfLines() == 0:
        return counts, capsule_fibers
        
    # Pre-fetch arrays and matrices for speed
    thal_arr = slicer.util.arrayFromVolume(thal_lm)
    capsule_arr = slicer.util.arrayFromVolume(capsule_lm)
    ras_to_ijk_mat = vtk.vtkMatrix4x4(); thal_lm.GetRASToIJKMatrix(ras_to_ijk_mat)

    lines = pd.GetLines(); lines.InitTraversal()
    idl = vtk.vtkIdList()
    while lines.GetNextCell(idl):
        hit_capsule_this_fiber = False
        for i in range(idl.GetNumberOfIds()):
            ras = pd.GetPoint(idl.GetId(i))
            lbl = _label_at_ras(ras, thal_lm, ras_to_ijk_mat, thal_arr)
            if lbl:
                counts[lbl] = counts.get(lbl, 0) + 1
            if not hit_capsule_this_fiber and _label_at_ras(ras, capsule_lm, ras_to_ijk_mat, capsule_arr):
                hit_capsule_this_fiber = True
        if hit_capsule_this_fiber:
            capsule_fibers += 1
            
    return counts, capsule_fibers

def basic_metrics(sel_node):
    """Calculates number of fibers and median fiber length."""
    pd = sel_node.GetPolyData()
    n_fib = pd.GetNumberOfLines()
    if n_fib == 0:
        return 0, 0.0

    lengths = []
    lines = pd.GetLines(); lines.InitTraversal()
    idl = vtk.vtkIdList()
    while lines.GetNextCell(idl):
        num_points = idl.GetNumberOfIds()
        if num_points < 2: continue
        
        length = 0.0
        for i in range(num_points - 1):
            p1 = np.array(pd.GetPoint(idl.GetId(i)))
            p2 = np.array(pd.GetPoint(idl.GetId(i+1)))
            length += np.linalg.norm(p2 - p1)
        lengths.append(length)

    return n_fib, np.median(lengths)

def resample_to_match(srcLabelMap, refLabelMap, out_name):
    """
    NOT USED
    """
    """Nearest-neighbour resample of src so it matches ref geometry."""
    outLM = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", out_name)
    p = {
        "inputVolume":      srcLabelMap.GetID(),
        "referenceVolume":  refLabelMap.GetID(),
        "outputVolume":     outLM.GetID(),
        "interpolationType":"nearestNeighbor"
    }
    slicer.cli.runSync(slicer.modules.resamplescalarvolume, None, p)
    return outLM

# ============================================================================
# ## 2. MAIN EXECUTION ##
# ============================================================================

# 1) channel geometry (identical for all groups) - read once
cap_names, cap_pos, cap_norm = read_elec_table("electrodes_RAS_normals.txt")
cap_norm = [[-x,-y,-z] for x,y,z in cap_norm]          # flipthe normals to point inwards

NAME2IDX   = {n:i for i,n in enumerate(cap_names)}     # quick lookup later
ALL_POS    = np.asarray(cap_pos)                       # list to ndarray for slicing
ALL_NORM   = np.asarray(cap_norm)

# 2) discover the strongest-*.txt files

strongest_paths = list(Path(EEG_STUFF).rglob("strongest_*.txt"))

# {(grpTag, comp) : electrodeName}
STRONGEST = {}
for p in strongest_paths:
    comp       = p.stem.split("_")[1]                 # N20 / P20 / ...
    folder_tag = p.parent.name.split("_")[:2]         # e.g. RecordSession_1_..._even -> ['RecordSession','1']
    sess_tag   = "S1" if folder_tag[1] == "1" else "S2"
    grp_tag    = f"{sess_tag}_{'N20P20' if comp in ('N20','P20') else 'P30N30'}"
    STRONGEST[(grp_tag, comp)] = pd.read_csv(p, sep=r"\s+").iloc[0]["name"]

# quick print
for k,v in STRONGEST.items():
    print(f" strongest {k[1]} in {k[0]} = {v}")

# Load Data
print("\n--- 2a. Loading Data ---")
hcp_node = load_polydata(ATLAS_FIBERS_PATH, "HCP_Fibers")

# --- (1) WHOLE-BRAIN bundle --------------------------------------------------
wb_path  = TRACT_DIR / "whole_brain_fibers_trimmed_icbm152_2009c.vtk"
wb_node  = load_polydata(str(wb_path), "Whole_Brain")

# --- (2) all *_fibers_trimmed_icbm152_2009c.vtk -----------------------------
trimmed_nodes = []
for vtk_path in TRACT_DIR.rglob("*_fibers_trimmed_icbm152_2009c.vtk"):
    # Exclude .vtp files and the whole brain file
    if vtk_path.suffix == ".vtk" and "whole_brain" not in vtk_path.name:
        trimmed_nodes.append(load_polydata(str(vtk_path), vtk_path.stem))

app = vtk.vtkAppendPolyData()
for n in trimmed_nodes:
    app.AddInputData(n.GetPolyData())
app.Update()
slicer.app.processEvents()

trimmed_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLFiberBundleNode",
                                                  "Merged_Trimmed")
trimmed_node.SetAndObservePolyData(app.GetOutput())
trimmed_node.CreateDefaultDisplayNodes()

for n in trimmed_nodes:
    slicer.mrmlScene.RemoveNode(n)
trimmed_nodes.clear()

# CHANGE IF NEEDED
patient_node = trimmed_node

thalamus_label = slicer.util.loadLabelVolume(THALAMUS_ROI_PATH)
motor_label = slicer.util.loadLabelVolume(MOTOR_CORTEX_ROI_PATH)
capsule_label = slicer.util.loadLabelVolume(INTERNAL_CAPSULE_ROI_PATH)
if not all([thalamus_label, motor_label, capsule_label]):
    raise RuntimeError("Failed to load one or more required ROI labelmaps.")

slicer.app.processEvents()

print("\n--- 2b. Creating ALL-channel geometry (visual) "
      "and STRONGEST‑only geometry (analysis) ---")

vis_seg_nodes  = {}        # every channel (for display only)
roi_seg_nodes  = {}        # just the strongest 2 (for ROI analysis)

for grp_tag, (mat_tpl, comps) in GROUPS.items():

    # ------------------------------------------------------------------
    #Load the .mat file that belongs to this group (session/condition)
    # ------------------------------------------------------------------
    session_tag = grp_tag[:2]          # "S1" or "S2"
    session     = SESSION_MAP[session_tag]

    mat_path = EEG_STUFF / mat_tpl.format(session=session)
    mat      = sio.loadmat(mat_path)

    raw_names = mat["channel_names"].ravel()
    mat_lookup = {n.upper(): i for i, n in
                enumerate([n.decode() if isinstance(n, bytes) else str(n) for n in raw_names])}

    MAT_IDX = [mat_lookup.get(n.upper()) for n in cap_names]   # None if not present

    # ----------------------------------------------------------------------

    # matrices from the .mat
    ampA, radA = mat[f"amp_{comps[0]}"].ravel(),    mat[f"radius_{comps[0]}"].ravel()
    ampB, radB = mat[f"amp_{comps[1]}"].ravel(),    mat[f"radius_{comps[1]}"].ravel()

    # --------- 1. build cylinders for all channels (visual only) ----
    all_poly  = []
    all_lbls  = []
    all_cols  = []
    for idx, ch_name in enumerate(cap_names):
        mat_idx = MAT_IDX[idx]
        if mat_idx is None:
            continue                    # not in the .mat file

        if np.isnan(radA[mat_idx]) and np.isnan(radB[mat_idx]):
            continue
        ampA_val, radA_val = ampA[mat_idx], radA[mat_idx]
        ampB_val, radB_val = ampB[mat_idx], radB[mat_idx]

        if abs(ampA_val) >= abs(ampB_val):
            amp, rad = ampA_val, radA_val
        else:
            amp, rad = ampB_val, radB_val
        colour = component_colour(amp)


        cyl = create_oriented_cylinder(
                electrodeRAS = ALL_POS[idx],
                normalRAS    = ALL_NORM[idx],
                radius_mm    = amplitude_to_radius(rad),
                height_mm    = HEIGHT_CYL)

        all_poly.append(cyl)
        all_lbls.append(f"{grp_tag}_{ch_name}")
        all_cols.append(component_colour(amp))

    if all_poly:
        vis_seg_nodes[grp_tag] = create_segmentation_from_cylinders(
                                     all_poly, all_lbls, all_cols,
                                     referenceVolume = thalamus_label)
        vis_seg_nodes[grp_tag].SetName(f"{grp_tag}_ALL")

    # --------- 2. cylinders for the two strongest electrodes only ----
    roi_poly, roi_lbls, roi_cols = [], [], []
    for comp in comps:                         # exactly two components
        key = (grp_tag, comp)
        if key not in STRONGEST:
            continue
        ch   = STRONGEST[key]
        idx      = NAME2IDX[ch]
        mat_idx  = MAT_IDX[idx]          
        amp      = float(np.asarray(mat[f"amp_{comp}"][mat_idx]).ravel()[0])
        rad      = mat[f"radius_{comp}"][mat_idx]


        cyl = create_oriented_cylinder(
                electrodeRAS = ALL_POS[idx],
                normalRAS    = ALL_NORM[idx],
                radius_mm    = amplitude_to_radius(rad),
                height_mm    = HEIGHT_CYL)

        roi_poly.append(cyl)
        roi_lbls.append(f"{grp_tag}_{comp}_{ch}")
        roi_cols.append(component_colour(amp))
        slicer.app.processEvents()

    if roi_poly:
        roi_seg_nodes[grp_tag] = create_segmentation_from_cylinders(
                                     roi_poly, roi_lbls, roi_cols,
                                     referenceVolume = thalamus_label)
        roi_seg_nodes[grp_tag].SetName(f"{grp_tag}_STRONGEST")


slicer.app.processEvents()

UNION_MASKS = []          # list of (maskTag, labelMapNode)

for grp_tag, seg in roi_seg_nodes.items():
    n_segs = seg.GetSegmentation().GetNumberOfSegments()
    for i in range(n_segs):
        seg_id   = seg.GetSegmentation().GetNthSegmentID(i)
        seg_name = seg.GetSegmentation().GetSegment(seg_id).GetName()  # readable name e.g. "S1_N20P20_N20_CP3"

        cyl_lm = export_segment_to_labelmap(
            segmentationNode = seg,
            segmentId        = seg_id,
            referenceVolume  = thalamus_label,
            outputName       = f"{seg.GetName()}_{i}_LM")

        UNION_MASKS.append((seg_name, build_union_labelmap(
            thalamus_label,
            cyl_lm,
            name=f"{seg_name}_ThalUnion")))

        slicer.app.processEvents()

# --- build the patient bundle (bridged or raw) ----------------------
if DO_BRIDGE:
    bridged_fibers = bridge_between_bundles(
        patient_node, patient_node,
        MAX_GAP_MM, MAX_ANGLE_DEG,
        out_name="Tracula_Bridged")
    patient_bundle_node = bridged_fibers
else:
    patient_bundle_node = trimmed_node        # original "Merged_Trimmed"


bundle_sets = [("HCP Atlas",       hcp_node),
               ("UKF Whole Brain", wb_node),
               ("Tracula Merged",  patient_bundle_node)]


for _, node in bundle_sets:
    node.GetDisplayNode().SetVisibility(False)

metrics_table = []   # rows: dicts and later pd.DataFrame

# Which label-pairs should a fibre simultaneously pass? 
THAL_ELEC_PAIRS = [("3", "1"),   # electrode / EP          thalamus-part-1
                   ("3", "2")]   # electrode / EP          thalamus-part-2
# (Edit the numbers here if your label values change !!!!!)

# Will hold the results exactly like before
selected_bundles = {}   # key = (bundleTag, maskTag)  →  fibre-node

for maskTag, maskLM in UNION_MASKS:
    print(f"\nSelecting fibres that pass through {maskTag} "
          f"using label pairs {THAL_ELEC_PAIRS}")
    for tag, bundle in bundle_sets:
        sel_node = select_through_tracks_by_pairs(
                       fiber_node   = bundle,
                       roi_labelmap = maskLM,
                       label_pairs  = THAL_ELEC_PAIRS,
                       out_name = f"{tag.replace(' ', '_')}_{maskTag}_through")
        
        nFib, medLen = basic_metrics(sel_node)
        nucCnt, inCaps = summarise_nuclei(sel_node,
                                          thalamus_label,
                                          capsule_label)

        metrics_table.append({
            "Group":      maskTag.split("_")[0],            # S1 / S2
            "Comp":       maskTag.split("_")[1][:3],        # N20 / P20 / ...
            "Electrode":  maskTag.split("_")[-1],           # CP3, Fz, ...
            "Bundle":     tag,                              # HCP Atlas / ...
            "Fibres":     nFib,
            "MedianLen":  medLen,
            "InCapsule":  inCaps,
            "ThalHits":   sum(nucCnt.values())
        })

        n = sel_node.GetPolyData().GetNumberOfLines()
        print(f"  • {tag:<20s}: {n:6d} fibres")
        selected_bundles[(tag, maskTag)] = sel_node

enable_vtk_warnings()

# safely tripwire
try:
    first_mask = next(k for _,k in selected_bundles)
except StopIteration:
    raise RuntimeError("no bundles selected - check earlier steps")

hcp_through  = selected_bundles[("HCP Atlas",  first_mask)]
wb_through   = selected_bundles[("UKF Whole Brain",  first_mask)]
trim_through = selected_bundles[("Tracula Merged",   first_mask)]

# --- 2f. Final Metrics and Reporting ---
print("\n--- 2f. Final Metrics and Reporting ---")
hcp_N,  hcp_med  = basic_metrics(hcp_through)
wb_N,   wb_med   = basic_metrics(wb_through)
trim_N, trim_med = basic_metrics(trim_through)

print(f"\nAtlas (HCP)       : {hcp_N} fibers, median length {hcp_med:.1f} mm")
print(f"Patient whole-brain: {wb_N} fibers, median length {wb_med:.1f} mm")
print(f"Patient trimmed    : {trim_N} fibers, median length {trim_med:.1f} mm\n")
slicer.app.processEvents()

df_metrics = pd.DataFrame(metrics_table)
csv_out    = Path(ELEC_DIR) / "thalamus_connectivity_scores.csv"
df_metrics.to_csv(csv_out, index=False)
print(f"\n Connectivity metrics written to {csv_out}")

df_metrics["Score"] = (
      df_metrics["Fibres"]
    + 5 * df_metrics["InCapsule"]
    + 0.5 * df_metrics["MedianLen"])

print(df_metrics.groupby(["Group","Comp"])[["Score"]].sum())

print("\n\nPipeline finished.")