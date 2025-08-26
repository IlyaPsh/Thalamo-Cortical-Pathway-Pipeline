#!/usr/bin/env python
from __future__ import annotations

# stdlib
import os
import re
from pathlib import Path

# 3rd-party
import matplotlib.pyplot as plt
from datetime import datetime          # just for filename tags
import mne
import numpy as np
import pandas as pd
import pyvista as pv
from lxml import etree as ET
from mne.bem import make_watershed_bem
from scipy.stats import pearsonr
from scipy.io import loadmat
import shutil
from scipy.io import savemat
from mne.transforms import apply_trans
from pyvista import PolyData

try:
    import ipywidgets as widgets
    from IPython.display import display
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False

# =========================================================================
# === 1. SCRIPT CONTROL PANEL (PARAMETERS) ================================
# =========================================================================
PARAMS = {

    "COMBINE_EVEN_ODD": True,   # ← flip to True to merge odd+even trials
    # For combined even/odd
    "OUTPUT_DIR_AVG": Path("/mnt/c/Users/2710i/OneDrive/Work/BachelorThesis/ssep_analysis_average"),

    # --- FILE PATHS & METADATA ---
    "EEG_FOLDER": Path("/mnt/c/Users/2710i/OneDrive/Work/gRecorder/gConverter/ConvertedFiles"),
    "EEG_FILES": [
        "RecordSession_1_2025.06.24_10.35.00.EDF",
        "RecordSession_2_2025.06.24_10.38.43.EDF",
    ],
    "TRANS_FILE": Path("/mnt/c/Users/2710i/OneDrive/Work/BachelorThesis/ssep_analysis_v2/mni_subject-trans.fif"),
    "MONTAGE_XML": Path("/mnt/c/Users/2710i/OneDrive/Work/gRecorder/Ilya.xml"),
    "BRAIN_XML": Path("/mnt/c/Users/2710i/OneDrive/Work/gRecorder/BrainModel_MNI_1005.xml"),
    "FS_SUBJECT": "mni_subject",
    #For not combined even/odd
    "OUTPUT_DIR": Path("/mnt/c/Users/2710i/OneDrive/Work/BachelorThesis/ssep_analysis_v2"),
    "MANUAL_BAD_CHANNELS": ["TP9", "FC4", "FC6","AF4", "FC3", "F5", "P7", "TP10", "O2", "NFp2h"], 


    "POSTERIOR_CH": ['P', 'PO', 'O', 'TP', 'CP'],
    "ANTERIOR_CH":  ['F', 'AF', 'FP', 'FC', 'NFp', 'C', 'Cz'],

    "STIMULATED_HEMISPHERE": "right",  # Options: "right" or "left"

    # --- PREPROCESSING TOGGLES ---
    "USE_ICA": False,
    "USE_AUTOREJECT": True, 

    # --- FILTERING PARAMETERS (Hz) ---
    "HIGH_PASS_ANALYSIS": 10.0,
    "LOW_PASS_ANALYSIS": 300.0,

    # --- EPOCHING & EVENT PARAMETERS ---
    "STIM_OFFSET_S": 0.020,
    "TMIN": -0.20,
    "TMAX": 0.20,
    "BASELINE": (-0.010, -0.005),
    "ANNOTATION_REGEX": r'HA-2024\.02\.02.*',

    # --- STRONGEST ELECTRODE LOGIC PARAMETERS ---
    "N20_WINDOW": (0.019, 0.021),
    "P30_WINDOW": (0.030, 0.032),

    "P20_WINDOW": (0.019, 0.021),
    "N30_WINDOW": (0.030, 0.032),

    # --- SOURCE LOCALIZATION PARAMETERS ---
    "BEM_SPACING": "oct6",
    "INVERSE_METHOD": "eLORETA",
    "INVERSE_LAMBDA2": 1.0 / 2**2, # x**2, where x is SNR
    "INVERSE_LOOSE": 'auto',       # Constrain sources to cortex or 'auto'
    "INVERSE_DEPTH": 0.8,       # Standard depth weighting

    # --- MISC ---
    "MRI_CRAS_OFFSET": np.array([0.5, -17.5, 18.5]),
}

PARAMS["FIDUCIALS_MAT"] = PARAMS["OUTPUT_DIR"] / "fiducials.mat"


# =========================================================================
# === 2. HELPER FUNCTIONS =================================================
# =========================================================================

def load_ft_fiducials(mat_path: Path,
                      *,
                      coord_frame: str = "mri",
                      units: str = "mm") -> dict:
    """
    Read positions of NAS, LPA, RPA from a FieldTrip *.mat* file that contains
    a single variable ``elec`` with the fields

        elec['elecpos'] → (N, 3)  XYZ values
        elec['label']   → (N,)    strings like 'NAS', 'LPA', …

    Returns a dictionary that can be passed straight to
    ``mne.channels.make_dig_montage``.
    """
    mat  = loadmat(mat_path, simplify_cells=True)
    elec = mat.get("elec")
    if elec is None:
        raise ValueError(f"{mat_path} does not contain an 'elec' variable")

    pos    = np.asarray(elec["elecpos"])       # (N, 3) float64
    pos += PARAMS["MRI_CRAS_OFFSET"]
    labels = np.asarray(elec["label"]).astype(str).ravel()

    lookup = {}
    for lab, xyz in zip(labels, pos):
        key = lab.strip().lower()
        if key in ("nas", "nasion"):
            lookup["nasion"] = xyz
        elif key == "lpa":
            lookup["lpa"] = xyz
        elif key == "rpa":
            lookup["rpa"] = xyz

    missing = {"nasion", "lpa", "rpa"} - lookup.keys()
    if missing:
        raise ValueError(f"Missing fiducials in {mat_path}: {', '.join(missing)}")

    # convert to metres if necessary
    if units.lower() == "mm":
        for k in ("nasion", "lpa", "rpa"):
            lookup[k] = lookup[k] / 1000.0

    lookup["coord_frame"] = coord_frame   # usually 'mri' for surface‑RAS
    return lookup

def get_or_create_coregistration(
    raw: mne.io.BaseRaw,
    subject: str,
    subjects_dir: str,
    trans_fname: Path,
) -> mne.transforms.Transform:
    """
    Load an existing *-trans.fif* or launch the coregistration GUI and
    block execution until the user saves and closes the window.
    """
    if trans_fname.exists():
        print(f"Using existing coreg file: {trans_fname}")
        return mne.read_trans(trans_fname)

    # ------------------------------------------------------------------
    # 1.  Build the path to pass via `inst=` (MNE 1.9 or higher check
    # ------------------------------------------------------------------
    inst_path = Path(raw.filenames[0])           # EDF/FIF file you just read
    print(f" No trans file launching GUI for {inst_path.name}")

    mne.gui.coregistration(
        inst=str(inst_path),                     # **path**, not Raw
        subject=subject,
        subjects_dir=subjects_dir,
        block=True,                              # halts until closed
        show=True,
    )

    # ------------------------------------------------------------------
    # 2.  Load transform
    # ------------------------------------------------------------------
    if not trans_fname.exists():
        raise FileNotFoundError(
            f"The GUI was closed but {trans_fname} was not written."
        )

    print(f"Coregistration done – reading {trans_fname.name}")
    return mne.read_trans(trans_fname)

def add_interactive_stc_report(report, stc, subjects_dir, subject, title):
    """
    IGNORE
    """
    import tempfile
    import os
    import mne
    import pyvista as pv
    import panel as pn

    # 1. Configure panel and the 3D backend
    # This tells panel how to handle VTK objects and make them responsive.
    pn.extension('vtk', sizing_mode='stretch_width')
    mne.viz.set_3d_backend("pyvistaqt")

    # 2. Build the Brain object with a specified window size
    brain = stc.plot(
        subject=subject,
        subjects_dir=subjects_dir,
        hemi="split",
        views="lat",
        cortex="low_contrast",
        clim=dict(kind='percent', lims=[99, 99.5, 99.9]),
        smoothing_steps=2,
        time_viewer=True, # This enables the time-series data
        show_traces=True,
        size=(800, 400),
        verbose=True,
    )

    # 3. Create an embeddable panel pane from the plot
    plotter = getattr(brain, "plotter", None) or brain._renderer.plotter
    vtk_pane = pn.pane.VTK(
        plotter.ren_win,
        width=plotter.window_size[0],
        height=plotter.window_size[1]
    )

    # 4. Save the pane to a temporary file with embed=True
    # This creates a self-contained HTML fragment, not a conflicting full page.
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as tmp:
        temp_fname = tmp.name
    
    vtk_pane.save(temp_fname, embed=True)
    
    with open(temp_fname, 'r') as f:
        html_str = f.read()
    
    os.remove(temp_fname)

    # 5. Add the clean HTML fragment to the report
    report.add_html(
        html_str,
        section="Interactive Plots",
        title=title,
    )

    brain.close()
# =========================================================================
# IGNORE
# =========================================================================
def add_pyvista_scene(report, brain, section, title):
    """
    Export the *current* scene from a PyVista Brain viewer and attach it
    to a report section.  Keeps the original static-image behaviour.
    """
    plotter = getattr(brain, "plotter", None) or brain._renderer.plotter
    html_str = plotter.export_html(None, return_html=True, autoplay=False)
    report.add_html(html_str, section=section, title=title)


def get_hemisphere_channels(all_ch_names: list[str], hemisphere: str) -> list[str]:
    """
    Filters channel names to include only those relevant for a given hemisphere stimulation.

    - 'right' stimulation: Includes midline ('z') and contralateral (odd-numbered) channels.
    - 'left' stimulation: Includes midline ('z') and contralateral (even-numbered) channels.
    """
    if hemisphere not in ["right", "left"]:
        print(f"Warning: Invalid hemisphere '{hemisphere}'. Returning all channels.")
        return all_ch_names

    print(f"Filtering channels for '{hemisphere}' hemisphere stimulation...")
    hemisphere_channels = []
    
    # Use regex to find numbers at the end of a label
    digit_pattern = re.compile(r'(\d+)$')

    for name in all_ch_names:
        name_upper = name.upper()
        
        # midline electrodes in both cases
        if name_upper.endswith('z'):
            hemisphere_channels.append(name)
            continue
            
        match = digit_pattern.search(name_upper)
        if match:
            number = int(match.group(1))
            # For RIGHT stimulation, keep ODD numbers (contralateral left hemisphere)
            if hemisphere == "right" and number % 2 != 0:
                hemisphere_channels.append(name)
            # For LEFT stimulation, keep EVEN numbers (contralateral right hemisphere)
            elif hemisphere == "left" and number % 2 == 0:
                hemisphere_channels.append(name)

    print(f"Found {len(hemisphere_channels)} relevant channels for this hemisphere.")
    return hemisphere_channels


def save_brain_panel(stc, views, fname, peak_time, subject, subjects_dir):
"""
IGNORE
"""

    mne.viz.set_3d_backend("pyvistaqt")
    brain = stc.plot(
        subject=subject, subjects_dir=subjects_dir, hemi='split',
        time_viewer=True, initial_time=peak_time,
        cortex='low_contrast', 
        clim=dict(kind='percent', lims=[99, 99.5, 99.9]),
        smoothing_steps=2, size=(1200,700)
    )
    for view in views:
        brain.show_view(view)
        img = brain.screenshot()
        plt.imsave(f"{fname}_{view}.png", img)
    brain.close()


def _vertex_normals(verts, faces):
    """
    IGNORE UNUSED
    """
    """Computes vertex normals for a surface mesh."""
    v_norm = np.zeros((verts.shape[0], 3))
    tri_n = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    for i in range(3):
        np.add.at(v_norm, faces[:, i], tri_n)
    v_norm /= np.linalg.norm(v_norm, axis=1, keepdims=True) + 1e-16
    return v_norm

def prepare_freesurfer():
    """Sets the FreeSurfer SUBJECTS_DIR environment variable. """
    subj_dir = Path("/mnt/c/FreeSurfer/freesurfer/subjects")
    os.environ["SUBJECTS_DIR"] = str(subj_dir)
    print(f"FreeSurfer environment ready (SUBJECTS_DIR set).")
    return subj_dir


def _auto_map(raw_names: list[str], target_names: list[str]):
    """Maps raw channel names to target names based on channel number."""
    digit_pattern = re.compile(r'\d+')
    mapping = {}
    for raw_name in raw_names:
        match = digit_pattern.search(raw_name)
        if match:
            ch_idx = int(match.group())
            if 1 <= ch_idx <= len(target_names):
                mapping[raw_name] = target_names[ch_idx - 1]
    print(f"Automatically generated channel map for {len(mapping)} channels.")
    return mapping

def load_brainmodel_electrodes(xml_file: Path):

    """Parses Brain.xml to get accurate RAS positions and normals. Check Paper for convention"""

    print(f"Loading accurate electrode positions from {xml_file.name}...")

    parser = ET.XMLParser(huge_tree=True, recover=True)

    root = ET.parse(xml_file, parser).getroot()

    lookup = {}

    for ele in root.iterfind(".//Electrode"):

        name = (ele.get("name") or ele.findtext("Label") or "").strip().upper()

        if not name: continue

        pos = np.fromstring((ele.findtext("./Positions/MNI/Head") or ""), sep=",")

        if pos.size != 3: continue

        pos += PARAMS["MRI_CRAS_OFFSET"]

        nrm = np.fromstring((ele.findtext("./Normals/MNI/Head") or ""), sep=",")

        if nrm.size != 3: nrm = pos / (np.linalg.norm(pos) + 1e-16)

        else: nrm /= np.linalg.norm(nrm) + 1e-16

        lookup[name] = (pos, nrm)

    print(f"✔ Found {len(lookup)} positions in Brain.xml.")

    return lookup 

def create_montage_and_df(channel_names: list[str],
                          brain_lookup: dict,
                          *,
                          fiducials: dict | None = None):
    
    ch_data = []
    for name in channel_names:
        if name.upper() in brain_lookup:
            pos, nrm = brain_lookup[name.upper()]
            ch_data.append({
                "name": name,
                "R(mm)": pos[0], "A(mm)": pos[1], "S(mm)": pos[2],
                "n_R":   nrm[0], "n_A":   nrm[1], "n_S":   nrm[2]
            })
    
    df_elec = pd.DataFrame(ch_data)

    ch_pos_mm = {
        d['name']: np.array([d['R(mm)'], d['A(mm)'], d['S(mm)']]) for d in ch_data
    }
    montage = mne.channels.make_dig_montage(
        ch_pos={n: p/1000.0 for n, p in ch_pos_mm.items()},
        nasion=fiducials.get("nasion") if fiducials else None,
        lpa   =fiducials.get("lpa")    if fiducials else None,
        rpa   =fiducials.get("rpa")    if fiducials else None,
        coord_frame=fiducials.get("coord_frame", "head") if fiducials else "head",
    )

    return montage, df_elec

def find_peak_electrodes(evoked: mne.Evoked, component_name: str, channels_to_consider: list[str], top_n: int = 3):
    """
    Finds top N electrodes for a component, searching only within a specified list of channels.
    """
    component_definitions = {
        "N20": {"window": PARAMS["N20_WINDOW"], "mode": "neg"},
        "P20": {"window": PARAMS["P20_WINDOW"], "mode": "pos"},
        "N30": {"window": PARAMS["N30_WINDOW"], "mode": "neg"},
        "P30": {"window": PARAMS["P30_WINDOW"], "mode": "pos"},
    }

    if component_name not in component_definitions:
        print(f"ERROR: Component '{component_name}' is not defined.")
        return []

    comp_info = component_definitions[component_name]
    window = comp_info["window"]
    mode = comp_info["mode"]
    polarity_str = "Negative" if mode == "neg" else "Positive"

    print(f"\n--- Finding Top {top_n} {polarity_str} Electrodes for {component_name} in window ({window[0]}-{window[1]}s)...")

    # 1. Get all "good" channels that are present in the final evoked data
    good_channels_in_evoked = [ch for ch in evoked.ch_names if ch not in evoked.info.get('bads', [])]

    # 2. Find the intersection of the good evoked channels and the desired hemisphere channels
    final_chs_to_search = [ch for ch in good_channels_in_evoked if ch in channels_to_consider]
    
    # --- New diagnostic prints ---
    print(f"  -> {len(evoked.ch_names)} total channels exist in the evoked data.")
    print(f"  -> {len(channels_to_consider)} channels are relevant to the selected hemisphere.")
    print(f"  -> Searching within {len(final_chs_to_search)} channels at the intersection.")

    if not final_chs_to_search:
        print("  -> No channels left to search after filtering. Aborting.")
        return []

    # Get the data for the final list of channels
    picks = mne.pick_channels(evoked.ch_names, include=final_chs_to_search)
    evoked_crop = evoked.copy().crop(tmin=window[0], tmax=window[1])
    data = evoked_crop.get_data(picks=picks)

    if data.size == 0:
        print("  -> No data found for the selected channels and time window.")
        return []

    # Find peak values and sort
    if mode == "neg":
        peak_values = np.min(data, axis=1)
        sorted_indices = np.argsort(peak_values)
    else: # mode == "pos"
        peak_values = np.max(data, axis=1)
        sorted_indices = np.argsort(peak_values)[::-1]

    # Format the top N results
    top_results = []
    print("  Rank | Electrode | Peak Value")
    print("  ---------------------------------")
    for rank, i in enumerate(sorted_indices[:top_n], 1):
        ch_name = final_chs_to_search[i]
        peak_val = peak_values[i]
        peak_idx  = np.argmin(data[i]) if mode == 'neg' else np.argmax(data[i])
        peak_time = evoked_crop.times[peak_idx]
        top_results.append((ch_name, peak_val, peak_time))   # 3-tuple
        print(f"  #{rank:<4}| {ch_name:<10}| {peak_val*1e6:6.2f} µV @ {peak_time*1e3:4.1f} ms")


    return top_results

def localise_source(epochs: mne.Epochs, evoked: mne.Evoked, fwd: mne.Forward,
                    subject: str, subjects_dir: str,
                    time_window: tuple, component_name: str, output_dir: Path) -> tuple:
    method = PARAMS["INVERSE_METHOD"]
    lambda2 = PARAMS["INVERSE_LAMBDA2"]
    loose = PARAMS["INVERSE_LOOSE"]
    depth = PARAMS["INVERSE_DEPTH"]

    print(f"\nLocalizing component: {component_name} @ {time_window[0]*1e3:.1f}–{time_window[1]*1e3:.1f} ms")

    rank = mne.compute_rank(evoked, rank='info')
    print(f"→ Rank structure: {rank}")

    noise_cov = mne.compute_covariance(
        epochs, tmin=PARAMS["BASELINE"][0], tmax=PARAMS["BASELINE"][1],
        method='auto', n_jobs=-1, rank=rank, verbose=True
    )
    print("→ Covariance matrix:")
    print(noise_cov.data)
    print(f"→ Cov shape: {noise_cov.data.shape}, diag std: {np.std(np.diag(noise_cov.data)):.2e}")

    inv = mne.minimum_norm.make_inverse_operator(
        evoked.info, fwd, noise_cov,
        loose=loose, depth=depth, rank=rank, verbose=True
    )
    print("Inverse operator created.")

    stc = mne.minimum_norm.apply_inverse(
        evoked, inv, lambda2=lambda2, method=method, verbose=True
    )
    print(f"→ STC shape: {stc.data.shape}")
    print(f"→ STC max abs: {np.max(np.abs(stc.data)):.3e}, min: {np.min(stc.data):.3e}")

    stc_crop = stc.copy().crop(*time_window)
    peak_vert_idx, peak_time = stc_crop.get_peak(mode='abs', vert_as_index=True)
    print(f"Peak found at {peak_time*1e3:.1f} ms, vertex index: {peak_vert_idx}")

    n_lh_verts = fwd['src'][0]['nuse']
    hemi = 'lh' if peak_vert_idx < n_lh_verts else 'rh'
    vert_idx = peak_vert_idx if hemi == 'lh' else peak_vert_idx - n_lh_verts

    print(f"Hemisphere: {hemi}, Vertex idx: {vert_idx}")

    surf_coords = fwd['src'][0 if hemi == 'lh' else 1]['rr'][vert_idx] * 1000
    src_normal = fwd['src'][0 if hemi == 'lh' else 1]['nn'][vert_idx]

    print(f"→ Peak MNI RAS (mm): {np.round(surf_coords, 2)}")
    print(f"→ Peak normal: {np.round(src_normal, 3)}")

    return surf_coords, src_normal, stc, peak_time

def make_report(
    *,
    condition_name: str,
    raw:               mne.io.Raw,
    evoked:            mne.Evoked,
    trans:             mne.transforms.Transform,
    stc_n20:           mne.SourceEstimate,
    peak_time_n20:     float,
    src_xyz_mm_n20:    np.ndarray,
    stc_p30:           mne.SourceEstimate,
    peak_time_p30:     float,
    src_xyz_mm_p30:    np.ndarray,
    peak_electrode_results_dict: dict,
    subj_dir:          Path,
    output_dir:        Path,
    fig_path:          Path,
) -> Path:
    """
    Build a fully-featured HTML report with static and scrollable source plots.
    """
    import mne, matplotlib.pyplot as plt, os, textwrap

    # 0. Initialise report
    report_path = output_dir / f"report_{condition_name}.html"
    rpt = mne.Report(title=f"SSEP Analysis Report: {condition_name}", verbose=True)

    # 1. Inject CSS to re-enable scrolling
    css_str = textwrap.dedent(
        """
        html, body { height: auto !important; overflow-y: auto !important; }
        #main-container, .container, .scroll-container { overflow-y: auto !important; }
        """
    )
    if hasattr(rpt, "add_css"):
        rpt.add_css(css_str)
    else:
        rpt.add_html(f"<style>{css_str}</style>", title="Custom Report Styling", section="Configuration")

    # 2. Data-quality & co-registration
    subjects_dir = os.environ.get("SUBJECTS_DIR", "")
    subject      = PARAMS["FS_SUBJECT"]
    rpt.add_raw(
        raw.copy().crop(tmin=20, tmax=80),
        title="Raw Data (60 s preview)", psd=True, butterfly=True, tags=("raw", "qc"),
    )
    plot_coregistration_check(
        info=raw.info, trans=trans, subject=subject,
        subjects_dir=subjects_dir, output_dir=output_dir,
    )
    for view in ("top", "left", "front"):
        p = output_dir / f"coregistration_{view}.png"
        if p.exists():
            rpt.add_image(p, title=f"Coregistration – {view}", tags=("coreg", "qc"))

    # 3. Evoked & electrode ranking
    fig_evoked = evoked.plot(show=False, window_title="Averaged EP")
    rpt.add_figure(fig_evoked, title="Averaged Evoked Potential", tags=("evoked", "qc"))
    plt.close(fig_evoked)

    # Loop through the dictionary of results and create a table for each component
    html_tables = ""
    table_style = "<style>table,th,td{border:1px solid #444;border-collapse:collapse;padding:6px}</style>"
    
    for component_name, results_list in peak_electrode_results_dict.items():
        html_tables += f"<h3>Top 3 Electrodes ({component_name})</h3>"
        html_tables += (
            "<table><tr>"
            "<th>Rank</th><th>Electrode</th><th>Peak (µV)</th><th>Time (ms)</th>"
            "</tr>"
        )

        if results_list:
            for r, (c, v, t) in enumerate(results_list, 1):
                html_tables += (
                    f"<tr><td>{r}</td>"
                    f"<td>{c}</td>"
                    f"<td>{v * 1e6:.2f}</td>"
                    f"<td>{t * 1e3:.1f}</td></tr>"
                )
        else:
            html_tables += "<tr><td colspan='4'>No peaks found</td></tr>"

        html_tables += "</table><br>"


    rpt.add_html(table_style + html_tables, title="Peak Electrodes", tags=("elec", "results"))

    # ------------------------------------------------------------------
    # 3.5: 2-D butterfly plot at each sensor position
    # ------------------------------------------------------------------

    evoked_for_plot = evoked.crop(tmin = 0.00, tmax = 0.05, include_tmax=True, verbose=None)

    fig_topo = mne.viz.plot_compare_evokeds(
        {"EP": evoked_for_plot},            # single condition - one line per sensor
        picks="eeg",               # only EEG channels
        axes="topo",               # grid layout
        colors={"EP": "k"},
        vlines=[0.020, 0.030], 
        styles={"EP": dict(linewidth=0.9)},
        show_sensors=False,
        show = False,
        title="Evoked Potential Topography (0-50ms)"
    )
    fig_topo[0].savefig(output_dir / "topo_subplot.png", dpi=300)
    plt.close(fig_topo[0].figure)

    extra_imgs = [
        ('montage_names.png',          'Montage (channel names)'),
        ('topo_subplot.png',           'Sensor‑topo EP grid'), 
    ]
    
    for fname, title in extra_imgs:
        p = output_dir / fname
        if p.exists():
            rpt.add_image(p, title=title, tags=('evoked',))

    # 4. Static EP grid + source snapshots
    rpt.add_image(fig_path, title="All channels 0–80 ms", tags=("evoked", "overview"))

    for tag, stc, t_pk, xyz in (
        ("N20", stc_n20, peak_time_n20, src_xyz_mm_n20),
        ("P30", stc_p30, peak_time_p30, src_xyz_mm_p30),
    ):
        ras = ", ".join(f"{x:.1f}" for x in xyz)
        rpt.add_html(
            f"<h3>{tag} Peak</h3><p>Time = {t_pk*1e3:0.1f} ms, MNI RAS = {ras} mm</p>",
            f"Source {tag}", tags=("source", "results")
        )
        mne.viz.set_3d_backend("pyvistaqt", verbose=True)
        brain = stc.plot(
            subject=subject, subjects_dir=subjects_dir, hemi="split",
            views=("lat", "med", "dorsal"), initial_time=t_pk, time_viewer=True,
            clim=dict(kind='percent', pos_lims=[99.5, 99.9, 99.99]), 
            size=(1000, 600), show_traces=True, verbose=True,
        )
        snap = output_dir / f"src_{tag.lower()}_peak.png"
        brain.save_image(snap)
        brain.close()
        rpt.add_image(snap, title=f"Source {tag} Peak", tags=("source", "results"))

    # ------------------------------------------------------------------ #
    # 5. Add a scrollable gallery of the source estimate over time       #
    # ------------------------------------------------------------------ #
    rpt.add_stc(
        stc=stc_n20, # Use the full STC object
        title="Source Localization Over Time",
        subjects_dir=subjects_dir,
        tags=("source", "gallery"),
    )

    # ------------------------------------------------------------------
    # 6. 3-D scalp maps at 0, 40 and 50 ms
    # ------------------------------------------------------------------
    mne.viz.set_3d_backend("pyvistaqt")                

    field_map = mne.make_field_map(
        evoked, trans=trans, subject=PARAMS["FS_SUBJECT"],
        ch_type="eeg", origin="auto", mode="accurate",
        head_source=("bem", "head"), subjects_dir=subj_dir)

    # (timestamp[s], label, view-order list) tuples
    time_specs = [
        (0.00, "0ms",  ["top", "left", "back"]),
        (0.02, "20ms", ["top", "left", "back"]),
        (0.03, "30ms", ["top", "left", "back"]),
    ]

    img_rows = []          # collect HTML rows here
    # camera presets just for the field map scene
    view_dict = {
        "top":  dict(azimuth= 90, elevation= 0),   # straight down
        "left": dict(azimuth= 180, elevation=  90),   # subject's left side
        "back": dict(azimuth= -90, elevation= 90),   # from behind
    }
    # 0 90 is false
    # 180 90 is mirror of the one above
    # 90 0 and 180 0 is the same top

    view_caption = {"top": "Superior",
                    "left": "Left lateral",
                    "back": "Posterior"}          # medical view names

    header_html = (
        "<tr><th style='padding:4px 8px;border:1px solid #444;'>Time</th>"
        + "".join(
            f"<th style='padding:4px 8px;border:1px solid #444;'>{view_caption[v]}</th>"
            for v in ["top", "left", "back"]      # same order as views
        )
        + "</tr>"
    )

    for t_req, lbl, views in time_specs:
        real_t = evoked.times[np.argmin(np.abs(evoked.times - t_req))]
        row_imgs = []
        for v in views:
            fig_field = evoked.plot_field(
                field_map, time=real_t, n_contours=51, interpolation="linear",
                show_density=True, time_viewer=False)
            pl = fig_field.plotter                       # shorthand

            print("\n--- actors & scalar arrays present ---")
            for name, act in pl.actors.items():
                mesh = pv.wrap(act.mapper.GetInput())
                p_scal = list(mesh.point_data.keys())
                c_scal = list(mesh.cell_data.keys())
                print(f"{name or '<unnamed>':12s}  "
                    f"cells={mesh.n_cells:6d}  "
                    f"point_data={p_scal}  cell_data={c_scal}")
            print("---------------------------------------\n")

            FIELD_NAMES = {"field", "Data", "scalars"} 

            def find_scalp_actor(plotter: pv.Plotter) -> pv.Actor:
                """
                IGNORE, me trying to find the names it actually uses

                Heuristic to grab the scalp–field surface returned by evoked.plot_field().

                Order of preference
                -------------------
                1. point‑data array named in FIELD_NAMES
                2. cell‑data array named in FIELD_NAMES
                3. *any* actor that carries at least one scalar array
                4. largest surface in the scene (before you add the pials)
                """
                # -- 1,2 -----------------------------------------------------------
                for act in plotter.actors.values():
                    mesh = pv.wrap(act.mapper.GetInput())

                    try:
                        if FIELD_NAMES.intersection(mesh.point_data.keys()):
                            return act
                    except KeyError:
                        pass  # .keys() never raises but we’re defensive

                    try:
                        if FIELD_NAMES.intersection(mesh.cell_data.keys()):
                            return act
                    except KeyError:
                        pass

                # -- 3 ---------------------------------------------------------------
                scalar_actors = [
                    act for act in plotter.actors.values()
                    if pv.wrap(act.mapper.GetInput()).point_data.n_arrays
                    or pv.wrap(act.mapper.GetInput()).cell_data.n_arrays
                ]
                if scalar_actors:
                    return scalar_actors[0]

                # -- 4 ---------------------------------------------------------------
                if plotter.actors:
                    return max(plotter.actors.values(),
                            key=lambda a: pv.wrap(a.mapper.GetInput()).n_cells)

                raise RuntimeError("Could not identify a scalp surface – "
                                "check the actors list for clues.")


            scalp_actor = find_scalp_actor(pl)
            
            
            # Make it translucent but keep its colormap
            scalp_actor.GetProperty().SetOpacity(0.7)
            scalp_actor.mapper.ScalarVisibilityOn()
            """

            # ---------------------------------------------------------------
            # IN THEORY PIALS CAN BE SHOWN BUT I DIDNT MANAGE
            # ---------------------------------------------------------------
            subject_dir = Path(os.environ["SUBJECTS_DIR"]) / PARAMS["FS_SUBJECT"] / "surf"
            for hemi in ("lh", "rh"):
                verts, faces = mne.read_surface(
                    subject_dir / f"{hemi}.pial", read_metadata=False)
                mesh = pv.PolyData(verts * 1e3, np.c_[np.full(len(faces), 3), faces])
                pl.add_mesh(mesh, color='lightgray', opacity=1.0,
                name=f'pial_{hemi}', pickable=False)
            
            pl.add_actor(scalp_actor)        
            scalp_actor.GetProperty().SetOpacity(1.0)
            """

            info  = evoked.info
            picks = mne.pick_types(info, eeg=True, exclude=[])
            xyz = np.array([info['chs'][pi]['loc'][:3] for pi in picks])
            xyz = apply_trans(trans, xyz) # Convert HEAD -> MRI
            labels = [info['ch_names'][pi] for pi in picks]


            pl.add_points(xyz, render_points_as_spheres=True, point_size=6,
                        color="white", name="eeg_sensors", pickable=False, render=False)
            pl.add_point_labels(
                xyz, labels, font_size=9, text_color="white",
                name="eeg_labels", shape=None, always_visible=True, pickable=False)

            pl.reset_camera()
            mne.viz.set_3d_view(fig_field, **view_dict[v])

            png = output_dir / f"fieldmap_{lbl}_{v}.png"
            pl.screenshot(filename=str(png))
            row_imgs.append(png)

            pl.close()                                    # instead of pl.plotter.close(), rookie mistake

        cells = "".join(
            f'<td style="border:1px solid #444;"><img src="{p.name}" width="280"></td>'
            for p in row_imgs
        )
        img_rows.append(
            f"<tr><th style='padding:4px 8px;border:1px solid #444;text-align:right;'>{lbl}</th>{cells}</tr>"
        )


    #  3x3 grid
    table_html = (
        "<h3>Scalp field maps</h3>"
        '<table style="border-collapse:collapse;">'
        + header_html           # the column headings
        + "".join(img_rows)     # the three rows
        + "</table>"
    )

    rpt.add_html(table_html, title="Field maps", tags=("field",))


    del fig_field 

    # 7. Write out
    rpt.save(report_path, overwrite=True, open_browser=False)
    print(f"Report saved → {report_path}")
    plt.close("all")
    return report_path

def plot_coregistration_check(
        info: mne.Info,
        trans,
        subject: str,
        subjects_dir: str,
        output_dir: Path,
        *,
        dpi: int = 300,
) -> None:
    """
    Render 'top', 'left' and 'front' views of the head-sensor alignment into
    PNG files, with clearly visible electrodes.
    """
    print("— coreg — building 3-D scene with visible electrodes")
    mne.viz.set_3d_backend("pyvistaqt")

    # Build the scene
    fig = mne.viz.plot_alignment(
    info=info,
    trans=trans,
    subject=subject,
    subjects_dir=subjects_dir,
    coord_frame="mri",
    interaction="trackball",
    surfaces=dict(head=0.4, outer_skull=0.1, brain=0.1),
    eeg=True,
    dig="fiducials",      
    meg=False, ecog=False, seeg=False, fnirs=False, dbs=False,
    show_axes=True,
    sensor_colors={"eeg": "yellow"},
    sensor_scales={"eeg": 0.5e-2},
    )

    # Camera presets
    cams = dict(top   = dict(azimuth=90,  elevation=0),
                left  = dict(azimuth=180, elevation=90),
                front = dict(azimuth=90,  elevation=90))

    for name, cam in cams.items():
        # Position the virtual camera
        mne.viz.set_3d_view(
            fig,
            azimuth   = cam["azimuth"],
            elevation = cam["elevation"],
            distance  = None,
            focalpoint= "auto",
        )

        # Grab a screenshot
        img = fig.plotter.screenshot(
            transparent_background=False,
            return_img=True,
        )

        # Write to disk
        out = output_dir / f"coregistration_{name}.png"
        plt.imsave(out, img)
        print(f"✔ saved {out.name}")
    fig.plotter.close()
    print("Coregistration plots saved.")

def write_outputs(output_dir: Path,
                  df_elec: pd.DataFrame,
                  dipole_results: dict,
                  src_xyz: np.ndarray,
                  src_n: np.ndarray):
    """Saves electrode and source-coordinate data to text files."""
    df_elec.to_csv(output_dir / "electrodes_RAS.txt",
                   sep="\t", index=False, float_format="%.4f")

    strongest = dipole_results.get("strongest_ch")
    if strongest:                                 # only if a channel was found
        df_elec.loc[df_elec["name"] == strongest] \
               .to_csv(output_dir / "largest_EP_electrode.txt",
                        sep="\t", index=False, float_format="%.4f")

    pd.DataFrame({
        "R(mm)": [src_xyz[0]], "A(mm)": [src_xyz[1]], "S(mm)": [src_xyz[2]],
        "n_R": [src_n[0]],    "n_A": [src_n[1]],    "n_S": [src_n[2]]
    }).to_csv(output_dir / "EP_source.txt",
              sep="\t", index=False, float_format="%.4f")
    
    
def write_strongest_electrode_files(output_dir: Path, df_elec: pd.DataFrame, strongest_electrodes: dict):
    """
    Saves the RAS and normal data for the strongest electrode of each component to a .txt file.

    Args:
        output_dir: The directory to save the files in.
        df_elec: DataFrame with all electrode names, RAS positions, and normals.
        strongest_electrodes: A dictionary like {"N20": "C3", "P30": "Fz", ...}.
    """
    print("\n--- Writing strongest electrode files ---")
    if not strongest_electrodes:
        print("No strongest electrodes found to write.")
        return

    for component, ch_name in strongest_electrodes.items():
        if not ch_name:
            print(f"  - No strongest electrode for {component}, skipping.")
            continue
        
        # Find the row for the strongest channel in the main electrode dataframe
        electrode_data = df_elec.loc[df_elec["name"] == ch_name]

        if electrode_data.empty:
            print(f"  - Could not find data for electrode '{ch_name}' for component {component}.")
            continue
            
        # Define the output file path
        output_file = output_dir / f"strongest_{component}.txt"
        
        # Save the data to the file
        electrode_data.to_csv(output_file, sep="\t", index=False, float_format="%.4f")
        print(f"Saved {output_file.name}")
    
def plot_all_ep_panels(
    evoked: mne.Evoked,
    bads: list[str],
    tmax: float,
    out_file: Path,
):
    """
    Grid of small multiples (one trace per channel, 0 → tmax s).

    Good channels: default color  
    Bad channels: red  
    """
    ch_names = evoked.ch_names
    n_ch = len(ch_names)
    import math

    n_cols = math.ceil(math.sqrt(n_ch))
    n_cols = min(n_cols, 10)
    n_rows = math.ceil(n_ch / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 2.0, n_rows * 1.9),
        sharex=True,
        sharey=True,
    )
    axes = axes.ravel()

    # ------------------------------------------------------------------
    # DATA SLICE ADJUST IF NEEDED
    # ------------------------------------------------------------------
    times = evoked.times
    sel = (times >= -0.020) & (times <= tmax)

    # eeg channels excluding bads
    pick_idx = mne.pick_types(evoked.info, eeg=True, exclude='bads')  
    # pick_idx is a numpy array of integers

    # rows = channels, columns = time points
    data_sel = evoked.data[pick_idx][:, sel]

    # symmetric y-limits
    vlim = np.max(np.abs(data_sel))
    ymin, ymax = -vlim, vlim


    # ------------------------------------------------------------------
    # iterate over channels
    # ------------------------------------------------------------------
    for ax_i, ch_name in enumerate(ch_names):
        ax = axes[ax_i]
        idx = evoked.ch_names.index(ch_name)

        color = "crimson" if ch_name in bads else None
        ax.plot(times[sel] * 1e3, evoked.data[idx, sel], linewidth=1.0, color=color)

        ax.set_title(ch_name, fontsize=8, pad=2)
        ax.axvline(0, color="k", linewidth=0.6)
        ax.set_xticks([-20, -10, 0, 10, 20, 30, 40, 50, 60, 70])
        ax.tick_params(axis="both", labelsize=6)
        ax.set_ylim(ymin, ymax)

    # any empty panes (grid bigger than channel count situation)
    for extra_ax in axes[n_ch:]:
        extra_ax.axis("off")

    fig.suptitle(
        "Evoked potentials 0–{:.0f} ms (good = default, bad = red)".format(tmax * 1e3),
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out_file, dpi=300)
    plt.close(fig)

from mne.baseline import rescale

def zscore_epochs(epochs: mne.Epochs, baseline):
    """
    IGNORE UNUSED
    Return a *copy* whose data are baseline-z-scored epoch-wise."""
    epochs_z = epochs.copy()
    data_z = rescale(
        epochs_z.get_data(),      # ndarray shape (n_ep, n_ch, n_t)
        epochs_z.times,
        baseline=baseline,
        mode="zscore") 
    epochs_z._data = data_z
    return epochs_z

def interpolate_stim_artifact(epochs: mne.Epochs, t_min: float, t_max: float) -> mne.Epochs:
    """
    Finds and interpolates a stimulation artifact in a given time window.
    """
    print(f"--- Interpolating stimulation artifact from {t_min * 1000:.1f} to {t_max * 1000:.1f} ms...")
    epochs_interp = epochs.copy()
    
    idx_min, idx_max = epochs_interp.time_as_index([t_min, t_max], use_rounding=True)
    data = epochs_interp.get_data()
    
    start_vals = data[:, :, idx_min - 1]
    end_vals   = data[:, :, idx_max + 1]
    
    num_points_to_interp = (idx_max - idx_min) + 1
    
    # interpolation line for each epoch and channel
    interp_segment = np.linspace(start_vals[:, :, np.newaxis], 
                                 end_vals[:, :, np.newaxis], 
                                 num_points_to_interp, axis=2)
 
    # replace the artifactual data segment
    data[:, :, idx_min:idx_max + 1] = interp_segment.squeeze(-1) # Squeeze the last dimension
    
    epochs_interp._data = data
    print("Artifact interpolation complete.")
    return epochs_interp

def compute_coregistration(
    info: mne.Info,
    subject: str,
    subjects_dir: str,
    *,
    trans_out: Path | None = None,
    n_icp: int = 10000,
    fid_w: float = 1.,
    hsp_dist: float = 0.005,
    verbose: bool = True,
) -> mne.transforms.Transform:
    from mne.coreg import Coregistration
    from mne.transforms import write_trans

    """
    AUTOMATIC COREGISTRATION
    Check Paper
    """

    print("Starting automatic coregistration procedure...")

    coreg = Coregistration(
        info=info,
        subject=subject,
        subjects_dir=subjects_dir,
    )

    print(f"Digitized points found: {len(info['dig'])}")
    for dig in info['dig']:
        print(f"  - {dig['kind']} | RAS (mm): {np.round(dig['r'] * 1000, 2)}")

    print("Aligning fiducials (coarse fit)...")
    coreg.fit_fiducials(
        nasion_weight=fid_w,
        lpa_weight=1.,
        rpa_weight=1.,
    )
    print(f"Fiducial transform:\n{coreg.trans}")

    print(f"Omitting head shape points > {hsp_dist*1000:.1f} mm from scalp...")
    coreg.omit_head_shape_points(distance=hsp_dist)

    print(f"Fine fit (ICP, {n_icp} iterations)...")
    coreg.fit_icp(
        n_iterations=n_icp,
        nasion_weight=fid_w,
        lpa_weight=1.,
        rpa_weight=1.,
        hsp_weight=1.,
        eeg_weight=1.,
    )

    trans = coreg.trans
    dists = coreg.compute_dig_mri_distances()
    rms = np.sqrt(np.mean(dists ** 2)) * 1000
    d_max = np.max(dists) * 1000

    print(f"RMS error: {rms:.2f} mm")
    print(f"Max error: {d_max:.2f} mm")
    print(f"Transform matrix:\n{trans}")

    if trans_out:
        write_trans(trans_out, trans, overwrite=True)
        print(f"Wrote transform to {trans_out}")

    return trans

def plot_montage_with_names(evoked, out_file):
    """labelled 2-D montage."""
    fig = evoked.plot_sensors(show_names=True, kind='topomap', sphere='auto', to_sphere=False)         
    fig.savefig(out_file, dpi=100)
    plt.close(fig)

def scalp_region_channels(ch_names: list[str], region: str) -> list[str]:
    """
    Return only channels whose names contain one of the region substrings.
    """
    if region == "posterior":
        keys = PARAMS["POSTERIOR_CH"]
    elif region == "anterior":
        keys = PARAMS["ANTERIOR_CH"]
    else:
        return ch_names
    return [c for c in ch_names if any(k.upper() in c.upper() for k in keys)]

def export_evoked_mat(evoked, peak_times, fname):
    """
    peak_times = {'N20': t1, 'P20': t2}  (s)
    """
    evoked_for_copy = evoked.copy()
    evoked_for_copy.drop_channels(evoked.info['bads'])
    out = {
        'channel_names': np.array(evoked_for_copy.ch_names, dtype=object)
    }
    for comp, t in peak_times.items():
        amps = evoked_for_copy.data[:, evoked_for_copy.time_as_index(t)]
        out[f'amp_{comp}']    = amps
        out[f'radius_{comp}'] = 5 + 20 * np.abs(amps) / np.max(np.abs(amps))

        # 1  = positive (red),  -1 = negative (blue)
        out[f'colour_{comp}'] = np.where(amps >= 0,  1, -1)
    savemat(fname, out)
    print(f"wrote {fname}")

# ========================================================
# === 3. MAIN EXECUTION  =================================
# ========================================================
def main():
    print("--- Initializing Pipeline Setup ---")
    subj_dir = prepare_freesurfer()

    # =========================================================================
    # === 1. ONE-TIME SETUP (BEM, Electrode Locations, and Coregistration) ====
    # =========================================================================
    
    # --- Create BEM Model (runs once per subject) ---
    print(f"--- Creating 3-layer BEM model for subject: {PARAMS['FS_SUBJECT']} ---")
    bem_model = mne.make_bem_model(
        subject=PARAMS["FS_SUBJECT"],
        subjects_dir=subj_dir,
        conductivity=[0.3, 0.006, 0.3]
    )
    bem_sol = mne.make_bem_solution(bem_model, verbose=True)

    # --- Load Electrode and Fiducial Information (runs once per subject) ---
    xml_ch_names = (ET.parse(PARAMS["MONTAGE_XML"]).getroot().findtext("electrodename").split(","))
    eeg_lkp = load_brainmodel_electrodes(PARAMS["BRAIN_XML"])

    # --- Perform Coregistration (runs interactively once, then just loads the file) ---
    print("--- Loading a representative file for coregistration setup ---")
    representative_raw_file = PARAMS["EEG_FOLDER"] / PARAMS["EEG_FILES"][0]
    raw_for_coreg = mne.io.read_raw_edf(representative_raw_file, preload=False, stim_channel=None)
    raw_for_coreg.rename_channels(_auto_map(raw_for_coreg.ch_names, xml_ch_names))
    
    # Create and set the montage on this temporary raw object
    fiducials = load_ft_fiducials(PARAMS["FIDUCIALS_MAT"],
                              coord_frame="mri",   # or "head" if already there 
                              units="mm")

    montage, df_elec = create_montage_and_df(
        raw_for_coreg.ch_names,
        eeg_lkp,
        fiducials=fiducials
    )
    raw_for_coreg.set_montage(montage, on_missing='warn')

    # This is the proper call to the manual coregistration function.
    # It will launch the GUI if the TRANS_FILE doesn't exist.
    # On all subsequent runs, it will simply load the file and skip the GUI.
    """
    trans = get_or_create_coregistration(
        raw=raw_for_coreg,
        subject=PARAMS["FS_SUBJECT"],
        subjects_dir=subj_dir,
        trans_fname=PARAMS["TRANS_FILE"],
    )
    del raw_for_coreg # Clean up the temporary object
    """
    trans = compute_coregistration(
        info=raw_for_coreg.info,
        subject=PARAMS["FS_SUBJECT"],
        subjects_dir=subj_dir)

    # =========================================================================
    # === 2. PER-FILE & PER-CONDITION PROCESSING LOOP =========================
    # =========================================================================
    from collections import defaultdict
    results_collector = defaultdict(dict)


    combine = PARAMS["COMBINE_EVEN_ODD"]
    trigger_types = ["combined"] if combine else ["odd", "even"]

    for eeg_file_name in PARAMS["EEG_FILES"]:
        file_base = Path(eeg_file_name).stem 
            # make sure there is a sub-dict for this file
        if file_base not in results_collector:
            results_collector[file_base] = {}


        # Choose where results go
        out_root = PARAMS["OUTPUT_DIR_AVG"] if combine else PARAMS["OUTPUT_DIR"]

        for trigger_type in trigger_types:
            # Folder naming:  two session folders if combine=True, else
            # one sub-folder per oddd / even condition
            condition = file_base if trigger_type == "combined" else f"{file_base}_{trigger_type}"
            out_dir = out_root / condition
            if out_dir.exists():
                # turn files -> folders OR just wipe previous runs directory
                if out_dir.is_file():
                    out_dir.unlink()          # remove the file
                else:
                    shutil.rmtree(out_dir)    # or keep it if you prefer
            out_dir.mkdir(parents=True, exist_ok=True)

            # --- Load data for the current condition ---
            raw = mne.io.read_raw_edf(PARAMS["EEG_FOLDER"] / eeg_file_name, preload=True, stim_channel=None, verbose=True)

            """
            CONVERT TO VOLTS PART
            """
            raw.apply_function(lambda x: x * 1e-6, picks="eeg", verbose=True) # Convert to Volts

            raw.rename_channels(_auto_map(raw.ch_names, xml_ch_names))
            
            # --- Set bad channels and the montage  ---
            raw.info["bads"] = [ch for ch in PARAMS["MANUAL_BAD_CHANNELS"] if ch in raw.ch_names]
            raw.set_montage(montage, on_missing='warn')

            raw, _ = mne.set_eeg_reference(raw, ref_channels="average", projection=True, verbose=True)
            raw.apply_proj()
            raw.filter(l_freq=PARAMS["HIGH_PASS_ANALYSIS"], h_freq=PARAMS["LOW_PASS_ANALYSIS"], 
                       method="iir", iir_params=dict(order=2, ftype="butter"), phase="zero", verbose=True)

            if PARAMS["USE_ICA"]:
                picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
                print("\n--- Applying ICA for artifact removal...")
                ica_tmp = (raw.copy()
                                 .filter(l_freq=1., h_freq=48.,
                                         method='iir',
                                         iir_params=dict(order=2, ftype='butter'),
                                         phase='zero'))
                ica = mne.preprocessing.ICA(
                    n_components= round(len(picks)/1), 
                    method="picard", random_state=42, max_iter="auto"
                )
                ica.fit(ica_tmp)
                ica.exclude, scores_muscle = ica.find_bads_muscle(raw)
                print(
                    f"Applying ICA to remove {len(ica.exclude)} "
                    "muscle component(s)."
                )
                """
                ICA PLOT IF NEEDED
                """
                """
                ica.plot_components(inst=raw)
                ica.plot_scores(scores_muscle)
                ica.plot_sources(raw, start=6.5, stop=9.5)
                """
                ica.apply(raw)
            # --- Epoching and Artifact Handling ---
            events = mne.events_from_annotations(raw, regexp=PARAMS["ANNOTATION_REGEX"], verbose=True)[0]
            stim_offset_samples = int(round(PARAMS["STIM_OFFSET_S"] *
                                raw.info['sfreq']))
            events[:, 0] += stim_offset_samples
            if trigger_type == "combined":
                events_subset = events                              
            else:
                idx = 0 if trigger_type == "odd" else 1
                events_subset = events[idx::2]                

            # Create epochs without baseline correction first
            epochs = mne.Epochs(raw, events_subset, tmin=PARAMS["TMIN"], tmax=PARAMS["TMAX"],
                                baseline=None, preload=True, reject=None, proj=True, verbose=True)
            
            # 1. Interpolate the artifact
            epochs = interpolate_stim_artifact(epochs, t_min=-0.002, t_max=0.002)
            # 2. Now apply baseline correction
            epochs.apply_baseline(PARAMS["BASELINE"])
            
            print(f"Created {len(epochs)} initial epochs for '{trigger_type}' triggers.")

            """
            AUTOREJECT
            """

            if PARAMS["USE_AUTOREJECT"]:
                from autoreject import AutoReject
                ar = AutoReject(n_interpolate=[1, 2, 4, 8], consensus=[0.4, 0.5, 1.0], 
                                picks=mne.pick_types(epochs.info, eeg=True, exclude='bads'),
                                n_jobs=-1, random_state=42, verbose=True)
                epochs_clean, log = ar.fit_transform(epochs, return_log=True)
                print(f"AutoReject kept {len(epochs_clean)} / {len(epochs)} epochs")
            else:
                epochs_clean = epochs

            evoked = epochs_clean.average(method='mean') # or method='median' but it's worse

            channels_to_consider = get_hemisphere_channels(evoked.ch_names, PARAMS["STIMULATED_HEMISPHERE"])
            
            strongest_electrodes = {}
            peak_results_for_report_dict = {}

            if trigger_type in ("even", "combined"):
                # For EVEN trials, analyze N20 and P20, and report both
                results_n20 = find_peak_electrodes(evoked, "N20", channels_to_consider, top_n=3)
                results_p20 = find_peak_electrodes(evoked, "P20", channels_to_consider, top_n=3)
                
                # Store results for the report
                peak_results_for_report_dict.update({"N20": results_n20, "P20": results_p20})
                
                # Store the single strongest for the .txt files
                if results_n20: strongest_electrodes["N20"] = results_n20[0][0]
                if results_p20: strongest_electrodes["P20"] = results_p20[0][0]

            if trigger_type in ("odd", "combined"):
                # For ODD trials, analyze P30 and N30, and report both
                results_p30 = find_peak_electrodes(evoked, "P30", channels_to_consider, top_n=3)
                results_n30 = find_peak_electrodes(evoked, "N30", channels_to_consider, top_n=3)

                # Store results for the report
                peak_results_for_report_dict.update({"P30": results_p30, "N30": results_n30})
                
                # Store the single strongest for the .txt files
                if results_p30: strongest_electrodes["P30"] = results_p30[0][0]
                if results_n30: strongest_electrodes["N30"] = results_n30[0][0]

            if trigger_type in ("even", "combined"):          # N20 + P20
                # strongest_electrodes now holds [(name, val, time), ...] from change (B)
                t_n20 = results_n20[0][2] if results_n20 else 0.020
                t_p20 = results_p20[0][2] if results_p20 else 0.020
                export_evoked_mat(evoked,
                    {'N20': t_n20, 'P20': t_p20},
                    out_dir / f"evokedNP20_{condition}.mat")
            if trigger_type in ("odd", "combined"):                               # P30 + N30
                t_p30 = results_p30[0][2] if results_p30 else 0.030
                t_n30 = results_n30[0][2] if results_n30 else 0.030
                export_evoked_mat(evoked,
                    {'P30': t_p30, 'N30': t_n30},
                    out_dir / f"evokedNP30_{condition}.mat")


            results_collector[file_base][trigger_type] = strongest_electrodes

            plot_montage_with_names(evoked, out_dir / 'montage_names.png')

            """
            plot_ep_traces_over_sensors(evoked, 0.020, 0.070,
                            out_dir / 'evoked_overlay_20_70ms.png')
            

            plot_head_heatmaps(evoked, raw.info, trans,
                            PARAMS["FS_SUBJECT"], subj_dir,
                            out_dir=out_dir)
            """
            fig_path = out_dir / f"EP_panels_{condition}.png"
            plot_all_ep_panels(evoked, bads=raw.info["bads"], tmax=0.070, out_file=fig_path)
            print(f"    EP figure saved to {fig_path}")

            # --- Source Localization ---
            # Pass the computed transform to the source localization
            fwd = mne.make_forward_solution(evoked.info, trans=trans, src=mne.setup_source_space(PARAMS["FS_SUBJECT"], spacing=PARAMS["BEM_SPACING"], subjects_dir=subj_dir, add_dist=False, verbose=True), bem=bem_sol, eeg=True, meg=False, mindist=3.0, n_jobs=-1, verbose=True)
            
            # Covariance
            dip_cov = mne.compute_covariance(epochs, tmin=PARAMS["BASELINE"][0], tmax=PARAMS["BASELINE"][1], method='auto', n_jobs=-1, verbose=True)
            
            # Dipole fit
            dip = mne.fit_dipole(evoked.copy().crop(0.018, 0.022), dip_cov, bem_sol, trans)[0]
            figdip = dip.plot_locations(trans, 'mni_subject', subjects_dir=subj_dir, mode='orthoview')

            print(dip)
            figdip.savefig(out_dir / f"dipole_{condition}.png")
            plt.close(figdip)
            
            src_xyz_n20, src_norm_n20, stc_n20, peak_time_n20 = localise_source(epochs_clean, evoked, fwd, PARAMS["FS_SUBJECT"], subj_dir, PARAMS["N20_WINDOW"], "N20", out_dir)
            src_xyz_p30, src_norm_p30, stc_p30, peak_time_p30 = localise_source(epochs_clean, evoked, fwd, PARAMS["FS_SUBJECT"], subj_dir, PARAMS["P30_WINDOW"], "P30", out_dir)

            # --- Reporting & Output ---
            print("\n--- Generating reports and saving outputs...")
            
            report_file = make_report(
                condition_name=condition, raw=raw, evoked=evoked, trans=trans,
                stc_n20=stc_n20, peak_time_n20=peak_time_n20, src_xyz_mm_n20=src_xyz_n20,
                stc_p30=stc_p30, peak_time_p30=peak_time_p30, src_xyz_mm_p30=src_xyz_p30,
                peak_electrode_results_dict=peak_results_for_report_dict, subj_dir =subj_dir,
                output_dir=out_dir, fig_path=fig_path
            )

            # --- Write coordinates and normals to file ---
            # Determine the strongest electrode for N20 (even) and P30 (odd)
            dummy_dip_res = None
            if trigger_type == "even" and "N20" in strongest_electrodes:
                dummy_dip_res = {"strongest_ch": strongest_electrodes["N20"]}

            # Pass the source normal (unused though)
            if dummy_dip_res is not None:
                write_outputs(out_dir, df_elec, dummy_dip_res, src_xyz_n20, src_norm_n20)
            write_strongest_electrode_files(out_dir, df_elec, strongest_electrodes)
            print(f"Analysis for '{condition}' complete. Report at: {report_file}")

# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()

