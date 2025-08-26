#!/usr/bin/env python3
"""
Iteratively shrink the inner-skull surface so it sits ≥ margin mm
everywhere inside the outer-skull surface.
"""
import argparse, pathlib, shutil, numpy as np
from scipy.spatial import cKDTree
from nibabel.freesurfer import read_geometry, write_geometry

SAFETY = 1e-3      # 1 µm

def load(fname):
    v, f = read_geometry(fname)
    return v.astype(float), f.astype(np.int32)



def shrink(inner_v, outer_v, margin):
    com = outer_v.mean(0)                                # coarse head centre
    for it in range(100):                                 # hard stop
        tree       = cKDTree(outer_v)
        dists, _   = tree.query(inner_v)                 # point-to-surface-vert
        bad        = dists < margin
        n_bad      = bad.sum()
        if n_bad == 0:
            print(f"✔ converged after {it} iteration(s)")
            return inner_v
        print(f"  ↪ iter {it+1}: moving {n_bad} vertices")
        vec        = inner_v[bad] - com                  # outward radial
        vec       /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-16
        step       = (margin - dists[bad] + SAFETY)[:, None]
        inner_v[bad] -= step * vec                       # pull inward
    raise RuntimeError("Still intersecting after 100 iterations")

def save(fname, v, f):                                   # FS .surf writer
    write_geometry(fname, v.astype(np.float32), f)

def main(a):
    inner_v, inner_f = load(a.inner)
    outer_v, _       = load(a.outer)

    fixed_v          = shrink(inner_v, outer_v, a.margin)

    out = pathlib.Path(a.out)
    bak = out.with_suffix(out.suffix + ".bak")
    if not bak.exists():
        shutil.copy(a.inner, bak)
    save(out, fixed_v, inner_f)
    print(f"✔ wrote repaired surface → {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner",  required=True)
    ap.add_argument("--outer",  required=True)
    ap.add_argument("--out",    required=True)
    ap.add_argument("--margin", type=float, default=1.5,
                    help="minimum clearance in mm (default 1.5)")
    main(ap.parse_args())
