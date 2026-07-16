#!/usr/bin/env python
"""Compute per-element reference energies from the delta_e entries of npz files.

Fits   delta_e_i ≈ sum_Z  n_{Z,i} * eps_Z
via linear least squares, where n_{Z,i} is the count of element Z in structure i
and eps_Z is its per-atom reference energy.

By default the fit pools the delta_e of ALL expansion orders (1-, 2-, and 3-body
combos) into a single least-squares problem. Every combo's delta_e is the raw
subsystem energy (DFT - DFTB over that subsystem's real atoms), so it scales
extensively with composition and is a valid row of the same linear model
regardless of order. 

Caveat: qm_elems counts only real atoms, while delta_e also contains the
(H) link-atom caps at covalent QM/MM boundaries. That boundary offset is folded
into the fitted references exactly as in the original one-body script -- pooling
orders does not change this treatment.
"""
import argparse
import glob
import os
import re

import numpy as np

# Matches the "combo(...)" fragment of an mbe-mm npz filename, e.g.
# "config_0000_combo(0).npz" (one body) or "config_0000_combo(0,1,2).npz".
COMBO_RE = re.compile(r"combo\(([^)]*)\)")


def n_body(path):
    """Return the number of monomers in a combo(...) filename, or None if absent."""
    m = COMBO_RE.search(os.path.basename(path))
    if m is None:
        return None
    return len([tok for tok in m.group(1).split(",") if tok.strip()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_dir", help="directory (searched recursively) of *.npz files")
    ap.add_argument("--energy-key", default="delta_e")
    ap.add_argument("--out", default="reference_energies.npz")
    ap.add_argument(
        "--orders",
        default="1,2,3",
        help="comma-separated expansion orders to pool into the fit "
        "(default: '1,2,3', i.e. all three orders)",
    )
    ap.add_argument(
        "--n-body",
        type=int,
        default=None,
        help="deprecated single-order override; if given, use only this order "
        "(equivalent to --orders <n>)",
    )
    args = ap.parse_args()

    if args.n_body is not None:
        orders = {args.n_body}
    else:
        orders = {int(tok) for tok in args.orders.split(",") if tok.strip()}
    if not orders:
        raise SystemExit("no expansion orders selected (see --orders)")

    files = sorted(glob.glob(os.path.join(args.npz_dir, "**", "*.npz"), recursive=True))
    files = [f for f in files if n_body(f) in orders]
    if not files:
        raise SystemExit(
            f"no combo(...) npz files of order(s) {sorted(orders)} found under {args.npz_dir}"
        )

    per_order = {o: sum(1 for f in files if n_body(f) == o) for o in sorted(orders)}
    print("orders pooled into fit: " + ", ".join(
        f"{o}-body={per_order[o]}" for o in sorted(orders)
    ))

    # First pass: collect the set of elements present.
    elements = set()
    rows = []  # (composition dict, energy)
    for f in files:
        d = np.load(f, allow_pickle=True)
        elems = [e.strip() for e in d["qm_elems"].tolist()]
        uniq, counts = np.unique(elems, return_counts=True)
        comp = dict(zip(uniq.tolist(), counts.tolist()))
        elements.update(comp)
        rows.append((comp, float(d[args.energy_key])))

    elements = sorted(elements)
    A = np.array([[comp.get(z, 0) for z in elements] for comp, _ in rows], dtype=float)
    b = np.array([e for _, e in rows], dtype=float)

    rank = np.linalg.matrix_rank(A)
    print(f"structures = {len(files)}, elements = {elements}")
    print(f"design-matrix rank = {rank} / {len(elements)} columns")
    if rank < len(elements):
        print(
            "WARNING: rank-deficient -> per-element references are NOT unique.\n"
            "         Only the per-atom mean is well defined for this data.\n"
            f"         per-atom mean = {b.mean() / A.sum(1).mean():.6f} (energy units / atom)"
        )

    eps, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
    ref = dict(zip(elements, eps.tolist()))
    pred = A @ eps
    rmse = float(np.sqrt(np.mean((pred - b) ** 2)))

    print("\nper-element reference energies (one least-squares solution):")
    for z in elements:
        print(f"  {z:>2}: {ref[z]: .6f}")
    print(f"\nfit RMSE on delta_e = {rmse:.6e}")

    np.savez(args.out, elements=np.array(elements), reference_energies=eps)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
