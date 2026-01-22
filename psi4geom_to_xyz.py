#!/usr/bin/env python3
"""
Convert .psi4geom files to .xyz for dimer geometries.

Usage examples:
  python psi4geom_to_xyz.py psi4_geoms --out-dir psi4_geoms_xyz
  python psi4geom_to_xyz.py path/to/geom.psi4geom --out-dir out_xyz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from moltimol import parse_psi4geom_string


def psi4geom_to_xyz(geom_text: str, out_path: Path) -> None:
    symA, symB, XA, XB, _ = parse_psi4geom_string(geom_text)
    symbols = list(symA) + list(symB)
    coords = np.vstack([XA, XB]) if len(symbols) else np.zeros((0, 3), float)
    nA = len(symA)
    nB = len(symB)

    with open(out_path, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write(f"{len(symbols)} {nA} {nB}\n")
        for sym, (x, y, z) in zip(symbols, coords):
            f.write(f"{sym} {x:.10f} {y:.10f} {z:.10f}\n")


def convert_file(path: Path, out_dir: Path, suffix: str) -> Path:
    out_name = f"{path.stem}{suffix}.xyz"
    out_path = out_dir / out_name
    psi4geom_to_xyz(path.read_text(), out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert .psi4geom files to .xyz.")
    parser.add_argument("input", help="Input .psi4geom file or directory")
    parser.add_argument("--out-dir", default="psi4geom_xyz", help="Output directory")
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix appended before .xyz (e.g., _dimer)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.is_dir():
        files = sorted(in_path.glob("*.psi4geom"))
        if not files:
            print(f"No .psi4geom files found in {in_path}")
            return 1
        for path in files:
            out_path = convert_file(path, out_dir, args.suffix)
        print(f"Converted {len(files)} files to {out_dir}")
        return 0

    if in_path.is_file() and in_path.suffix == ".psi4geom":
        out_path = convert_file(in_path, out_dir, args.suffix)
        print(f"Wrote {out_path}")
        return 0

    print(f"Input must be a .psi4geom file or a directory: {in_path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
