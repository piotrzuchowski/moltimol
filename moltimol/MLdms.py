import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psi4

# Allow running as a script while still importing local package code.
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import moltimol as molmol
from prop_sapt import Dimer, calc_property

def sample_R_beta_mode(n, rmin, rmax, mode_frac=1 / 3, concentration=10.0):
    """
    Sample R in [rmin,rmax] using a Beta distribution whose mode is at mode_frac
    (fraction of the interval from rmin).
    """
    if not (rmin < rmax):
        raise ValueError("Require rmin < rmax")
    if not (0.0 < mode_frac < 1.0):
        raise ValueError("Require 0 < mode_frac < 1")
    if concentration <= 2.0:
        raise ValueError("Require concentration > 2 to have an interior mode (a,b>1).")

    c = concentration - 2.0
    a = 1.0 + mode_frac * c
    b = 1.0 + (1.0 - mode_frac) * c

    x = np.random.beta(a, b, size=n)
    R = rmin + (rmax - rmin) * x
    return R


def setup_psi4_defaults(psi4_options=None):
    psi4.set_memory("2 GB")
    psi4.set_num_threads(2)
    psi4.set_options(
        {
            "basis": "aug-cc-pvdz",
            "scf_type": "direct",
            "save_jk": True,
            "DF_BASIS_SCF": "aug-cc-pvdz-jkfit",
            "DF_BASIS_SAPT": "aug-cc-pvdz-ri",
        }
    )
    if psi4_options:
        psi4.set_options(psi4_options)


def _write_psi4geom(
    path,
    symA,
    symB,
    coords,
    n_atoms_A,
    psi4_units="angstrom",
    charge_mult_A="0 1",
    charge_mult_B="0 1",
):
    coords_out = np.asarray(coords, float)
    if psi4_units != "angstrom":
        raise ValueError("Only angstrom units are supported.")

    with open(path, "w") as f:
        f.write("symmetry c1\n")
        f.write("no_com\n")
        f.write("no_reorient\n")
        f.write(f"units {psi4_units}\n")
        f.write(f"{charge_mult_A}\n")
        for sym, (x, y, z) in zip(symA, coords_out[:n_atoms_A]):
            f.write(f"{sym} {x:.10f} {y:.10f} {z:.10f}\n")
        f.write("--\n")
        f.write(f"{charge_mult_B}\n")
        for sym, (x, y, z) in zip(symB, coords_out[n_atoms_A:]):
            f.write(f"{sym} {x:.10f} {y:.10f} {z:.10f}\n")


def body_frame_coord_columns(symA, symB, coords_body, n_atoms_A):
    """
    Build per-atom body-frame coordinate columns:
      x_<sym>A<i>, y_<sym>A<i>, z_<sym>A<i>, then x_<sym>B<i>, ...
    """
    cols = {}
    order = []
    for i, sym in enumerate(symA):
        label = f"{sym}A{i+1}"
        for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
            key = f"{axis}_{label}"
            cols[key] = float(coords_body[i, idx])
            order.append(key)
    for j, sym in enumerate(symB):
        label = f"{sym}B{j+1}"
        k = n_atoms_A + j
        for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
            key = f"{axis}_{label}"
            cols[key] = float(coords_body[k, idx])
            order.append(key)
    return cols, order


def generate_psi4geom_files(
    n_samples=10000,
    fileA="CO.xyz",
    fileB="CO.xyz",
    r_min=3.0,
    r_max=10.0,
    mode_frac=1 / 3,
    concentration=10.0,
    sigma_noise=0.1,
    seed=None,
    out_dir="psi4_geoms",
    psi4_units="angstrom",
    charge_mult_A="0 1",
    charge_mult_B="0 1",
    body_frame=False,
    hist_png="R_hist.png",
):
    """
    Generate SAPT-style Psi4 geometry files for a dimer.
    Names: <fileA>_<fileB>_<id>.psi4geom (id is zero-padded).
    """
    if seed is not None:
        np.random.seed(seed)

    symA, _ = molmol.read_xyz(fileA)
    symB, _ = molmol.read_xyz(fileB)
    n_atoms_A = len(symA)
    n_atoms_B = len(symB)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stemA = Path(fileA).stem
    stemB = Path(fileB).stem

    Rs = sample_R_beta_mode(
        n_samples, r_min, r_max, mode_frac=mode_frac, concentration=concentration
    )
    for i, R in enumerate(Rs):
        u = np.random.uniform(0, 1)
        theta = np.arccos(1 - 2 * u)
        phi = np.random.uniform(0, 2 * np.pi)
        u1, u2, u3 = np.random.rand(3)
        alphaE = 2 * np.pi * u1
        betaE = np.arccos(2 * u2 - 1)
        gammaE = 2 * np.pi * u3
        eulerA = (0.0, 0.0, 0.0)
        eulerB = (alphaE, betaE, gammaE)

        _, _, coords, _ = molmol.merge_monomers_jacobi_XYZ(
            fileA,
            fileB,
            R=R,
            theta=theta,
            phi=phi,
            eulerA=eulerA,
            eulerB=eulerB,
            sigma_noise=sigma_noise,
        )

        if body_frame:
            massesA = np.array([molmol.mass_of(s) for s in symA])
            massesB = np.array([molmol.mass_of(s) for s in symB])
            XA = coords[:n_atoms_A]
            XB = coords[n_atoms_A:]
            origin, _, _, _, B = molmol.build_dimer_frame_principal(
                XA, XB, massesA=massesA, massesB=massesB
            )
            coords = (coords - origin) @ B

        filename = f"{stemA}_{stemB}_{i:06d}.psi4geom"
        _write_psi4geom(
            out_path / filename,
            symA,
            symB,
            coords,
            n_atoms_A,
            psi4_units=psi4_units,
            charge_mult_A=charge_mult_A,
            charge_mult_B=charge_mult_B,
        )

    if hist_png:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.hist(Rs, bins=50, color="#2F6B4F", edgecolor="white")
        plt.xlabel("R (Angstrom)")
        plt.ylabel("Count")
        plt.title("Sampled R Distribution")
        plt.tight_layout()
        plt.savefig(hist_png, dpi=150)
        plt.close()


def run_propsapt_batch(
    geom_dir="psi4_geoms",
    batch_index=0,
    batch_size=500,
    out_csv=None,
    method_low="propSAPT",
    method_high=None,
    psi4_options=None,
):
    """
    Run propSAPT dipole calculations for one batch of geometry files.
    Writes one CSV per batch with body-frame coords and dipole components.
    """
    setup_psi4_defaults(psi4_options=psi4_options)

    geom_dir = Path(geom_dir)
    files = sorted(geom_dir.glob("*.psi4geom"))
    start = batch_index * batch_size
    end = start + batch_size
    batch_files = files[start:end]
    if not batch_files:
        raise ValueError("No geometry files found for this batch index.")

    rows = []
    for path in batch_files:
        geom_id = int(path.stem.split("_")[-1])
        geom_str = path.read_text()
        dimer = Dimer(geom_str)
        dipole_df = calc_property(dimer, "dipole", results=os.devnull)

        coords = np.asarray(dimer.dimer.geometry().to_array(), float)
        n_atoms_A = 2
        XA = coords[:n_atoms_A]
        XB = coords[n_atoms_A:]
        massesA = np.array([molmol.mass_of("C"), molmol.mass_of("O")])
        massesB = np.array([molmol.mass_of("C"), molmol.mass_of("O")])
        RA = molmol.center_of_mass_mass(XA, massesA)
        RB = molmol.center_of_mass_mass(XB, massesB)
        R_com = float(np.linalg.norm(RB - RA))
        origin, _, _, _, B = molmol.build_dimer_frame_principal(
            XA, XB, massesA=massesA, massesB=massesB
        )
        coords_body = (coords - origin) @ B

        def axis_vec(col):
            return np.array(
                [
                    float(dipole_df.loc["X", col]),
                    float(dipole_df.loc["Y", col]),
                    float(dipole_df.loc["Z", col]),
                ],
                dtype=float,
            )

        dipole_body_cols = {}
        for col in dipole_df.columns:
            vec_lab = axis_vec(col)
            vec_body = B.T @ vec_lab
            dipole_body_cols[f"{col}_x"] = vec_body[0]
            dipole_body_cols[f"{col}_y"] = vec_body[1]
            dipole_body_cols[f"{col}_z"] = vec_body[2]

        coord_cols, coord_order = body_frame_coord_columns(
            ["C", "O"], ["C", "O"], coords_body, n_atoms_A
        )

        rows.append(
            {
                "geom_id": geom_id,
                "method_low": method_low,
                "method_high": method_high if method_high is not None else "",
                "R": R_com,
                **coord_cols,
                **dipole_body_cols,
            }
        )

    base_cols = [
        "geom_id",
        "method_low",
        "method_high",
        "R",
    ]
    coord_cols = coord_order if rows else []
    dipole_cols = []
    if rows:
        dipole_cols = [
            k for k in rows[0].keys() if k not in (set(base_cols) | set(coord_cols))
        ]
    df = pd.DataFrame(rows, columns=base_cols + coord_cols + dipole_cols)
    if out_csv is None:
        out_csv = f"propsapt_batch_{batch_index:03d}.csv"
    df.to_csv(out_csv, index=False)


def sample_co_dimer_geometries(
    n_samples=1000,
    fileA="CO.xyz",
    fileB="CO.xyz",
    r_min=3.0,
    r_max=8.0,
    mode_frac=1 / 3,
    concentration=10.0,
    sigma_noise=0.1,
    seed=None,
    write_xyz=False,
    write_psi4_dat=False,
    psi4_units="angstrom",
    charge_mult_A="0 1",
    charge_mult_B="0 1",
    out_prefix="dimer_sample",
    psi4_options=None,
    out_csv="ml_dimer_data.csv",
    method_low="propSAPT",
    method_high=None,
):
    """
    Sample CO-CO dimer geometries and return a list of dicts.

    CSV schema (body frame, Angstrom, a.u. dipoles):
      geom_id, method_low, method_high, R,
      per-atom body coords: x_<sym>A<i>,y_<sym>A<i>,z_<sym>A<i>, then B,
      A_x,A_y,A_z, B_x,B_y,B_z, dmu_total_x,dmu_total_y,dmu_total_z
    """
    setup_psi4_defaults(psi4_options=psi4_options)
    if seed is not None:
        np.random.seed(seed)

    symA, _ = molmol.read_xyz(fileA)
    symB, _ = molmol.read_xyz(fileB)
    n_atoms_A = len(symA)
    n_atoms_B = len(symB)

    data = []
    csv_rows = []
    Rs = sample_R_beta_mode(
        n_samples, r_min, r_max, mode_frac=mode_frac, concentration=concentration
    )
    for i, R in enumerate(Rs):
        u = np.random.uniform(0, 1)
        theta = np.arccos(1 - 2 * u)
        phi = np.random.uniform(0, 2 * np.pi)
        u1, u2, u3 = np.random.rand(3)
        alphaE = 2 * np.pi * u1
        betaE = np.arccos(2 * u2 - 1)
        gammaE = 2 * np.pi * u3
        eulerA = (0.0, 0.0, 0.0)
        eulerB = (alphaE, betaE, gammaE)

        dimer_sample, symbols, coords, mol_string = molmol.merge_monomers_jacobi_XYZ(
            fileA,
            fileB,
            R=R,
            theta=theta,
            phi=phi,
            eulerA=eulerA,
            eulerB=eulerB,
            sigma_noise=sigma_noise,
        )

        entry = {
            "index": i,
            "R": R,
            "theta": theta,
            "phi": phi,
            "eulerA": eulerA,
            "eulerB": eulerB,
            "symbols": symbols,
            "coords": coords,
            "mol_string": mol_string,
            "n_atoms_A": n_atoms_A,
            "n_atoms_B": n_atoms_B,
        }
        data.append(entry)

        dimer = Dimer(mol_string)
        dipole_df = calc_property(dimer, "dipole", results=os.devnull)

        massesA = np.array([molmol.mass_of(s) for s in symA])
        massesB = np.array([molmol.mass_of(s) for s in symB])
        XA = coords[:n_atoms_A]
        XB = coords[n_atoms_A:]
        RA = molmol.center_of_mass_mass(XA, massesA)
        RB = molmol.center_of_mass_mass(XB, massesB)
        R_com = float(np.linalg.norm(RB - RA))
        origin, ex, ey, ez, B = molmol.build_dimer_frame_principal(
            XA, XB, massesA=massesA, massesB=massesB
        )
        coords_body = (coords - origin) @ B

        def axis_vec(col):
            return np.array(
                [
                    float(dipole_df.loc["X", col]),
                    float(dipole_df.loc["Y", col]),
                    float(dipole_df.loc["Z", col]),
                ],
                dtype=float,
            )

        dipole_body_cols = {}
        for col in dipole_df.columns:
            vec_lab = axis_vec(col)
            vec_body = B.T @ vec_lab
            dipole_body_cols[f"{col}_x"] = vec_body[0]
            dipole_body_cols[f"{col}_y"] = vec_body[1]
            dipole_body_cols[f"{col}_z"] = vec_body[2]

        coord_cols, coord_order = body_frame_coord_columns(
            symA, symB, coords_body, n_atoms_A
        )

        csv_rows.append(
            {
                "geom_id": i,
                "method_low": method_low,
                "method_high": method_high if method_high is not None else "",
                "R": R_com,
                **coord_cols,
                **dipole_body_cols,
            }
        )

        if write_xyz:
            filename = f"{out_prefix}_{i+1:04d}.xyz"
            with open(filename, "w") as f:
                f.write(f"{len(symbols)}\n")
                f.write(f"{len(symbols)} {n_atoms_A} {n_atoms_B}\n")
                for sym, (x, y, z) in zip(symbols, coords):
                    f.write(f"{sym} {x:.10f} {y:.10f} {z:.10f}\n")

        if write_psi4_dat:
            dat_name = f"{out_prefix}_{i+1:04d}.dat"
            with open(dat_name, "w") as f:
                f.write("symmetry c1\n")
                f.write("no_com\n")
                f.write("no_reorient\n")
                f.write(f"units {psi4_units}\n")
                f.write(f"{charge_mult_A}\n")
                for sym, (x, y, z) in zip(symA, coords[:n_atoms_A]):
                    f.write(f"{sym} {x:.10f} {y:.10f} {z:.10f}\n")
                f.write("--\n")
                f.write(f"{charge_mult_B}\n")
                for sym, (x, y, z) in zip(symB, coords[n_atoms_A:]):
                    f.write(f"{sym} {x:.10f} {y:.10f} {z:.10f}\n")

    base_cols = [
        "geom_id",
        "method_low",
        "method_high",
        "R",
    ]
    coord_cols = coord_order if csv_rows else []
    dipole_cols = []
    if csv_rows:
        dipole_cols = [
            k for k in csv_rows[0].keys() if k not in (set(base_cols) | set(coord_cols))
        ]
    columns = base_cols + coord_cols + dipole_cols
    csv_df = pd.DataFrame(csv_rows, columns=columns)
    if out_csv:
        csv_df.to_csv(out_csv, index=False)
    return data, csv_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML dimer sampling utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="Generate psi4 geometry files.")
    gen_parser.add_argument("--n-samples", type=int, default=50000)
    gen_parser.add_argument("--fileA", default="CO.xyz")
    gen_parser.add_argument("--fileB", default="CO.xyz")
    gen_parser.add_argument("--r-min", type=float, default=3.0)
    gen_parser.add_argument("--r-max", type=float, default=8.0)
    gen_parser.add_argument("--mode-frac", type=float, default=1 / 3)
    gen_parser.add_argument("--concentration", type=float, default=10.0)
    gen_parser.add_argument("--sigma-noise", type=float, default=0.1)
    gen_parser.add_argument("--seed", type=int, default=None)
    gen_parser.add_argument("--out-dir", default="psi4_geoms")
    gen_parser.add_argument("--psi4-units", default="angstrom")
    gen_parser.add_argument("--charge-mult-A", default="0 1")
    gen_parser.add_argument("--charge-mult-B", default="0 1")
    gen_parser.add_argument("--body-frame", action="store_true")
    gen_parser.add_argument("--hist-png", default="R_hist.png")

    batch_parser = subparsers.add_parser("batch", help="Run propSAPT for one batch.")
    batch_parser.add_argument("--geom-dir", default="psi4_geoms")
    batch_parser.add_argument("--batch-index", type=int, default=0)
    batch_parser.add_argument("--batch-size", type=int, default=500)
    batch_parser.add_argument("--out-csv", default=None)
    batch_parser.add_argument("--method-low", default="propSAPT")
    batch_parser.add_argument("--method-high", default=None)

    sample_parser = subparsers.add_parser("sample", help="Sample + compute propSAPT directly.")
    sample_parser.add_argument("--n-samples", type=int, default=1000)
    sample_parser.add_argument("--fileA", default="CO.xyz")
    sample_parser.add_argument("--fileB", default="CO.xyz")
    sample_parser.add_argument("--r-min", type=float, default=3.0)
    sample_parser.add_argument("--r-max", type=float, default=8.0)
    sample_parser.add_argument("--mode-frac", type=float, default=1 / 3)
    sample_parser.add_argument("--concentration", type=float, default=10.0)
    sample_parser.add_argument("--sigma-noise", type=float, default=0.1)
    sample_parser.add_argument("--seed", type=int, default=None)
    sample_parser.add_argument("--out-csv", default="ml_dimer_data.csv")
    sample_parser.add_argument("--method-low", default="propSAPT")
    sample_parser.add_argument("--method-high", default=None)

    args = parser.parse_args()

    if args.command == "generate":
        generate_psi4geom_files(
            n_samples=args.n_samples,
            fileA=args.fileA,
            fileB=args.fileB,
            r_min=args.r_min,
            r_max=args.r_max,
            mode_frac=args.mode_frac,
            concentration=args.concentration,
            sigma_noise=args.sigma_noise,
            seed=args.seed,
            out_dir=args.out_dir,
            psi4_units=args.psi4_units,
            charge_mult_A=args.charge_mult_A,
            charge_mult_B=args.charge_mult_B,
            body_frame=args.body_frame,
            hist_png=args.hist_png,
        )
    elif args.command == "batch":
        run_propsapt_batch(
            geom_dir=args.geom_dir,
            batch_index=args.batch_index,
            batch_size=args.batch_size,
            out_csv=args.out_csv,
            method_low=args.method_low,
            method_high=args.method_high,
        )
    elif args.command == "sample":
        samples, dipoles = sample_co_dimer_geometries(
            n_samples=args.n_samples,
            fileA=args.fileA,
            fileB=args.fileB,
            r_min=args.r_min,
            r_max=args.r_max,
            mode_frac=args.mode_frac,
            concentration=args.concentration,
            sigma_noise=args.sigma_noise,
            seed=args.seed,
            out_csv=args.out_csv,
            method_low=args.method_low,
            method_high=args.method_high,
        )
        print(f"Generated {len(samples)} samples.")
        print(dipoles.head())
