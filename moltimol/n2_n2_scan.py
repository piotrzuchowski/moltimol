import numpy as np
import psi4

import moltimol as molmol


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


def n2_dimer_geom(R, bond_length=1.0977):
    """
    Build a parallel N2-N2 dimer with COMs separated by R (angstrom).
    Both monomers are aligned along the z-axis.
    """
    dz = bond_length / 2.0
    zA = (-dz, dz)
    zB = (R - dz, R + dz)
    geom = (
        "0 1\n"
        "symmetry c1\n"
        "no_reorient\n"
        "no_com\n"
        "units angstrom\n"
        f"N 0.0 0.0 {zA[0]:.6f}\n"
        f"N 0.0 0.0 {zA[1]:.6f}\n"
        "--\n"
        "0 1\n"
        f"N 0.0 0.0 {zB[0]:.6f}\n"
        f"N 0.0 0.0 {zB[1]:.6f}\n"
    )
    return geom


def psi4_dipole(mol_string):
    mol = psi4.geometry(mol_string)
    _, wfn = psi4.properties(
        "HF",
        properties=["DIPOLE"],
        molecule=mol,
        return_wfn=True,
    )
    for var in ("CURRENT DIPOLE", "DIPOLE", "SCF DIPOLE"):
        try:
            return np.array(wfn.variable(var), dtype=float)
        except Exception:
            continue
    for var in ("CURRENT DIPOLE", "DIPOLE", "SCF DIPOLE"):
        try:
            return np.array(psi4.core.variable(var), dtype=float)
        except Exception:
            continue
    raise RuntimeError("Dipole not found in Psi4 wavefunction variables.")


def scan_n2_n2_dipole(
    r_min=3.0,
    r_max=8.0,
    step=0.1,
    out_csv="n2_n2_dipole_scan.csv",
    geom_dir=None,
    method_low="HF",
    method_high=None,
):
    psi4.set_memory("2 GB")
    psi4.set_num_threads(2)
    psi4.set_options(
        {
            "basis": "aug-cc-pvdz",
            "scf_type": "df",
            "e_convergence": 1e-8,
            "d_convergence": 1e-8,
        }
    )

    rows = []
    if geom_dir:
        import os

        os.makedirs(geom_dir, exist_ok=True)
    symA = ["N", "N"]
    symB = ["N", "N"]
    massesA = np.array([molmol.mass_of(s) for s in symA], dtype=float)
    massesB = np.array([molmol.mass_of(s) for s in symB], dtype=float)

    r = r_min
    geom_id = 0
    while r <= r_max + 1e-9:
        geom = n2_dimer_geom(r)
        mu_lab = psi4_dipole(geom)

        XA = np.array([[0.0, 0.0, -0.5 * 1.0977], [0.0, 0.0, 0.5 * 1.0977]])
        XB = np.array([[0.0, 0.0, r - 0.5 * 1.0977], [0.0, 0.0, r + 0.5 * 1.0977]])
        coords = np.vstack([XA, XB])
        origin, _, _, _, B = molmol.build_dimer_frame_principal(
            XA, XB, massesA=massesA, massesB=massesB
        )
        coords_body = (coords - origin) @ B
        mu_body = B.T @ mu_lab

        coord_cols, coord_order = body_frame_coord_columns(
            symA, symB, coords_body, len(symA)
        )

        rows.append(
            {
                "geom_id": geom_id,
                "method_low": method_low,
                "method_high": method_high if method_high is not None else "",
                "R": round(r, 6),
                **coord_cols,
                "dmu_total_x": float(mu_body[0]),
                "dmu_total_y": float(mu_body[1]),
                "dmu_total_z": float(mu_body[2]),
            }
        )
        if geom_dir:
            fname = f"n2_n2_{round(r, 3):06.3f}.psi4geom"
            with open(os.path.join(geom_dir, fname), "w") as f:
                f.write(geom)
        r += step
        geom_id += 1

    base_cols = ["geom_id", "method_low", "method_high", "R"]
    coord_cols = coord_order if rows else []
    dipole_cols = []
    if rows:
        dipole_cols = [
            k for k in rows[0].keys() if k not in (set(base_cols) | set(coord_cols))
        ]
    import pandas as pd

    df = pd.DataFrame(rows, columns=base_cols + coord_cols + dipole_cols)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    scan_n2_n2_dipole(geom_dir="psi4_geoms_n2n2")
