import numpy as np
import pandas as pd
import psi4

import moltimol as molmol
from prop_sapt import Dimer, calc_property


def sample_co_dimer_geometries(
    n_samples=1000,
    fileA="CO.xyz",
    fileB="CO.xyz",
    r_min=3.0,
    r_max=8.0,
    sigma_noise=0.1,
    seed=None,
    write_xyz=False,
    write_psi4_dat=False,
    psi4_units="bohr",
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
      x_CA,y_CA,z_CA, x_OA,y_OA,z_OA, x_CB,y_CB,z_CB, x_OB,y_OB,z_OB,
      A_x,A_y,A_z, B_x,B_y,B_z, dmu_total_x,dmu_total_y,dmu_total_z
    """
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
    if seed is not None:
        np.random.seed(seed)

    symA, _ = molmol.read_xyz(fileA)
    symB, _ = molmol.read_xyz(fileB)
    n_atoms_A = len(symA)
    n_atoms_B = len(symB)

    data = []
    csv_rows = []
    for i in range(n_samples):
        R = np.random.uniform(r_min, r_max)
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
        dipole_df = calc_property(dimer, "dipole")

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

        muA_lab = axis_vec("x1_pol,r_A") + axis_vec("x1_exch,r_A")
        muB_lab = axis_vec("x1_pol,r_B") + axis_vec("x1_exch,r_B")
        mu_total_lab = axis_vec("x_induced")

        muA_body = B.T @ muA_lab
        muB_body = B.T @ muB_lab
        mu_total_body = B.T @ mu_total_lab

        body_CA, body_OA = coords_body[0], coords_body[1]
        body_CB, body_OB = coords_body[n_atoms_A], coords_body[n_atoms_A + 1]

        csv_rows.append(
            {
                "geom_id": i,
                "method_low": method_low,
                "method_high": method_high if method_high is not None else "",
                "R": R_com,
                "x_CA": body_CA[0],
                "y_CA": body_CA[1],
                "z_CA": body_CA[2],
                "x_OA": body_OA[0],
                "y_OA": body_OA[1],
                "z_OA": body_OA[2],
                "x_CB": body_CB[0],
                "y_CB": body_CB[1],
                "z_CB": body_CB[2],
                "x_OB": body_OB[0],
                "y_OB": body_OB[1],
                "z_OB": body_OB[2],
                "A_x": muA_body[0],
                "A_y": muA_body[1],
                "A_z": muA_body[2],
                "B_x": muB_body[0],
                "B_y": muB_body[1],
                "B_z": muB_body[2],
                "dmu_total_x": mu_total_body[0],
                "dmu_total_y": mu_total_body[1],
                "dmu_total_z": mu_total_body[2],
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

    csv_df = pd.DataFrame(csv_rows)
    if out_csv:
        csv_df.to_csv(out_csv, index=False)
    return data, csv_df


if __name__ == "__main__":
    samples, dipoles = sample_co_dimer_geometries(n_samples=3, write_xyz=False, seed=0)
    print(f"Generated {len(samples)} samples.")
    print(dipoles.head())
