import numpy as np

import moltimol as molmol


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
):
    """
    Sample CO-CO dimer geometries and return a list of dicts.
    """
    if seed is not None:
        np.random.seed(seed)

    symA, _ = molmol.read_xyz(fileA)
    symB, _ = molmol.read_xyz(fileB)
    n_atoms_A = len(symA)
    n_atoms_B = len(symB)

    data = []
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

        # Placeholder: compute dipole components for this geometry here.
        # Example: dipole = your_blackbox(coords, symbols, mol_string)

    return data


if __name__ == "__main__":
    samples = sample_co_dimer_geometries(n_samples=1000, write_xyz=False, seed=0)
    print(f"Generated {len(samples)} samples.")
