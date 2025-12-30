import moltimol as molmol
import numpy as np
import psi4

# Set memory and processors for psi4
psi4.set_memory('2 GB')
psi4.set_num_threads(2)

# --- Example of using merge_monomers_jacobi_XYZ ---

# Define the parameters for the dimer
eulerA_0 = (0, 0, 0) 

# Set the basis set for the calculation
psi4.set_options({
    'basis': 'jun-cc-pvdz',
    'freeze_core': 'true',
    'guess': 'sad',
    'save_jk': True,
    "DF_BASIS_SAPT": "jun-cc-pvdz-ri",
    "DF_BASIS_SCF": "jun-cc-pvdz-jkfit"
})

#
# # Generate multiple samples with noise
#
Nsamples=3
symA, _ = molmol.read_xyz("CO.xyz")
symB, _ = molmol.read_xyz("CO.xyz")
n_atoms_A = len(symA)
n_atoms_B = len(symB)
for i in range(Nsamples):
    #draw R from uniform distribution
    R = np.random.uniform(3.0, 8.0)  
    #draw theta from cos distribution 
    u = np.random.uniform(0, 1)
    theta = np.arccos(1 - 2 * u)
    #draw phi from uniform distribution
    phi = np.random.uniform(0, 2 * np.pi)
    #draw euler angles from uniform distribution
    u1, u2, u3 = np.random.rand(3)
    alphaE = 2*np.pi*u1
    betaE  = np.arccos(2*u2 - 1)     # cos(beta) uniform in [-1,1]
    gammaE = 2*np.pi*u3
    eulerB = (alphaE, betaE, gammaE)
    dimer_sample, symbols, coords, mol_string = molmol.merge_monomers_jacobi_XYZ(
        "CO.xyz",
        "CO.xyz",
        R=R,
        theta=theta,
        phi=phi,
        eulerA=eulerA_0,
        eulerB=eulerB,
        sigma_noise=0.1  # Increased noise for sampling
    )
    sample_filename = f"dimer_sample_{i+1:03d}.xyz"
    total_atoms = len(symbols)
    with open(sample_filename, "w") as f:
        f.write(f"{total_atoms}\n")
        f.write(f"{total_atoms} {n_atoms_A} {n_atoms_B}\n")
        for sym, (x, y, z) in zip(symbols, coords):
            f.write(f"{sym} {x:.10f} {y:.10f} {z:.10f}\n")

    print(f"Wrote geometry: {sample_filename}")
