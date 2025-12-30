import numpy as np
import warnings
import psi4

# Try to get atomic masses from ASE; fall back to a small built-in table.
try:
    from ase.data import atomic_masses, atomic_numbers

    def mass_of(symbol):
        """
        Return atomic mass in atomic mass units (u) for a given element symbol using ASE data.
        """
        # try direct lookup first, then some common capitalizations
        Z = atomic_numbers.get(symbol)
        if Z is None:
            for s in (symbol.capitalize(), symbol.title(), symbol.upper()):
                Z = atomic_numbers.get(s)
                if Z is not None:
                    break
        if Z is None:
            raise KeyError(f"Element symbol '{symbol}' not found in ASE atomic_numbers.")
        return float(atomic_masses[Z])

except Exception:
    warnings.warn(
        "ASE not available. Falling back to a small built-in mass table. "
        "Install ASE (`pip install ase`) for full periodic table support."
    )
    builtin_masses = {
        "H": 1.0079,
        "C": 12.011,
        "N": 14.007,
        "O": 15.999,
        "F": 18.998,
        "Cl": 35.45,
        # add more as needed
    }

    def mass_of(symbol):
        try:
            return float(builtin_masses[symbol])
        except KeyError:
            raise KeyError(
                f"Element '{symbol}' not in built-in table. Install ASE for full coverage."
            )


def read_xyz(filename):
    """
    Reads an XYZ file and returns symbols and Cartesian coordinates (in Å).
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    n = int(lines[0])
    coords = []
    symbols = []

    for line in lines[2:2 + n]:
        parts = line.split()
        sym = parts[0]
        x, y, z = map(float, parts[1:4])
        symbols.append(sym)
        coords.append([x, y, z])

    return np.array(symbols), np.array(coords)


def rotation_matrix_from_euler(alpha, beta, gamma):
    """
    Rotation matrix R = Rz(gamma) * Ry(beta) * Rz(alpha)
    Z-Y-Z convention (common in chemistry).
    """
    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sa, sb, sg = np.sin(alpha), np.sin(beta), np.sin(gamma)

    Rz1 = np.array([[ca, -sa, 0],
                    [sa, ca, 0],
                    [0, 0, 1]])

    Ry = np.array([[cb, 0, sb],
                   [0, 1, 0],
                   [-sb, 0, cb]])

    Rz2 = np.array([[cg, -sg, 0],
                    [sg, cg, 0],
                    [0, 0, 1]])

    return Rz2 @ Ry @ Rz1


def jacobi_vector(R, theta, phi):
    """
    Build Jacobi vector from spherical coordinates.
    theta ∈ [0, π], phi ∈ [0, 2π]
    """
    return np.array([
        R * np.sin(theta) * np.cos(phi),
        R * np.sin(theta) * np.sin(phi),
        R * np.cos(theta)
    ])


def center_of_mass_symbols(symbols, coords):
    """
    Shift coordinates so that COM = 0.
    """
    # mass_of(...) returns atomic mass in atomic mass units (u)
    m = np.array([mass_of(s) for s in symbols]).reshape(-1, 1)
    com = np.sum(coords * m, axis=0) / np.sum(m)
    return coords - com

#
# utilities for projecting to dimer frame  
#

def _unit(v, eps=1e-14):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n

def center_of_geometry(X):
    X = np.asarray(X, float)
    return X.mean(axis=0)

def center_of_mass_mass(X, masses):
    X = np.asarray(X, float)
    m = np.asarray(masses, float)
    return (X * m[:, None]).sum(axis=0) / m.sum()


def center_of_mass(a, b):
    """
    Center of mass utility supporting:
      - (symbols, coords) -> COM-shifted coords
      - (coords, masses)  -> COM vector
    """
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.ndim == 1 and b_arr.ndim == 2:
        return center_of_mass_symbols(a, b)
    if a_arr.ndim == 2 and b_arr.ndim == 1:
        return center_of_mass_mass(a, b)
    raise ValueError("center_of_mass expects (symbols, coords) or (coords, masses).")

def axis_from_atoms(X, i, j):
    """Unit vector from atom i to atom j in coordinates X."""
    X = np.asarray(X, float)
    return _unit(X[j] - X[i])

def build_dimer_frame(
    XA, XB,
    refA="cog", refB="cog",
    massesA=None, massesB=None,
    axisA=None, axisB=None,
    eps=1e-10,
):
    """
    Returns (origin, ex, ey, ez, B) where columns of B are [ex, ey, ez].
    You can project vectors with mu_body = B.T @ mu_lab.
    """

    XA = np.asarray(XA, float)
    XB = np.asarray(XB, float)

    # Reference points RA, RB
    if refA == "com":
        if massesA is None: raise ValueError("massesA required for refA='com'")
        RA = center_of_mass(XA, massesA)
    else:
        RA = center_of_geometry(XA)

    if refB == "com":
        if massesB is None: raise ValueError("massesB required for refB='com'")
        RB = center_of_mass(XB, massesB)
    else:
        RB = center_of_geometry(XB)

    # Dimer axis ez (A -> B)
    R = RB - RA
    ez = _unit(R)

    # Choose an in-plane vector u to define ex (prefer axisA, else axisB, else fallback axis)
    def proj_perp(u, ez):
        return u - np.dot(u, ez) * ez

    # Build u from axis specs if provided:
    # axisA can be a 3-vector or a tuple (i,j) atom indices in A
    u_candidates = []
    if axisA is not None:
        if isinstance(axisA, tuple):
            u_candidates.append(axis_from_atoms(XA, axisA[0], axisA[1]))
        else:
            u_candidates.append(_unit(axisA))
    if axisB is not None:
        if isinstance(axisB, tuple):
            u_candidates.append(axis_from_atoms(XB, axisB[0], axisB[1]))
        else:
            u_candidates.append(_unit(axisB))

    # Always provide a safe fallback
    u_candidates += [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]

    ex = None
    for u in u_candidates:
        xp = proj_perp(u, ez)
        if np.linalg.norm(xp) > eps:
            ex = _unit(xp)
            break
    if ex is None:
        raise ValueError("Failed to build a stable ex axis (degenerate geometry).")

    ey = _unit(np.cross(ez, ex))  # right-handed
    # Re-orthogonalize ex (numerical safety)
    ex = _unit(np.cross(ey, ez))

    # Choose an origin for reporting (midpoint between monomer refs is common)
    origin = 0.5 * (RA + RB)

    # Basis matrix: columns are basis vectors in lab XYZ
    B = np.column_stack([ex, ey, ez])
    return origin, ex, ey, ez, B

def project_to_dimer_frame(mu_lab, B):
    """mu_body = [mu_x, mu_y, mu_z] in the dimer frame."""
    mu_lab = np.asarray(mu_lab, float)
    return B.T @ mu_lab



def add_cartesian_noise(symbols, coords, sigma=0.1):
    """
    Add independent Gaussian noise to each atom and Cartesian component.

    coords : (N,3) numpy array (bohr)
    sigma  : noise amplitude (bohr)
    """
    noise = np.random.normal(scale=sigma, size=coords.shape)
    noisy = coords + noise
    return center_of_mass(symbols, noisy)


#  optional alternative noise function with element-dependent scaling
#  def add_cartesian_noise(symbols, coords, sigma=0.02):
#     noisy = coords.copy()
#     for i, s in enumerate(symbols):
#         scale = sigma
#         if s not in ["H"]:
#             scale *= 0.5
#         noisy[i] += np.random.normal(scale=scale, size=3)
#     return center_of_mass(symbols, noisy)


def merge_monomers_jacobi(A, B,
                          R, theta, phi,
                          eulerA=(0, 0, 0), eulerB=(0, 0, 0)):
    """
    Merge two monomers A and B using Jacobi coordinates.
    A, B: coordinate arrays (COM-centered)
    R, theta, phi: Jacobi coordinates
    eulerA, eulerB: Euler angles (alpha, beta, gamma)
    """
    # 1. Rotations
    RA = rotation_matrix_from_euler(*eulerA)
    RB = rotation_matrix_from_euler(*eulerB)

    A_rot = A @ RA.T
    B_rot = B @ RB.T

    # 2. Jacobi vector
    R_vec = jacobi_vector(R, theta, phi)

    # 3. Translate B by Jacobi vector
    B_global = B_rot + R_vec

    # 4. Merge
    coords = np.vstack([A_rot, B_global])

    return coords


def merge_monomers_jacobi_XYZ(
    xyzA, xyzB,
    R, theta, phi,
    eulerA=(0, 0, 0),
    eulerB=(0, 0, 0),
    sigma_noise=0.0
):
    """
    Given two XYZ files (each monomer), build a Jacobi dimer.
    R in bohr, coords in bohr.
    """

    # --- Read monomers ---
    symA, A = read_xyz(xyzA)
    symB, B = read_xyz(xyzB)

    # --- Center of mass ---
    A = center_of_mass(symA, A)
    B = center_of_mass(symB, B)
    # --- Add noise ---
    if sigma_noise > 0.0:
        A = add_cartesian_noise(symA, A, sigma=sigma_noise)
        B = add_cartesian_noise(symB, B, sigma=sigma_noise)

    # --- Rotations ---
    RA = rotation_matrix_from_euler(*eulerA)
    RB = rotation_matrix_from_euler(*eulerB)

    A_rot = A @ RA.T
    B_rot = B @ RB.T

    # --- Jacobi placement ---
    R_vec = jacobi_vector(R, theta, phi)
    B_global = B_rot + R_vec

    # --- merge ---
    coords = np.vstack([A_rot, B_global])
    symbols = list(symA) + list(symB)

    # --- Create Psi4 molecule ---
    coords_A = coords[:len(symA)]
    coords_B = coords[len(symA):]

    mol_string = "0 1\n"
    for s, c in zip(symA, coords_A):
        mol_string += f"{s} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}\n"

    mol_string += "--\n"
    mol_string += "0 1\n"
    for s, c in zip(symB, coords_B):
        mol_string += f"{s} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}\n"

    return psi4.geometry(mol_string), symbols, coords, mol_string
