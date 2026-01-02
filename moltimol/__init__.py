"""
Lightweight helpers for generating and transforming molecular dimers.

Conventions:
- Coordinates are in angstrom unless otherwise noted.
- Frame builders return (origin, ex, ey, ez, B) with B columns [ex, ey, ez].
  Transform coords: X_body = (X - origin) @ B
  Transform vectors: v_body = B.T @ v_lab
"""

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


def center_of_mass(symbols, coords):
    """
    Shift coordinates so that COM = 0 (symbols are element symbols).
    """
    # mass_of(...) returns atomic mass in atomic mass units (u)
    m = np.array([mass_of(s) for s in symbols]).reshape(-1, 1)
    com = np.sum(coords * m, axis=0) / np.sum(m)
    return coords - com

def center_of_mass_mass(coords, masses):
    """
    Return COM vector from coordinates and masses.
    """
    X = np.asarray(coords, float)
    m = np.asarray(masses, float)
    return (X * m[:, None]).sum(axis=0) / m.sum()

def _unit(v, eps=1e-14):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n

def inertia_tensor(X, masses):
    """Inertia tensor about the COM. Units: mass*length^2 (consistent with X units)."""
    X = np.asarray(X, float)
    m = np.asarray(masses, float)
    R = center_of_mass_mass(X, m)
    r = X - R
    I = np.zeros((3, 3), float)
    for mi, ri in zip(m, r):
        r2 = float(np.dot(ri, ri))
        I += mi * (r2 * np.eye(3) - np.outer(ri, ri))
    return I


def principal_axis(X, masses, which="min", eps=1e-12):
    """
    Return a signed principal axis eigenvector (unit).
    which="min" -> axis of smallest principal moment (good for linear molecules)
    which="max" -> axis of largest principal moment (often plane-normal for planar molecules)
    """
    I = inertia_tensor(X, masses)
    vals, vecs = np.linalg.eigh(I)  # columns are eigenvectors
    idx = int(np.argmin(vals) if which == "min" else np.argmax(vals))
    u = vecs[:, idx]

    # Fix sign deterministically so it doesn't randomly flip
    X = np.asarray(X, float)
    m = np.asarray(masses, float)
    R = center_of_mass_mass(X, m)
    maxm = np.max(m)
    cand = np.where(np.isclose(m, maxm))[0]
    if len(cand) > 1:
        d = np.linalg.norm(X[cand] - R, axis=1)
        anchor_idx = int(cand[np.argmax(d)])
    else:
        anchor_idx = int(cand[0])
    anchor = X[anchor_idx] - R
    if np.linalg.norm(anchor) > eps and np.dot(u, anchor) < 0.0:
        u = -u

    return _unit(u, eps)

def build_dimer_frame_principal(
    XA, XB,
    massesA, massesB,
    eps=1e-10,
):
    """
    Build a dimer frame using COMs and principal axes of inertia.
    Returns (origin, ex, ey, ez, B) where columns of B are [ex, ey, ez].
    """
    XA = np.asarray(XA, float)
    XB = np.asarray(XB, float)

    RA = center_of_mass_mass(XA, massesA)
    RB = center_of_mass_mass(XB, massesB)
    ez = _unit(RB - RA, eps=eps)

    uA = principal_axis(XA, massesA, which="min", eps=eps)
    uB = principal_axis(XB, massesB, which="min", eps=eps)

    def proj_perp(u, ez):
        return u - np.dot(u, ez) * ez

    ex = None
    for u in (uA, uB, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
        xp = proj_perp(u, ez)
        if np.linalg.norm(xp) > eps:
            ex = _unit(xp, eps=eps)
            break
    if ex is None:
        raise ValueError("Failed to build a stable ex axis (degenerate geometry).")

    ey = _unit(np.cross(ez, ex), eps=eps)
    ex = _unit(np.cross(ey, ez), eps=eps)

    origin = 0.5 * (RA + RB)
    B = np.column_stack([ex, ey, ez])
    return origin, ex, ey, ez, B

def add_cartesian_noise(symbols, coords, sigma=0.1):
    """
    Add independent Gaussian noise to each atom and Cartesian component.

    coords : (N,3) numpy array (angstrom)
    sigma  : noise amplitude (angstrom)
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
    R in angstrom, coords in angstrom.
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

    mol_string = (
        "units angstrom\n"
        "symmetry c1\n"
        "no_reorient\n"
        "no_com\n"
        "0 1\n"
    )
    for s, c in zip(symA, coords_A):
        mol_string += f"{s} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}\n"

    mol_string += "--\n"
    mol_string += "0 1\n"
    for s, c in zip(symB, coords_B):
        mol_string += f"{s} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}\n"

    return psi4.geometry(mol_string), symbols, coords, mol_string



def build_dimer_frame_COM(
    XA, XB,
    massesA=None, massesB=None,
    uA=None, uB=None,
    principalA="min", principalB="min",
    origin="A",
    eps=1e-10
):
    """
    Build a right-handed dimer frame.
      ez: along COM(A)->COM(B)
      ex: in plane perpendicular to  ez, from projected reference vector (uA preferred, then uB, then lab axis)
      ey: ez x ex

    Inputs:
      XA, XB : (NA,3), (NB,3) coordinates in any length unit
      massesA, massesB : arrays of masses (optional; if None uses equal weights)
      uA, uB : optional reference direction vectors in LAB frame (3,) for monomer A/B
      principalA/principalB : if uA/uB not given, use principal axis ("min" or "max")
      origin: "midpoint" (between COMs) or "A" (COM of A) or "B" (COM of B)

    Returns:
      origin_vec, ex, ey, ez, B
      where B has columns [ex, ey, ez] in LAB coordinates.
      Convert coords: X_body = (X - origin_vec) @ B
      Convert vectors: v_body = B.T @ v_lab
    """
    XA = np.asarray(XA, float)
    XB = np.asarray(XB, float)

    if massesA is None:
        massesA = np.ones(len(XA))
    if massesB is None:
        massesB = np.ones(len(XB))

    RA = center_of_mass_mass(XA, massesA)
    RB = center_of_mass_mass(XB, massesB)

    ez = _unit(RB - RA, eps)
    # choose reference vectors if not provided
    if uA is None:
        uA = principal_axis(XA, massesA, which=principalA, eps=eps)
    else:
        uA = _unit(uA, eps)

    if uB is None:
        uB = principal_axis(XB, massesB, which=principalB, eps=eps)
    else:
        uB = _unit(uB, eps)

    # build ex by picking the candidate whose projection onto plane ⟂ ez is largest
    def proj_perp(u):
        return u - np.dot(u, ez) * ez

    candidates = []
    for u in (uA, uB, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
        if u is None:
            continue
        xp = proj_perp(u)
        candidates.append((np.linalg.norm(xp), xp))

    candidates.sort(key=lambda t: t[0], reverse=True)
    if candidates[0][0] < eps:
        raise ValueError("Failed to define ex: all reference vectors parallel to ez (axial degeneracy).")

    ex = _unit(candidates[0][1], eps)
    ey = _unit(np.cross(ez, ex), eps)
    ex = _unit(np.cross(ey, ez), eps)  # re-orthogonalize

    if origin == "midpoint":
        origin_vec = 0.5 * (RA + RB)
    elif origin == "A":
        origin_vec = RA
    elif origin == "B":
        origin_vec = RB
    else:
        raise ValueError("origin must be 'midpoint', 'A', or 'B'.")

    B = np.column_stack([ex, ey, ez])
    return origin_vec, ex, ey, ez, B
