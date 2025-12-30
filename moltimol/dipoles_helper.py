import numpy as np
import psi4

from . import center_of_mass

# ------------------ TT damping + tensors (a.u.) ------------------

def f_tt(n, x):
    """Tang–Toennies f_n(x) = 1 - e^{-x} sum_{k=0}^n x^k/k!  (scalar x)."""
    s, term = 0.0, 1.0
    for k in range(n + 1):
        s += term
        if k < n:
            term *= x / (k + 1)
    return 1.0 - np.exp(-x) * s

def T_dipole_tt(R, b):
    """
    TT-damped dipole tensor:
      T = 3 f5(bR) RR^T / R^5 - f3(bR) I / R^3
    """
    R = np.asarray(R, float)
    r2 = float(R @ R)
    if r2 == 0.0:
        raise ValueError("|R|=0")
    r = np.sqrt(r2)
    x = b * r
    f3 = f_tt(3, x)
    f5 = f_tt(5, x)
    return 3.0 * f5 * np.outer(R, R) / r**5 - f3 * np.eye(3) / r**3

def E_charge_tt(R, b):
    """
    TT-damped electric field kernel for a point charge:
      E = f3(bR) * R / R^3
    (charge q multiplies this).
    """
    R = np.asarray(R, float)
    r2 = float(R @ R)
    if r2 == 0.0:
        raise ValueError("|R|=0")
    r = np.sqrt(r2)
    return f_tt(3, b * r) * R / r**3


# ------------------ geometry helpers ------------------

def _center_of_mass(X, masses=None):
    X = np.asarray(X, float)
    if masses is None:
        return X.mean(axis=0)
    return center_of_mass(X, masses)

def principal_axis(X, masses=None, use_psi4=True):
    """
    Returns a unit vector along a principal axis of inertia (smallest eigenvalue),
    which is robust for linear monomers (CO, CO2).
    """
    if use_psi4 and isinstance(X, psi4.core.Molecule):
        try:
            I = X.inertia_tensor()
            if hasattr(I, "to_array"):
                I = I.to_array()
            else:
                I = np.array(I)
        except Exception:
            I = None
    else:
        I = None

    if I is None:
        X = np.asarray(X, float)
        C = _center_of_mass(X, masses)
        Y = X - C
        if masses is None:
            m = np.ones(len(X))
        else:
            m = np.asarray(masses, float)

        # Inertia tensor: I = sum m (r^2 I - r r^T)
        I = np.zeros((3, 3))
        for ri, mi in zip(Y, m):
            r2 = float(ri @ ri)
            I += mi * (r2 * np.eye(3) - np.outer(ri, ri))

    w, V = np.linalg.eigh(I)
    v = V[:, np.argmin(w)]
    n = np.linalg.norm(v)
    if n < 1e-14:
        raise ValueError("Degenerate inertia axis.")
    return v / n

def build_dimer_frame(XA, XB, massesA=None, massesB=None, eps=1e-10):
    """
    Dimer frame:
      ez = unit(COM_B - COM_A)
      ex = unit( uA projected perpendicular to ez )  (uA = principal axis of monomer A)
      ey = ez x ex
    Returns origin (midpoint of COMs), basis matrix B=[ex,ey,ez] as columns.
    """
    RA = _center_of_mass(XA, massesA)
    RB = _center_of_mass(XB, massesB)
    RAB = RB - RA
    nR = np.linalg.norm(RAB)
    if nR < eps:
        raise ValueError("Monomer reference points coincide.")
    ez = RAB / nR

    uA = principal_axis(XA, massesA)
    # if uA nearly parallel to ez, try monomer B axis
    if np.linalg.norm(uA - (uA @ ez) * ez) < eps:
        uA = principal_axis(XB, massesB)

    xp = uA - (uA @ ez) * ez
    if np.linalg.norm(xp) < eps:
        # last resort fallback
        cand = np.array([1.0, 0.0, 0.0])
        xp = cand - (cand @ ez) * ez
        if np.linalg.norm(xp) < eps:
            cand = np.array([0.0, 1.0, 0.0])
            xp = cand - (cand @ ez) * ez

    ex = xp / np.linalg.norm(xp)
    ey = np.cross(ez, ex)
    ey = ey / np.linalg.norm(ey)
    # re-orthogonalize ex
    ex = np.cross(ey, ez)
    ex = ex / np.linalg.norm(ex)

    origin = 0.5 * (RA + RB)
    B = np.column_stack([ex, ey, ez])
    return origin, B


# ------------------ main: charges + polarizable atoms ------------------

def dimer_dipole_charges_polarizable(
    XA, XB,
    qA, qB,
    alphaA, alphaB,
    massesA=None, massesB=None,
    mutual=True,
    include_intra=False,
    c_scale=2.8,
):
    """
    Compute total dipole (lab and dimer-frame) for a dimer using:
      - permanent dipole from atomic charges
      - induced atomic dipoles from TT-damped charge fields
      - optional mutual induction among polarizable atoms (linear solve)
    All in atomic units.

    Inputs:
      XA (NA,3), XB (NB,3): coordinates in bohr
      qA (NA,), qB (NB,)  : atomic partial charges (e)
      alphaA (NA,), alphaB (NB,): isotropic polarizabilities (a0^3), can be 0 for nonpolarizable atoms
      mutual: solve self-consistent induction (recommended)
      include_intra: include intramolecular polarization couplings (often False for “intermolecular-only” baseline)
      c_scale: TT range factor; b_ij = c_scale / ell_ij. (larger -> stronger damping at given R)

    Returns dict with:
      mu_lab (3,), mu_body (3,), mu_perm_q (3,), mu_ind (N,3), origin, B
    """
    XA = np.asarray(XA, float); XB = np.asarray(XB, float)
    qA = np.asarray(qA, float); qB = np.asarray(qB, float)
    alphaA = np.asarray(alphaA, float); alphaB = np.asarray(alphaB, float)

    # combine
    X = np.vstack([XA, XB])
    q = np.concatenate([qA, qB])
    alpha = np.concatenate([alphaA, alphaB])
    mon = np.array([0]*len(XA) + [1]*len(XB), dtype=int)

    # origin convention (only matters if total charge != 0)
    origin, B = build_dimer_frame(XA, XB, massesA=massesA, massesB=massesB)

    # permanent dipole from charges about origin
    mu_perm_q = ((q[:, None]) * (X - origin)).sum(axis=0)

    # build list of polarizable sites
    pol_idx = np.where(alpha > 0.0)[0]
    Np = len(pol_idx)

    # external field from charges (E0) at polarizable sites
    E0 = np.zeros((Np, 3))
    for a, i in enumerate(pol_idx):
        Ei = np.zeros(3)
        for j in range(len(X)):
            if j == i:
                continue
            if (not include_intra) and (mon[j] == mon[i]):
                continue
            Rij = X[i] - X[j]
            # charge->dipole damping scale uses ell_i = alpha_i^(1/3)
            ell = alpha[i] ** (1.0/3.0)
            b = c_scale / ell
            Ei += q[j] * E_charge_tt(Rij, b)
        E0[a] = Ei

    # one-shot induction (no mutual coupling)
    if not mutual or Np == 0:
        mu_ind = np.zeros((len(X), 3))
        if Np > 0:
            mu_pol = alpha[pol_idx, None] * E0
            mu_ind[pol_idx] = mu_pol
        mu_lab = mu_perm_q + mu_ind.sum(axis=0)
        mu_body = B.T @ mu_lab
        return dict(mu_lab=mu_lab, mu_body=mu_body, mu_perm_q=mu_perm_q,
                    mu_ind=mu_ind, origin=origin, B=B)

    # mutual induction: solve (I - A T) mu = A E0
    M = np.eye(3*Np)
    rhs = np.zeros(3*Np)

    for a, i in enumerate(pol_idx):
        ai = alpha[i]
        rhs[3*a:3*a+3] = ai * E0[a]  # A*E0
        for b, k in enumerate(pol_idx):
            if a == b:
                continue
            if (not include_intra) and (mon[k] == mon[i]):
                continue
            Rik = X[i] - X[k]
            # dipole->dipole damping scale ell_ik = (alpha_i alpha_k)^(1/6)
            ell = (alpha[i] * alpha[k]) ** (1.0/6.0)
            bpar = c_scale / ell
            T = T_dipole_tt(Rik, bpar)
            # block: M[a,b] -= alpha_i * T
            M[3*a:3*a+3, 3*b:3*b+3] -= ai * T

    mu_pol_flat = np.linalg.solve(M, rhs)
    mu_pol = mu_pol_flat.reshape(Np, 3)

    mu_ind = np.zeros((len(X), 3))
    mu_ind[pol_idx] = mu_pol

    mu_lab = mu_perm_q + mu_ind.sum(axis=0)
    mu_body = B.T @ mu_lab

    return dict(mu_lab=mu_lab, mu_body=mu_body, mu_perm_q=mu_perm_q,
                mu_ind=mu_ind, origin=origin, B=B)


# ------------------ convenience: components in dimer frame ------------------

def dipole_components_in_dimer_frame(mu_lab, B):
    """Return (mu_x, mu_y, mu_z) in the dimer frame."""
    return (B.T @ np.asarray(mu_lab, float))
