from pathlib import Path

import numpy as np
import psi4
from scipy.spatial import cKDTree

from prop_sapt import Dimer

BOHR_TO_ANGSTROM = 0.52917721092


def parse_psi4geom_string(text):
    """
    Parse a Psi4 geometry block with monomer separator ("--").
    Returns (symA, symB, coords_A, coords_B, units).
    """
    units = "angstrom"
    symA, symB = [], []
    coordsA, coordsB = [], []
    section = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("units"):
            units = lower.split()[1]
            continue
        if lower.startswith("symmetry") or lower.startswith("no_com") or lower.startswith("no_reorient"):
            continue
        if line.startswith("--"):
            section = 1
            continue
        parts = line.split()
        if len(parts) == 2 and all(p.replace("+", "").replace("-", "").isdigit() for p in parts):
            continue
        if len(parts) >= 4:
            sym = parts[0]
            x, y, z = map(float, parts[1:4])
            if section == 0:
                symA.append(sym)
                coordsA.append([x, y, z])
            else:
                symB.append(sym)
                coordsB.append([x, y, z])

    coordsA = np.array(coordsA, float)
    coordsB = np.array(coordsB, float)
    if units == "bohr":
        coordsA *= BOHR_TO_ANGSTROM
        coordsB *= BOHR_TO_ANGSTROM
    return symA, symB, coordsA, coordsB, units


def compute_sapt0_batch(geom_strings, basis="aug-cc-pvdz", psi4_options=None):
    """
    Compute SAPT0 energies for a batch of geometry strings.
    Returns a list of dicts with energy components per geometry.
    """
    psi4.set_options({"basis": basis})
    if psi4_options:
        psi4.set_options(psi4_options)

    results = []
    for i, geom in enumerate(geom_strings):
        dimer = Dimer(geom)
        sapt0 = dimer.sapt0()
        row = {"geom_id": i}
        if hasattr(sapt0, "to_dict"):
            row.update(sapt0.to_dict())
        elif isinstance(sapt0, dict):
            row.update(sapt0)
        else:
            row["sapt0_total"] = float(sapt0)
        results.append(row)
    return results


def nearest_neighbor_distances(symA, symB, coordsA, coordsB):
    """
    Compute nearest-neighbor distances in canonical order.
    Returns dict with sorted distances for A-only, B-only, and AB pairs.
    """
    XA = np.asarray(coordsA, float)
    XB = np.asarray(coordsB, float)

    def pairwise_distances(X):
        n = len(X)
        if n < 2:
            return np.array([], float)
        diff = X[:, None, :] - X[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(n, k=1)
        return np.sort(d[iu])

    def cross_distances(X, Y):
        if len(X) == 0 or len(Y) == 0:
            return np.array([], float)
        diff = X[:, None, :] - Y[None, :, :]
        d = np.linalg.norm(diff, axis=2).ravel()
        return np.sort(d)

    return {
        "A_only": pairwise_distances(XA),
        "B_only": pairwise_distances(XB),
        "AB": cross_distances(XA, XB),
    }


def cross_dist_feature(
    XA,
    XB,
    sort=True,
    flatten=True,
    round_decimals=None,
    check_shapes=True,
):
    """
    Build a rotation/translation-invariant feature vector from cross distances.
    """
    XA = np.asarray(XA, dtype=float)
    XB = np.asarray(XB, dtype=float)

    if check_shapes:
        if XA.ndim != 2 or XA.shape[1] != 3:
            raise ValueError(f"XA must have shape (nA,3); got {XA.shape}")
        if XB.ndim != 2 or XB.shape[1] != 3:
            raise ValueError(f"XB must have shape (nB,3); got {XB.shape}")
        if XA.shape[0] == 0 or XB.shape[0] == 0:
            raise ValueError("XA and XB must contain at least one atom each.")

    D = np.linalg.norm(XA[:, None, :] - XB[None, :, :], axis=2)

    if round_decimals is not None:
        D = np.round(D, decimals=int(round_decimals))

    if not flatten:
        return D

    feat = D.ravel()
    if sort:
        feat = np.sort(feat)
    return feat


def nn_distances_features(F):
    """
    Compute nearest-neighbor distances/indices in feature space using a KD-tree.
    """
    F = np.asarray(F, float)
    Fn = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-12)
    tree = cKDTree(Fn)
    d, idx = tree.query(Fn, k=2)
    return d[:, 1], idx[:, 1]


def load_psi4geom_features(geom_dir, round_decimals=None):
    """
    Load .psi4geom files and build cross-distance features for each geometry.
    """
    geom_dir = Path(geom_dir)
    files = sorted(geom_dir.glob("*.psi4geom"))
    if not files:
        raise ValueError("No .psi4geom files found.")

    features = []
    for path in files:
        symA, symB, XA, XB, _ = parse_psi4geom_string(path.read_text())
        feat = cross_dist_feature(
            XA, XB, sort=True, flatten=True, round_decimals=round_decimals
        )
        features.append(feat)
    return np.asarray(features, float), [str(p) for p in files]


def find_duplicate_pairs_nn(features, files, q=0.01):
    """
    Find near-duplicate geometries using NN distances and a quantile threshold.
    """
    d_nn, idx_nn = nn_distances_features(features)
    tau = float(np.quantile(d_nn, q))
    pairs = []
    for i, (d, j) in enumerate(zip(d_nn, idx_nn)):
        if d <= tau:
            pairs.append((files[i], files[j], float(d)))
    return {"tau": tau, "pairs": pairs, "nn_dist": d_nn, "nn_idx": idx_nn}


def find_duplicates_in_psi4_geoms(geom_dir, q=0.01, round_decimals=None):
    """
    Convenience wrapper to find duplicates in a folder of .psi4geom files.
    """
    features, files = load_psi4geom_features(geom_dir, round_decimals=round_decimals)
    return find_duplicate_pairs_nn(features, files, q=q)
