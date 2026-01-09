from pathlib import Path
import shutil

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


def intra_distance_stats(symA, symB, coordsA, coordsB):
    """
    Compute mean/median of intra-monomer distances for A and B.
    """
    d = nearest_neighbor_distances(symA, symB, coordsA, coordsB)
    A_only = d["A_only"]
    B_only = d["B_only"]

    def stats(arr):
        if arr.size == 0:
            return {"mean": np.nan, "median": np.nan, "sigma": np.nan}
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "sigma": float(np.std(arr)),
        }

    return {
        "A_only": stats(A_only),
        "B_only": stats(B_only),
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


def find_duplicate_pairs_nn(features, files, q=0.01, hist_png=None):
    """
    Find near-duplicate geometries using NN distances and a quantile threshold.
    """
    d_nn, idx_nn = nn_distances_features(features)
    tau = float(np.quantile(d_nn, q))
    print(f"NN distances count: {len(d_nn)}")
    print(f"NN distances mean: {float(np.mean(d_nn)):.6f}")
    print(f"NN distances max: {float(np.max(d_nn)):.6f}")
    if hist_png:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.hist(d_nn, bins=50, color="#345995", edgecolor="white")
        plt.xlabel("NN distance")
        plt.ylabel("Count")
        plt.title("Nearest-neighbor distances")
        plt.tight_layout()
        plt.savefig(hist_png, dpi=150)
        plt.close()
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


def find_collisions_in_psi4_geoms(geom_dir, dmin=1.5, report=True):
    """
    Find geometries where any inter-monomer (A-B) distance is below dmin (angstrom).
    """
    geom_dir = Path(geom_dir)
    files = sorted(geom_dir.glob("*.psi4geom"))
    if not files:
        raise ValueError("No .psi4geom files found.")

    collisions = []
    for path in files:
        symA, symB, XA, XB, _ = parse_psi4geom_string(path.read_text())
        if len(XA) == 0 or len(XB) == 0:
            continue
        diff = XA[:, None, :] - XB[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        dmin_ab = float(np.min(d))
        if dmin_ab < dmin:
            collisions.append(
                {
                    "file": str(path),
                    "dmin_ab": dmin_ab,
                }
            )
    if report:
        print(f"Checked {len(files)} geometries in {geom_dir}")
        print(f"Collision threshold dmin = {dmin:.3f} angstrom")
        print(f"Collisions found: {len(collisions)}")
        for item in collisions[:10]:
            print(f"- {item['file']} (dmin_ab={item['dmin_ab']:.3f})")
        if len(collisions) > 10:
            print(f"... {len(collisions) - 10} more")
    return collisions


def reduce_psi4geom_dataset(
    geom_dir,
    out_dir,
    dmin=1.5,
    q=0.01,
    round_decimals=None,
):
    """
    Reduce a dataset by removing collisions and near-duplicate geometries.
    Writes remaining .psi4geom files to out_dir.
    """
    geom_dir = Path(geom_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collisions + intra stats
    collisions = []
    collision_files = set()
    for path in sorted(geom_dir.glob("*.psi4geom")):
        symA, symB, XA, XB, _ = parse_psi4geom_string(path.read_text())
        stats = intra_distance_stats(symA, symB, XA, XB)
        diff = XA[:, None, :] - XB[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        dmin_ab = float(np.min(d)) if d.size else np.nan
        if dmin_ab < dmin:
            collisions.append(
                {
                    "file": str(path),
                    "dmin_ab": dmin_ab,
                    "A_only_mean": stats["A_only"]["mean"],
                    "A_only_median": stats["A_only"]["median"],
                    "A_only_sigma": stats["A_only"]["sigma"],
                    "B_only_mean": stats["B_only"]["mean"],
                    "B_only_median": stats["B_only"]["median"],
                    "B_only_sigma": stats["B_only"]["sigma"],
                }
            )
            collision_files.add(path.name)

    # Near-duplicates
    dup = find_duplicates_in_psi4_geoms(geom_dir, q=q, round_decimals=round_decimals)
    dup_pairs = dup["pairs"]
    dup_remove = set()
    for a, b, _ in dup_pairs:
        dup_remove.add(Path(b).name)

    # Copy retained files
    kept = 0
    for path in sorted(geom_dir.glob("*.psi4geom")):
        name = path.name
        if name in collision_files:
            continue
        if name in dup_remove:
            continue
        shutil.copy2(path, out_dir / name)
        kept += 1

    print(f"Source geometries: {len(list(geom_dir.glob('*.psi4geom')))}")
    print(f"Collisions removed: {len(collision_files)} (dmin={dmin:.3f})")
    if collisions:
        print("Collision intra-distance stats (first 5):")
        for item in collisions[:5]:
            print(
                f"- {item['file']}: "
                f"A_mean={item['A_only_mean']:.3f} "
                f"A_med={item['A_only_median']:.3f} "
                f"A_sig={item['A_only_sigma']:.3f} "
                f"B_mean={item['B_only_mean']:.3f} "
                f"B_med={item['B_only_median']:.3f} "
                f"B_sig={item['B_only_sigma']:.3f}"
            )
    print(f"Duplicates removed: {len(dup_remove)} (q={q:.3f})")
    print(f"Kept geometries: {kept}")
