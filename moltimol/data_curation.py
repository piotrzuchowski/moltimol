from pathlib import Path
import shutil

import numpy as np
from scipy.spatial import cKDTree

from moltimol import center_of_mass_mass, mass_of, parse_psi4geom_string

from prop_sapt import Dimer


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


def bond_length_histograms(
    geom_dir,
    out_dir="bond_histograms",
    bins=50,
    max_files=None,
):
    """
    Create histograms of intra-monomer bond lengths for a geometry pool.

    Writes PNGs:
      - A_only_bonds.png
      - B_only_bonds.png
      - AB_combined_bonds.png
    """
    geom_dir = Path(geom_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(geom_dir.glob("*.psi4geom"))
    if max_files is not None:
        files = files[: int(max_files)]

    A_all = []
    B_all = []
    for path in files:
        symA, symB, XA, XB, _ = parse_psi4geom_string(path.read_text())
        d = nearest_neighbor_distances(symA, symB, XA, XB)
        if d["A_only"].size:
            A_all.append(d["A_only"])
        if d["B_only"].size:
            B_all.append(d["B_only"])

    A_all = np.concatenate(A_all) if A_all else np.array([], float)
    B_all = np.concatenate(B_all) if B_all else np.array([], float)

    import matplotlib.pyplot as plt

    def _plot(data, title, filename):
        plt.figure(figsize=(6, 4))
        plt.hist(data, bins=bins, color="#2F6B4F", edgecolor="white")
        plt.xlabel("Bond length (angstrom)")
        plt.ylabel("Count")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=150)
        plt.close()

    if A_all.size:
        _plot(A_all, "Monomer A bond lengths", "A_only_bonds.png")
    if B_all.size:
        _plot(B_all, "Monomer B bond lengths", "B_only_bonds.png")
    if A_all.size or B_all.size:
        combined = np.concatenate([A_all, B_all]) if B_all.size else A_all
        _plot(combined, "All intra-monomer bond lengths", "AB_combined_bonds.png")

    print(f"Processed {len(files)} geometries from {geom_dir}")
    print(f"Wrote histograms to {out_dir}")


def com_axis_angle_histograms(
    geom_dir,
    out_dir="angle_histograms",
    bins=60,
    max_files=None,
):
    """
    Create histograms of cos(theta) between the COMA->COMB axis and monomer vectors.

    For each geometry:
      - COMA is the center of mass of monomer A.
      - COMB is the center of mass of monomer B.
      - axis is COMA -> COMB.
      - For each atom in A, we compute the angle between axis and COMA->atomA.
      - For each atom in B, we compute the angle between (-axis) and COMB->atomB,
        so the reference axis always points "from the other monomer".

    This yields an orientation diversity diagnostic (cos(theta) in [-1, 1])
    across all geometries in geom_dir.

    Outputs:
      - A_only_angles.png: angles for monomer A atoms.
      - B_only_angles.png: angles for monomer B atoms.
      - AB_combined_angles.png: all A+B angles combined.

    Parameters
    ----------
    geom_dir : str or Path
        Directory with .psi4geom files.
    out_dir : str or Path, default "angle_histograms"
        Output directory for PNGs.
    bins : int, default 60
        Number of histogram bins.
    max_files : int or None, default None
        If set, only the first max_files geometries are processed.
    """
    geom_dir = Path(geom_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(geom_dir.glob("*.psi4geom"))
    if max_files is not None:
        files = files[: int(max_files)]

    A_cos = []
    B_cos = []
    for path in files:
        symA, symB, XA, XB, _ = parse_psi4geom_string(path.read_text())
        if XA.size == 0 or XB.size == 0:
            continue
        massesA = np.array([mass_of(s) for s in symA], dtype=float)
        massesB = np.array([mass_of(s) for s in symB], dtype=float)
        comA = center_of_mass_mass(XA, massesA)
        comB = center_of_mass_mass(XB, massesB)
        axis = comB - comA
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            continue
        axis_u = axis / axis_norm

        for ra in XA:
            va = ra - comA
            na = np.linalg.norm(va)
            if na < 1e-12:
                continue
            cosang = np.clip(np.dot(axis_u, va / na), -1.0, 1.0)
            A_cos.append(cosang)

        axis_b = -axis_u
        for rb in XB:
            vb = rb - comB
            nb = np.linalg.norm(vb)
            if nb < 1e-12:
                continue
            cosang = np.clip(np.dot(axis_b, vb / nb), -1.0, 1.0)
            B_cos.append(cosang)

    A_cos = np.array(A_cos, float)
    B_cos = np.array(B_cos, float)

    import matplotlib.pyplot as plt

    def _plot(data, title, filename):
        plt.figure(figsize=(6, 4))
        plt.hist(data, bins=bins, color="#2F6B4F", edgecolor="white")
        plt.xlabel("cos(theta)")
        plt.ylabel("Count")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=150)
        plt.close()

    if A_cos.size:
        _plot(A_cos, "Monomer A: COMA->COMB vs COMA->atomA", "A_only_angles.png")
    if B_cos.size:
        _plot(B_cos, "Monomer B: COMB->COMA vs COMB->atomB", "B_only_angles.png")
    if A_cos.size or B_cos.size:
        combined = np.concatenate([A_cos, B_cos]) if B_cos.size else A_cos
        _plot(combined, "All COM-axis angles", "AB_combined_angles.png")

    print(f"Processed {len(files)} geometries from {geom_dir}")
    print(f"Wrote histograms to {out_dir}")


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


def find_duplicates_in_psi4_geoms(geom_dir, q=0.01, round_decimals=None, hist_png=None, hist2d_png=None):
    """
    Convenience wrapper to find duplicates in a folder of .psi4geom files.
    """
    features, files = load_psi4geom_features(geom_dir, round_decimals=round_decimals)
    result = find_duplicate_pairs_nn(features, files, q=q, hist_png=hist_png)
    if hist2d_png:
        R_vals = []
        for fpath in files:
            symA, symB, XA, XB, _ = parse_psi4geom_string(Path(fpath).read_text())
            massesA = np.array([mass_of(s) for s in symA])
            massesB = np.array([mass_of(s) for s in symB])
            RA = center_of_mass_mass(XA, massesA)
            RB = center_of_mass_mass(XB, massesB)
            R_vals.append(float(np.linalg.norm(RB - RA)))
        R_vals = np.asarray(R_vals, float)
        d_nn = result["nn_dist"]
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.hist2d(R_vals, d_nn, bins=50, cmap="viridis")
        plt.xlabel("R (Angstrom)")
        plt.ylabel("NN distance")
        plt.title("NN distance vs R")
        plt.colorbar(label="Count")
        plt.tight_layout()
        plt.savefig(hist2d_png, dpi=150)
        plt.close()
    return result


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
    list_txt=None,
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
    kept_list = []
    for path in sorted(geom_dir.glob("*.psi4geom")):
        name = path.name
        if name in collision_files:
            continue
        if name in dup_remove:
            continue
        shutil.copy2(path, out_dir / name)
        kept += 1
        geom_id = ""
        try:
            geom_id = str(int(path.stem.split("_")[-1]))
        except Exception:
            geom_id = ""
        kept_list.append((name, geom_id))

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
    if list_txt:
        with open(list_txt, "w") as f:
            f.write("filename\tgeom_id\n")
            for name, geom_id in kept_list:
                f.write(f"{name}\t{geom_id}\n")


def reduce_psi4geom_pool(*args, **kwargs):
    """
    Backward-compatible alias for reduce_psi4geom_dataset.
    """
    if "kept_list_path" in kwargs and "list_txt" not in kwargs:
        kwargs["list_txt"] = kwargs.pop("kept_list_path")
    return reduce_psi4geom_dataset(*args, **kwargs)


def fps_select_indices(features, k, start_idx=0, start_mode="index"):
    """
    Farthest point sampling (FPS) on a feature matrix.

    Parameters
    ----------
    features : array-like, shape (n_samples, n_features)
        Feature vectors (e.g., cross-distance features).
    k : int
        Number of samples to select.
    start_idx : int, default 0
        Index of the initial seed point (used when start_mode="index").
    start_mode : {"index","most_isolated"}, default "index"
        If "most_isolated", starts from the point with the largest NN distance.

    Returns
    -------
    list[int]
        Indices of selected samples in selection order.
    """
    F = np.asarray(features, float)
    n = F.shape[0]
    if k <= 0 or k > n:
        raise ValueError("k must be in [1, n]")
    if start_mode == "most_isolated":
        d_nn, _ = nn_distances_features(F)
        start_idx = int(np.argmax(d_nn))
    elif start_mode != "index":
        raise ValueError("start_mode must be 'index' or 'most_isolated'")

    selected = [start_idx]
    dist = np.full(n, np.inf)
    for _ in range(1, k):
        last = F[selected[-1]]
        d = np.linalg.norm(F - last, axis=1)
        dist = np.minimum(dist, d)
        next_idx = int(np.argmax(dist))
        selected.append(next_idx)
    return selected


def fps_reduce_psi4geoms(
    geom_dir,
    out_dir,
    k,
    start_idx=0,
    round_decimals=None,
    list_txt=None,
):
    """
    Reduce dataset with FPS using cross-distance features.

    Parameters
    ----------
    geom_dir : str or Path
        Directory with .psi4geom files.
    out_dir : str or Path
        Output directory for selected files.
    k : int
        Number of geometries to keep.
    start_idx : int, default 0
        Initial seed index in the sorted file list.
    round_decimals : int or None
        If set, rounds distances before feature construction.

    Returns
    -------
    list[str]
        Paths of selected files written to out_dir.
    """
    features, files = load_psi4geom_features(geom_dir, round_decimals=round_decimals)
    sel_idx = fps_select_indices(features, k, start_idx=start_idx)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_files = []
    kept_list = []
    for i in sel_idx:
        src = Path(files[i])
        dst = out_dir / src.name
        shutil.copy2(src, dst)
        selected_files.append(str(dst))
        geom_id = ""
        try:
            geom_id = str(int(src.stem.split("_")[-1]))
        except Exception:
            geom_id = ""
        kept_list.append((src.name, geom_id))

    print(f"FPS selected {len(selected_files)} of {len(files)} geometries.")
    if list_txt:
        with open(list_txt, "w") as f:
            f.write("filename\tgeom_id\n")
            for name, geom_id in kept_list:
                f.write(f"{name}\t{geom_id}\n")
    return selected_files
