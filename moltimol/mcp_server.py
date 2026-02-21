"""
MCP server exposing key moltimol workflows as tools.

Run with:
  python -m moltimol.mcp_server
or:
  moltimol-mcp
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

FastMCP = None
_MCP_IMPORT_ERROR = None
try:
    from mcp.server.fastmcp import FastMCP as _FastMCP
except Exception as exc:
    _MCP_IMPORT_ERROR = exc
else:
    FastMCP = _FastMCP


def _require_fastmcp():
    if FastMCP is None:
        msg = (
            "Missing optional dependency 'mcp'. "
            "Install with: python -m pip install -e .[mcp]"
        )
        if _MCP_IMPORT_ERROR is not None:
            msg = f"{msg} (import error: {_MCP_IMPORT_ERROR})"
        raise RuntimeError(msg)
    return FastMCP


def _count_csv_rows(path: str) -> Optional[int]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r") as f:
        row_count = sum(1 for _ in f)
    return max(row_count - 1, 0)


def _as_preview(items: List[Any], limit: int = 20) -> List[Any]:
    return items[: max(limit, 0)]


def _convert_psi4geom_text_to_xyz(geom_text: str, out_path: Path) -> None:
    import numpy as np

    from moltimol import parse_psi4geom_string

    symA, symB, XA, XB, _ = parse_psi4geom_string(geom_text)
    symbols = list(symA) + list(symB)
    coords = np.vstack([XA, XB]) if symbols else np.zeros((0, 3), float)
    nA = len(symA)
    nB = len(symB)

    with open(out_path, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write(f"{len(symbols)} {nA} {nB}\n")
        for sym, (x, y, z) in zip(symbols, coords):
            f.write(f"{sym} {x:.10f} {y:.10f} {z:.10f}\n")


def _convert_psi4geom_file(path: Path, out_dir: Path, suffix: str) -> Path:
    out_name = f"{path.stem}{suffix}.xyz"
    out_path = out_dir / out_name
    _convert_psi4geom_text_to_xyz(path.read_text(), out_path)
    return out_path


def create_server():
    mcp_cls = _require_fastmcp()
    mcp = mcp_cls("moltimol")

    @mcp.tool()
    def generate_psi4geom_files(
        n_samples: int = 50000,
        fileA: str = "CO.xyz",
        fileB: str = "CO.xyz",
        r_min: float = 3.0,
        r_max: float = 8.0,
        mode_frac: float = 1 / 3,
        concentration: float = 10.0,
        sigma_noise: float = 0.1,
        seed: Optional[int] = None,
        out_dir: str = "psi4_geoms",
        psi4_units: str = "angstrom",
        charge_mult_A: str = "0 1",
        charge_mult_B: str = "0 1",
        body_frame: bool = False,
        hist_png: str = "R_hist.png",
    ) -> Dict[str, Any]:
        from moltimol.MLdms_general import generate_psi4geom_files as _generate

        out_path = Path(out_dir)
        before = {p.name for p in out_path.glob("*.psi4geom")} if out_path.exists() else set()

        _generate(
            n_samples=n_samples,
            fileA=fileA,
            fileB=fileB,
            r_min=r_min,
            r_max=r_max,
            mode_frac=mode_frac,
            concentration=concentration,
            sigma_noise=sigma_noise,
            seed=seed,
            out_dir=out_dir,
            psi4_units=psi4_units,
            charge_mult_A=charge_mult_A,
            charge_mult_B=charge_mult_B,
            body_frame=body_frame,
            hist_png=hist_png,
        )

        after_files = sorted(out_path.glob("*.psi4geom"))
        new_files = [p.name for p in after_files if p.name not in before]
        hist_path = str(Path(hist_png).resolve()) if hist_png else None

        return {
            "output_directory": str(out_path.resolve()),
            "new_file_count": len(new_files),
            "total_file_count": len(after_files),
            "new_files_preview": _as_preview(new_files),
            "histogram_path": hist_path,
        }

    @mcp.tool()
    def run_propsapt_batch(
        geom_dir: str = "psi4_geoms",
        batch_index: int = 0,
        batch_size: int = 500,
        out_csv: Optional[str] = None,
        out_npz: Optional[str] = None,
        method_low: str = "propSAPT",
        method_high: Optional[str] = None,
        sapt0: bool = False,
    ) -> Dict[str, Any]:
        from moltimol.MLdms_general import run_propsapt_batch as _run

        _run(
            geom_dir=geom_dir,
            batch_index=batch_index,
            batch_size=batch_size,
            out_csv=out_csv,
            out_npz=out_npz,
            method_low=method_low,
            method_high=method_high,
            compute_sapt0=sapt0,
        )

        csv_path = out_csv or f"propsapt_batch_{batch_index:03d}.csv"
        npz_path = out_npz or f"propsapt_batch_{batch_index:03d}.npz"

        return {
            "csv_path": str(Path(csv_path).resolve()),
            "npz_path": str(Path(npz_path).resolve()),
            "row_count": _count_csv_rows(csv_path),
        }

    @mcp.tool()
    def run_sapt0_batch(
        geom_dir: str = "psi4_geoms",
        batch_index: int = 0,
        batch_size: int = 500,
        out_csv: Optional[str] = None,
        out_npz: Optional[str] = None,
        basis: str = "aug-cc-pvdz",
    ) -> Dict[str, Any]:
        from moltimol.MLdms_general import run_sapt0_batch as _run

        _run(
            geom_dir=geom_dir,
            batch_index=batch_index,
            batch_size=batch_size,
            out_csv=out_csv,
            out_npz=out_npz,
            basis=basis,
        )

        csv_path = out_csv or f"sapt0_batch_{batch_index:03d}.csv"
        npz_path = out_npz or f"sapt0_batch_{batch_index:03d}.npz"

        return {
            "csv_path": str(Path(csv_path).resolve()),
            "npz_path": str(Path(npz_path).resolve()),
            "row_count": _count_csv_rows(csv_path),
        }

    @mcp.tool()
    def sample_dimer_geometries(
        n_samples: int = 1000,
        fileA: str = "CO.xyz",
        fileB: str = "CO.xyz",
        r_min: float = 3.0,
        r_max: float = 8.0,
        mode_frac: float = 1 / 3,
        concentration: float = 10.0,
        sigma_noise: float = 0.1,
        seed: Optional[int] = None,
        out_csv: Optional[str] = "ml_dimer_data.csv",
        out_npz: Optional[str] = None,
        method_low: str = "propSAPT",
        method_high: Optional[str] = None,
        sapt0: bool = False,
    ) -> Dict[str, Any]:
        from moltimol.MLdms_general import sample_dimer_geometries as _sample

        data, df = _sample(
            n_samples=n_samples,
            fileA=fileA,
            fileB=fileB,
            r_min=r_min,
            r_max=r_max,
            mode_frac=mode_frac,
            concentration=concentration,
            sigma_noise=sigma_noise,
            seed=seed,
            out_csv=out_csv,
            out_npz=out_npz,
            method_low=method_low,
            method_high=method_high,
            compute_sapt0=sapt0,
        )

        first_sample = None
        if data:
            first_sample = {
                "index": int(data[0]["index"]),
                "R": float(data[0]["R"]),
                "n_atoms_A": int(data[0]["n_atoms_A"]),
                "n_atoms_B": int(data[0]["n_atoms_B"]),
            }

        return {
            "sample_count": len(data),
            "table_rows": int(len(df)),
            "csv_path": str(Path(out_csv).resolve()) if out_csv else None,
            "npz_path": str(Path(out_npz).resolve()) if out_npz else None,
            "first_sample": first_sample,
        }

    @mcp.tool()
    def find_duplicates_in_psi4_geoms(
        geom_dir: str,
        q: float = 0.01,
        round_decimals: Optional[int] = None,
        hist_png: Optional[str] = None,
        hist2d_png: Optional[str] = None,
        preview_limit: int = 20,
    ) -> Dict[str, Any]:
        import numpy as np

        from moltimol.data_curation import find_duplicates_in_psi4_geoms as _find

        result = _find(
            geom_dir=geom_dir,
            q=q,
            round_decimals=round_decimals,
            hist_png=hist_png,
            hist2d_png=hist2d_png,
        )
        nn_dist = np.asarray(result["nn_dist"], dtype=float)
        pairs = [
            {"file_a": str(a), "file_b": str(b), "distance": float(d)}
            for a, b, d in result["pairs"]
        ]
        stats = {
            "count": int(nn_dist.size),
            "mean": float(nn_dist.mean()) if nn_dist.size else None,
            "min": float(nn_dist.min()) if nn_dist.size else None,
            "max": float(nn_dist.max()) if nn_dist.size else None,
        }
        return {
            "tau": float(result["tau"]),
            "pair_count": len(pairs),
            "pairs_preview": _as_preview(pairs, preview_limit),
            "nn_distance_stats": stats,
            "hist_png": str(Path(hist_png).resolve()) if hist_png else None,
            "hist2d_png": str(Path(hist2d_png).resolve()) if hist2d_png else None,
        }

    @mcp.tool()
    def find_collisions_in_psi4_geoms(
        geom_dir: str,
        dmin: float = 1.5,
        report: bool = False,
        preview_limit: int = 20,
    ) -> Dict[str, Any]:
        from moltimol.data_curation import find_collisions_in_psi4_geoms as _find

        collisions = _find(geom_dir=geom_dir, dmin=dmin, report=report)
        preview = [
            {"file": str(item["file"]), "dmin_ab": float(item["dmin_ab"])}
            for item in collisions[: max(preview_limit, 0)]
        ]
        return {
            "collision_count": len(collisions),
            "dmin": float(dmin),
            "collisions_preview": preview,
        }

    @mcp.tool()
    def reduce_psi4geom_pool(
        geom_dir: str,
        out_dir: str,
        dmin: float = 1.5,
        q: float = 0.01,
        round_decimals: Optional[int] = None,
        kept_list_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        from moltimol.data_curation import reduce_psi4geom_pool as _reduce

        _reduce(
            geom_dir=geom_dir,
            out_dir=out_dir,
            dmin=dmin,
            q=q,
            round_decimals=round_decimals,
            kept_list_path=kept_list_path,
        )
        out_path = Path(out_dir)
        kept_files = sorted(out_path.glob("*.psi4geom"))
        return {
            "output_directory": str(out_path.resolve()),
            "kept_count": len(kept_files),
            "kept_files_preview": _as_preview([p.name for p in kept_files]),
            "kept_list_path": str(Path(kept_list_path).resolve()) if kept_list_path else None,
        }

    @mcp.tool()
    def fps_reduce_psi4geoms(
        geom_dir: str,
        out_dir: str,
        k: int,
        start_idx: int = 0,
        round_decimals: Optional[int] = None,
        list_txt: Optional[str] = None,
    ) -> Dict[str, Any]:
        from moltimol.data_curation import fps_reduce_psi4geoms as _fps

        selected = _fps(
            geom_dir=geom_dir,
            out_dir=out_dir,
            k=k,
            start_idx=start_idx,
            round_decimals=round_decimals,
            list_txt=list_txt,
        )
        return {
            "output_directory": str(Path(out_dir).resolve()),
            "selected_count": len(selected),
            "selected_preview": _as_preview(selected),
            "list_txt": str(Path(list_txt).resolve()) if list_txt else None,
        }

    @mcp.tool()
    def com_axis_angle_histograms(
        geom_dir: str,
        out_dir: str = "angle_histograms",
        bins: int = 60,
        max_files: Optional[int] = None,
    ) -> Dict[str, Any]:
        from moltimol.data_curation import com_axis_angle_histograms as _hist

        _hist(geom_dir=geom_dir, out_dir=out_dir, bins=bins, max_files=max_files)
        out_path = Path(out_dir)
        expected = ["A_only_angles.png", "B_only_angles.png", "AB_combined_angles.png"]
        written = [name for name in expected if (out_path / name).exists()]
        return {
            "output_directory": str(out_path.resolve()),
            "written_files": written,
        }

    @mcp.tool()
    def bond_length_histograms(
        geom_dir: str,
        out_dir: str = "bond_histograms",
        bins: int = 50,
        max_files: Optional[int] = None,
    ) -> Dict[str, Any]:
        from moltimol.data_curation import bond_length_histograms as _hist

        _hist(geom_dir=geom_dir, out_dir=out_dir, bins=bins, max_files=max_files)
        out_path = Path(out_dir)
        expected = ["A_only_bonds.png", "B_only_bonds.png", "AB_combined_bonds.png"]
        written = [name for name in expected if (out_path / name).exists()]
        return {
            "output_directory": str(out_path.resolve()),
            "written_files": written,
        }

    @mcp.tool()
    def psi4geom_to_xyz(
        input_path: str,
        out_dir: str = "psi4geom_xyz",
        suffix: str = "",
    ) -> Dict[str, Any]:
        in_path = Path(input_path)
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if in_path.is_dir():
            files = sorted(in_path.glob("*.psi4geom"))
            if not files:
                raise ValueError(f"No .psi4geom files found in {in_path}")
            written = [str(_convert_psi4geom_file(path, out_path, suffix)) for path in files]
            return {
                "mode": "directory",
                "output_directory": str(out_path.resolve()),
                "converted_count": len(written),
                "converted_preview": _as_preview(written),
            }

        if in_path.is_file() and in_path.suffix == ".psi4geom":
            written_path = _convert_psi4geom_file(in_path, out_path, suffix)
            return {
                "mode": "file",
                "output_directory": str(out_path.resolve()),
                "converted_count": 1,
                "converted_preview": [str(written_path)],
            }

        raise ValueError(f"Input must be a .psi4geom file or directory: {input_path}")

    return mcp


def main() -> None:
    try:
        server = create_server()
    except RuntimeError as exc:
        raise SystemExit(str(exc))
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
