import numpy as np
import pandas as pd


def _write_psi4geom(path, zsep, symbol="H"):
    content = (
        f"{symbol} 0.0 0.0 0.0\n"
        "--\n"
        f"{symbol} 0.0 0.0 {zsep:.6f}\n"
    )
    path.write_text(content)


def test_run_propsapt_batch_with_sapt0(tmp_path, monkeypatch):
    from moltimol import parse_psi4geom_string
    import moltimol.MLdms_general as mldms

    class _FakeWavefunction:
        def __init__(self, coords):
            self._coords = coords

        def geometry(self):
            return self

        def to_array(self):
            return self._coords

    class _FakeDimer:
        def __init__(self, geom_str):
            symA, symB, XA, XB, _ = parse_psi4geom_string(geom_str)
            coords = np.vstack([XA, XB])
            if coords.size == 0:
                raise ValueError("Empty geometry in test.")
            self.dimer = _FakeWavefunction(coords)

        def sapt0(self):
            return {"sapt0_total": -0.1}

    def _fake_calc_property(_dimer, _prop, results=None):
        return pd.DataFrame(
            {"dmu_total": [0.1, 0.2, 0.3]}, index=["X", "Y", "Z"]
        )

    monkeypatch.setattr(mldms, "Dimer", _FakeDimer)
    monkeypatch.setattr(mldms, "calc_property", _fake_calc_property)

    from moltimol.MLdms_general import run_propsapt_batch

    _write_psi4geom(tmp_path / "H_H_000000.psi4geom", 3.0)
    _write_psi4geom(tmp_path / "H_H_000001.psi4geom", 4.0)

    out_csv = tmp_path / "propsapt_batch_000.csv"
    out_npz = tmp_path / "propsapt_batch_000.npz"
    run_propsapt_batch(
        geom_dir=tmp_path,
        batch_index=0,
        batch_size=2,
        out_csv=out_csv,
        out_npz=out_npz,
        compute_sapt0=True,
    )

    df = pd.read_csv(out_csv)
    assert len(df) == 2
    assert "geom_id" in df.columns
    assert any("sapt0" in c for c in df.columns)
    assert "dmu_total_x" in df.columns
