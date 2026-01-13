# Agent Guide (moltimol)

Quick guide for developers working in this repo.

## Environment

- Python env: `psi4_env`
- Editable install:
  - `conda run -n psi4_env python -m pip install -e /path/to/mol_sampler`

## Core Commands

Run tests:

```bash
pytest
```

Run a single test:

```bash
pytest -k dipole_rotation_consistency
```

## Sampling + propSAPT / SAPT0

Generate psi4geom files:

```bash
python -m moltimol.MLdms_general generate \
  --fileA CO.xyz --fileB CO.xyz \
  --n-samples 500 --out-dir psi4_geoms
```

Batch propSAPT (dipoles), with optional SAPT0:

```bash
python -m moltimol.MLdms_general batch \
  --geom-dir psi4_geoms --batch-index 0 --batch-size 500 --sapt0
```

SAPT0-only batch:

```bash
python -m moltimol.MLdms_general sapt0 \
  --geom-dir psi4_geoms --batch-index 0 --batch-size 500
```

Sample + compute directly:

```bash
python -m moltimol.MLdms_general sample --n-samples 10 --sapt0
```

## Data Curation

Prune collisions + near-duplicates:

```python
from moltimol.data_curation import reduce_psi4geom_pool
reduce_psi4geom_pool("psi4_geoms", "psi4_geoms_pruned", dmin=1.5, q=0.01)
```

FPS selection:

```python
from moltimol.data_curation import fps_reduce_psi4geoms
fps_reduce_psi4geoms("psi4_geoms", "psi4_geoms_fps", k=1000, start_mode="most_isolated")
```

Angular diversity (cos(theta)) histograms:

```python
from moltimol.data_curation import com_axis_angle_histograms
com_axis_angle_histograms("psi4_geoms", out_dir="angle_histograms")
```

Bond length histograms:

```python
from moltimol.data_curation import bond_length_histograms
bond_length_histograms("psi4_geoms", out_dir="bond_histograms")
```

## N2–N2 Scan

Scan dipole vs R and write psi4geom files:

```bash
python moltimol/n2_n2_scan.py
```
