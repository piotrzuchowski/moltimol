# moltimol (minimal)

Small repo containing a single helper module for handling molecular interactions.
It focuses on generating dimer geometries, rotating molecules, and adding noise to
coordinates, which is useful for sampling potential energy surfaces (PESs).

## Requirements

- Python 3.8+
- numpy
- psi4 (used to build Psi4 geometry objects)

Optional:
- ase (for full periodic table support for atomic masses)
- pytest (to run tests)

## Units

All geometry inputs/outputs in this repo are assumed to be in angstrom.

## Files

- `moltimol_helper.py` — helper functions
- `moltimol/` — package implementation
- `sample_molecules.py` — example script generating XYZ geometries
- `CO.xyz` — example monomer input
- `moltimol/MLdms_general.py` — main CLI for sampling/propSAPT batches
- `moltimol/MLdms.py` — backward-compatible wrapper around `moltimol/MLdms_general.py`

## Tests

```bash
pytest
```

Dipole frame consistency test:

```bash
pytest -k dipole_rotation_consistency
```

This test builds a random CO–CO dimer, rotates it into the principal-axis
dimer frame, computes Psi4 dipoles in both frames, and checks that the body-frame
dipole matches the lab-frame dipole rotated by the frame matrix `B`.

## Data Curation

Example FPS selection from a geometry pool:

```python
from moltimol.data_curation import fps_reduce_psi4geoms
fps_reduce_psi4geoms("psi4_geoms", "psi4_geoms_fps", k=1000, start_idx=0)
```

Duplicate and collision scans:

```python
from moltimol.data_curation import find_duplicates_in_psi4_geoms, find_collisions_in_psi4_geoms
find_duplicates_in_psi4_geoms("psi4_geoms", q=0.01, hist_png="nn_hist.png", hist2d_png="nn_vs_R.png")
find_collisions_in_psi4_geoms("psi4_geoms", dmin=1.5)
```
