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

## Files

- `moltimol_helper.py` — helper functions
- `moltimol/` — package implementation
- `sample_molecules.py` — example script generating XYZ geometries
- `CO.xyz` — example monomer input

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
