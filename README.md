# moltimol

Utilities for molecular dimer geometry generation, frame transforms, dipole/property
batch calculations, and dataset curation for SAPT-style workflows.

## What is in this repo

- `moltimol/` — package code (geometry, rotations, frame utilities, curation helpers)
- `moltimol/MLdms_general.py` — main CLI (`generate`, `batch`, `sapt0`, `sample`)
- `moltimol/MLdms.py` — backward-compatible wrapper around `MLdms_general`
- `moltimol/data_curation.py` — duplicate/collision checks + FPS reduction
- `psi4geom_to_xyz.py` — convert `.psi4geom` files to `.xyz`
- `sample_molecules.py` — standalone example script

## Requirements

- Python `>=3.8`
- package install from this repo (`pip install -e .`)
- runtime dependencies used by workflows:
  - `numpy`
  - `pandas` (CSV/NPZ tabular output)
  - `scipy` (KD-tree nearest-neighbor features for curation)
  - `matplotlib` (histograms/diagnostics)
  - `psi4` (geometry + SAPT0 fallback calculations)
  - `prop_sapt` (for `Dimer` and dipole property evaluation)

Optional:

- `ase` (broader element mass support)
- `pytest` (tests)

## Quick start

```bash
# example environment name used in this repo
conda activate psi4_env

# editable install
python -m pip install -e .
```

Generate a geometry pool:

```bash
python -m moltimol.MLdms_general generate \
  --fileA CO.xyz --fileB CO.xyz \
  --n-samples 500 --out-dir psi4_geoms
```

Run one dipole batch on those geometries:

```bash
python -m moltimol.MLdms_general batch \
  --geom-dir psi4_geoms --batch-index 0 --batch-size 500
```

Run SAPT0-only batch:

```bash
python -m moltimol.MLdms_general sapt0 \
  --geom-dir psi4_geoms --batch-index 0 --batch-size 500
```

Directly sample + compute without writing `.psi4geom` files:

```bash
python -m moltimol.MLdms_general sample --n-samples 100
```

## MCP server

This repo now includes an MCP server that exposes the current `moltimol`
workflows as MCP tools (generation, batch runs, curation, and conversion).

### Install

```bash
python -m pip install -e .[mcp]
```

### Run server (stdio transport)

```bash
python -m moltimol.mcp_server
```

or:

```bash
moltimol-mcp
```

### Example MCP client config

Use your local Python interpreter from the environment where `moltimol` is installed:

```json
{
  "mcpServers": {
    "moltimol": {
      "command": "/path/to/python",
      "args": ["-m", "moltimol.mcp_server"],
      "cwd": "/path/to/mol_sampler"
    }
  }
}
```

You can also copy `mcp_server.example.json` as a starting template.

### Exposed tools

- `generate_psi4geom_files`
- `run_propsapt_batch`
- `run_sapt0_batch`
- `sample_dimer_geometries`
- `find_duplicates_in_psi4_geoms`
- `find_collisions_in_psi4_geoms`
- `reduce_psi4geom_pool`
- `fps_reduce_psi4geoms`
- `com_axis_angle_histograms`
- `bond_length_histograms`
- `psi4geom_to_xyz`

Dependency notes:

- `generate_psi4geom_files`, `run_propsapt_batch`, `run_sapt0_batch`, and `sample_dimer_geometries` require `psi4` and/or `prop_sapt`.
- `find_*`, `reduce_*`, `fps_*`, histogram tools, and `psi4geom_to_xyz` can run without `psi4`.

## Units and geometry assumptions

- All geometry inputs/outputs are treated as angstrom.
- `.psi4geom` parsing/writing in this repo currently supports only `units angstrom`.
- Dimers are stored as two monomer blocks separated by `--`.

## CLI reference (`python -m moltimol.MLdms_general`)

### `generate`

Creates `.psi4geom` files named `<stemA>_<stemB>_<id>.psi4geom`.

```bash
python -m moltimol.MLdms_general generate [options]
```

| Option | Default | Meaning |
|---|---:|---|
| `--n-samples` | `50000` | Number of dimers to generate. |
| `--fileA` | `CO.xyz` | Monomer A XYZ file path. |
| `--fileB` | `CO.xyz` | Monomer B XYZ file path. |
| `--r-min` | `3.0` | Lower bound of sampled COM distance `R`. |
| `--r-max` | `8.0` | Upper bound of sampled COM distance `R`. |
| `--mode-frac` | `1/3` | Beta mode location within `[r-min, r-max]` interval. Must be in `(0, 1)`. |
| `--concentration` | `10.0` | Beta concentration (spread). Must be `> 2`. |
| `--sigma-noise` | `0.1` | Cartesian Gaussian noise (angstrom) added during merge. |
| `--seed` | `None` | Random seed for reproducibility. |
| `--out-dir` | `psi4_geoms` | Output directory for generated `.psi4geom` files. |
| `--psi4-units` | `angstrom` | Geometry units string. Currently only `angstrom` is accepted. |
| `--charge-mult-A` | `"0 1"` | Charge/multiplicity line for monomer A block. |
| `--charge-mult-B` | `"0 1"` | Charge/multiplicity line for monomer B block. |
| `--body-frame` | off | If set, rotates coordinates to principal-axis body frame before writing. |
| `--hist-png` | `R_hist.png` | Path for histogram of sampled `R`; set empty string to disable saving. |

### `batch`

Runs propSAPT dipole calculations on an existing `.psi4geom` pool.

```bash
python -m moltimol.MLdms_general batch [options]
```

| Option | Default | Meaning |
|---|---:|---|
| `--geom-dir` | `psi4_geoms` | Directory containing `.psi4geom` files. |
| `--batch-index` | `0` | 0-based batch number; start = `batch-index * batch-size`. |
| `--batch-size` | `500` | Number of geometries processed in this run. |
| `--out-csv` | auto | Output CSV path. Auto name: `propsapt_batch_{batch_index:03d}.csv`. |
| `--out-npz` | auto | Output NPZ path. Auto name: `propsapt_batch_{batch_index:03d}.npz`. |
| `--method-low` | `propSAPT` | Metadata value written to output column `method_low`. |
| `--method-high` | `None` | Metadata value written to output column `method_high` (empty if unset). |
| `--sapt0` | off | Also compute and append SAPT0 columns for each geometry. |

### `sapt0`

Runs SAPT0 for a batch of existing `.psi4geom` files.

```bash
python -m moltimol.MLdms_general sapt0 [options]
```

| Option | Default | Meaning |
|---|---:|---|
| `--geom-dir` | `psi4_geoms` | Directory containing `.psi4geom` files. |
| `--batch-index` | `0` | 0-based batch number. |
| `--batch-size` | `500` | Number of geometries processed in this run. |
| `--out-csv` | auto | Output CSV path. Auto name: `sapt0_batch_{batch_index:03d}.csv`. |
| `--out-npz` | auto | Output NPZ path. Auto name: `sapt0_batch_{batch_index:03d}.npz`. |
| `--basis` | `aug-cc-pvdz` | Psi4 basis set for SAPT0. |

### `sample`

Samples dimers and computes dipoles directly (no intermediate geometry directory).

```bash
python -m moltimol.MLdms_general sample [options]
```

| Option | Default | Meaning |
|---|---:|---|
| `--n-samples` | `1000` | Number of dimer samples. |
| `--fileA` / `--fileB` | `CO.xyz` | Monomer XYZ inputs. |
| `--r-min` / `--r-max` | `3.0` / `8.0` | COM-distance range for sampling. |
| `--mode-frac` | `1/3` | Beta mode location in interval fraction `(0, 1)`. |
| `--concentration` | `10.0` | Beta concentration (`> 2`). |
| `--sigma-noise` | `0.1` | Cartesian noise level (angstrom). |
| `--seed` | `None` | Random seed. |
| `--out-csv` | `ml_dimer_data.csv` | CSV output path. |
| `--out-npz` | `None` | Optional NPZ output path. |
| `--method-low` | `propSAPT` | Output metadata column `method_low`. |
| `--method-high` | `None` | Output metadata column `method_high`. |
| `--sapt0` | off | Also include SAPT0 components when available. |

## Output columns (batch/sample)

Outputs include:

- `geom_id`, `method_low`, `method_high`, `R`
- body-frame coordinates for each atom as `x_<sym>Ai`, `y_<sym>Ai`, `z_<sym>Ai`, etc.
- body-frame dipole components for each reported dipole vector as `<name>_x`, `<name>_y`, `<name>_z`
- optional SAPT0 component columns when SAPT0 is enabled

Exact dipole column names depend on what `prop_sapt.calc_property(..., "dipole")` returns.

## Data curation utilities (`moltimol.data_curation`)

### Duplicate scan

```python
from moltimol.data_curation import find_duplicates_in_psi4_geoms

result = find_duplicates_in_psi4_geoms(
    geom_dir="psi4_geoms",
    q=0.01,
    round_decimals=None,
    hist_png="nn_hist.png",
    hist2d_png="nn_vs_R.png",
)
```

- `q`: quantile for nearest-neighbor distance threshold (`tau`)
- `round_decimals`: optional distance rounding before feature construction
- returns dict with `tau`, `pairs`, `nn_dist`, and `nn_idx`

### Collision scan

```python
from moltimol.data_curation import find_collisions_in_psi4_geoms

collisions = find_collisions_in_psi4_geoms(
    geom_dir="psi4_geoms",
    dmin=1.5,
    report=True,
)
```

- flags geometries where any inter-monomer atom pair is `< dmin` angstrom

### Prune pool (collisions + near-duplicates)

```python
from moltimol.data_curation import reduce_psi4geom_pool

reduce_psi4geom_pool(
    geom_dir="psi4_geoms",
    out_dir="psi4_geoms_pruned",
    dmin=1.5,
    q=0.01,
    round_decimals=None,
    kept_list_path="pruned_kept.txt",
)
```

- `kept_list_path` is a backward-compatible alias of `list_txt`

### FPS reduction

```python
from moltimol.data_curation import fps_reduce_psi4geoms

fps_reduce_psi4geoms(
    geom_dir="psi4_geoms",
    out_dir="psi4_geoms_fps",
    k=1000,
    start_idx=0,
    round_decimals=None,
    list_txt="fps_kept.txt",
)
```

- `k`: number of geometries to keep
- `start_idx`: initial seed index in sorted file order

### Angular and bond-length diagnostics

```python
from moltimol.data_curation import com_axis_angle_histograms, bond_length_histograms

com_axis_angle_histograms("psi4_geoms", out_dir="angle_histograms", bins=60, max_files=None)
bond_length_histograms("psi4_geoms", out_dir="bond_histograms", bins=50, max_files=None)
```

## Convert `.psi4geom` to `.xyz`

```bash
python psi4geom_to_xyz.py psi4_geoms --out-dir psi4_geoms_xyz --suffix _dimer
```

Options:

- positional `input`: one `.psi4geom` file or a directory
- `--out-dir` (default `psi4geom_xyz`)
- `--suffix` (default empty string; appended before `.xyz`)

## Tests

Run all tests:

```bash
pytest
```

Run dipole frame consistency test:

```bash
pytest -k dipole_rotation_consistency
```

This checks that body-frame dipoles match lab-frame dipoles rotated by the frame matrix `B`.
