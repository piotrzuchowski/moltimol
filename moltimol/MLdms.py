"""
Thin wrapper around MLdms_general for backward compatibility.
"""
import os
import sys

# Allow running as a script while still importing local package code.
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from moltimol.MLdms_general import (  # noqa: F401
    body_frame_coord_columns,
    compute_sapt0_batch,
    generate_psi4geom_files,
    run_propsapt_batch,
    sample_R_beta_mode,
    sample_dimer_geometries,
    setup_psi4_defaults,
)


def sample_co_dimer_geometries(*args, **kwargs):
    """
    Backward-compatible alias for sample_dimer_geometries.
    """
    return sample_dimer_geometries(*args, **kwargs)


def main():
    """
    Entry point that forwards to the general CLI.
    """
    from moltimol.MLdms_general import main as _main

    _main()


if __name__ == "__main__":
    main()
