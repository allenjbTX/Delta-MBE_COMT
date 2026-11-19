#!/usr/bin/env python3
"""Train the two-head active-QM residual-closure MBE(2) model."""

import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SHARED_CLOSURE_DIRECTORY = HERE.parent / "closure-model"
ACTIVEQM_DIRECTORY = HERE.parent / "closure-model-activeQM"
for directory in (SHARED_CLOSURE_DIRECTORY, ACTIVEQM_DIRECTORY):
    if not directory.is_dir():
        raise FileNotFoundError("required implementation not found: {}".format(directory))

sys.path.insert(0, str(SHARED_CLOSURE_DIRECTORY))
import train_delta_mlip_closure as shared_trainer  # noqa: E402

sys.path.insert(0, str(ACTIVEQM_DIRECTORY))
from activeqm_model import configure_zero_energy_heads  # noqa: E402
from activeqm_npz_loader import NpzDataLoader  # noqa: E402


# Reuse the common closure loss and grouped-snapshot DDP implementation while
# keeping serialization and logs local to this model directory.
shared_trainer.__file__ = str(HERE / shared_trainer.__file__.split("/")[-1])
shared_trainer.NpzDataLoader = NpzDataLoader
shared_trainer.zero_initialize_higher_order_heads = configure_zero_energy_heads


if __name__ == "__main__":
    os.chdir(HERE)
    sys.exit(shared_trainer.main())

