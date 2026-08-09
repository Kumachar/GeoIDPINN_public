"""CPU-safe launcher for the fair spatial NB forecasting experiment."""

import os
import runpy
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
experiment_path = Path(__file__).with_name("real_data_fair_forecast_spatial_nb_experiment.py")
globals().update(runpy.run_path(str(experiment_path), run_name="__main__"))
