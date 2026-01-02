#!/usr/bin/env python
# coding: utf-8

# # Run direct forecast
# 
# This notebook runs direct forecasts for evaluating the Prithvi Precip retrievals.

# In[1]:
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import xarray as xr

from pytorch_retrieve import load_model
from prithvi_precip.forecast import (
    DirectForecastLoader,
    run_direct_forecast
)
from torch.utils.data import DataLoader
from prithvi_precip.forecast import run_direct_forecast

#
# CONFIGURATION
#

MODEL_PATH = "prithvi_precip_direct.pt"
INPUT_PATH = "/data1/prithvi_precip/data/test_data_geos/"
OUTPUT_PATH = "prithvi_precip_direct.pt"
START_TIME = np.datetime64("2025-03-01")
END_TIME = np.datetime64("2025-09-01")

output_path = Path(OUTPUT_PATH)
output_path.mkdir(exist_ok=True)

#
# MODEL
#
mdl = load_model(MODEL_PATH)


#
# DATA LOADER
#
init_times = np.arange(START_TIME, END_TIME, np.timedelta64(24, "h"))
input_data = DirectForecastLoader(
    input_data_path=INPUT_PATH,
    init_times=init_times,
    n_steps=32,
    source="geos",
    center_meridionally=False,
    batch_size=32,
    observation_layers=32
)
data_loader = DataLoader(
    input_data,
    batch_size=None,
    collate_fn=lambda x: x,
    num_workers=8
)


#
# INFERENCE
#
run_direct_forecast(
    model=mdl,
    data_loader=data_loader,
    output_path=output_path,
    dtype=torch.bfloat16,
    device="cuda:1",
    forward_kwargs={"obs_only": True}
)
