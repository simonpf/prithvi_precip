from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import xarray as xr
import os

import torch
from torch.utils.data import DataLoader
from prithvi_precip.forecast.data_loaders import AutoregressiveForecastLoader
from pytorch_retrieve.architectures import load_and_compile_model
from pytorch_retrieve.training import load_weights
from prithvi_precip.forecast.runners import run_autoregressive_forecast


PRITHVI_DATA_PATH = "/path/to/scaling/factors"
INPUT_DATA_PATH = "/path/to/input/data"
OUTPUT_PATH = "/path/to/output/data"
START_TIME = np.datetime64("2020-01-01")
END_TIME = np.datetime64("2020-01-01T01:00:00")
DEVICE = "cuda:0"

os.environ['PRITHVI_DATA_PATH'] = PRITHVI_DATA_PATH
mdl = load_and_compile_model("model.toml")
model_weights = Path("/mnt/ssd-data1/prithvi_tc/prithvi.wxc.rollout.2300m.v1.pt")
load_weights(
    {"backbone": model_weights},
    mdl
)


def post_process_results(
    inpt: Dict[str, torch.Tensor],
    results: Dict[str, torch.Tensor],
    init_times: np.ndarray,
    valid_times: np.ndarray,
) -> xr.Dataset:
    """
    Extracts surface winds, surface pressure and 850 winds from forecast results.

    Args:
        inpt: The batch containing the input data.
        results: A dictionary containing the forecast results.

    Return:
        An xarray.Dataset containing the results to be stored.
    """
    static = inpt["static"]
    if static.dim() == 5:
        lats = np.rad2deg(inpt["static"][0, 0, 0, :, 0].float().cpu().numpy())
        lons = np.rad2deg(inpt["static"][0, 0, 1, 0, :].float().cpu().numpy())
    else:
        lats = np.rad2deg(inpt["static"][0, 0, :, 0].float().cpu().numpy())
        lons = np.rad2deg(inpt["static"][0, 1, 0, :].float().cpu().numpy())

    var_inds = [9, 17, 18, -18, -4]

    # Stack output steps into single array
    pred = np.stack([step[:, [9, 17, 18, -18, -4]].float().cpu().numpy() for step in results["y"]], axis=1)
    # Add analysis to forecast results
    analysis = inpt["x"][:, 1:, var_inds].float().cpu().numpy()
    pred = np.concatenate((analysis, pred), 1)
    valid_times = np.concatenate([init_times.reshape((init_times.shape[0], 1)), valid_times], 1)

    dataset = xr.Dataset({
        "initialization_time": (("batch",), init_times),
        "valid_time": (("batch", "step",), valid_times),
        "latitude": (("latitude",), lats),
        "longitude": (("longitude",), lons)
    })

    slp = pred[:, :, 0]
    u10 = pred[:, :, 1]
    v10 = pred[:, :, 2]
    u850 = pred[:, :, 3]
    v850 = pred[:, :, 4]

    dataset["slp"] = (("batch", "step", "latitude", "longitude"), slp)
    dataset["u10"] = (("batch", "step", "latitude", "longitude"), u10)
    dataset["v10"] = (("batch", "step", "latitude", "longitude"), v10)
    dataset["u850"] = (("batch", "step", "latitude", "longitude"), u850)
    dataset["v850"] = (("batch", "step", "latitude", "longitude"), v850)

    for var in ["slp", "u10", "v10", "u850", "v850"]:
        dataset[var].encoding = {"dtype": "float32", "zlib": True}

    return dataset



# Adapt this to run forecast for longer time range.
init_times = np.arange(
    np.timedelta64(6, "h")
)

data_loader = AutoregressiveForecastLoader(
    Path(INPUT_DATA_PATH),
    init_times=init_times,
    n_steps=20, # Number of 6-hour forecast steps.
    input_time=6, # Input data time step
    center_meridionally=False,
    full_climatology=True,
)
data_loader = DataLoader(data_loader, batch_size=None, num_workers=1, collate_fn=lambda x: x)
print(f"Found input data for {len(data_loader)} forecasts.")


output_path = Path(OUTPUT_PATH)
output_path.mkdir(exist_ok=True)
device = DEVICE
run_autoregressive_forecast(
    mdl,
    data_loader,
    output_path,
    post_process_fn=post_process_results,
    device=device,
    dtype=torch.float32,
)

