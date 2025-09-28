"""
prithvi_precip.forecast.runners
===============================

Functions to drive forecasts across several inputs.
"""
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import xarray as xr


def post_process_results(inpt: Dict[str, torch.Tensor], results: Dict[str, torch.Tensor]) -> xr.Dataset:
    """
    Default post processing function extracting results from forecast and storing
    them in an xr.Dataset.

    Args:
        inpt: The batch containing the input data.
        results: A dictionary containing the forecast results.
    """
    static = inpt["static"]
    if static.dim() == 5:
        lats = np.rad2deg(inpt["static"][0, 0, 0, :, 0].float().cpu().numpy())
        lons = np.rad2deg(inpt["static"][0, 0, 1, 0, :].float().cpu().numpy())
    else:
        lats = np.rad2deg(inpt["static"][0, 0, :, 0].float().cpu().numpy())
        lons = np.rad2deg(inpt["static"][0, 1, 0, :].float().cpu().numpy())

    dataset = xr.Dataset({
        "latitude": (("latitude",), lats),
        "longitude": (("longitude",), lons)
    })

    for key, tnsr in results.items():
        if isinstance(tnsr, list):
            res = [tensor.expected_value().float().cpu().numpy()[:, 0] for tensor in tnsr]
            dataset[key] = (("batch", "step", "latitude", "longitude"), np.stack(res, axis=1))
        else:
            res = tnsr.expected_value().float().cpu().numpy()[:, 0]
            dataset[key] = (("batch", "latitude", "longitude"), res)
        dataset[key].encoding = {"dtype": np.float32, "zlib": True}

    return dataset


def run_direct_forecast(
        model: nn.Module,
        data_loader: Any,
        output_path: Path,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        post_process_fn: Callable[[Dict[str, Any]], xr.Dataset] = post_process_results,
        forward_kwargs: Optional[Dict[str, Any]] = None
):
    """
    Run forecast and store results.

    Args:
        model: A torch.nn.Module containing the model to use to perform the forecasts.
        data_loader: A data loader to use to load the input data.
        output_path: The path to which to write the results.
        dtype: The dtype to use for the forecast.
        device: The device to run the forecast on.
        post_process_fn: A post-processing function to use to turn the raw model outputs into
            an xarray.Dataset containing the results.
        forward_kwargs: Keyword arguments forwarded to the models forward method.
    """
    model = model.to(device=device, dtype=dtype).eval()
    results = []

    if forward_kwargs is None:
        forward_kwargs = {}

    for inpt in tqdm(iter(data_loader), total=len(data_loader)):

        init_time, valid_times, batch = inpt
        batch = {
            name: tnsr.to(device=device, dtype=dtype) for name, tnsr in batch.items()
        }

        with torch.no_grad():
            res = post_process_fn(batch, model(batch, **forward_kwargs))
            res = res.rename(batch="valid_time")
            res["initialization_time"] = init_time
            res["valid_time"] = (("valid_time",), valid_times)

        date = init_time.astype("datetime64[s]").item()
        fname = date.strftime("forecast_%Y%m%d%H%M.nc")
        output_file = output_path / fname
        if output_file.exists():
            existing = xr.load_dataset(output_file)
            res = xr.concat([res, existing], "valid_time").sortby("valid_time")
            res.to_netcdf(output_file)
        else:
            res.to_netcdf(output_file)


def run_autoregressive_forecast(
        model: nn.Module,
        data_loader: Any,
        output_path: Path,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        post_process_fn: Callable[[Dict[str, Any]], xr.Dataset] = post_process_results
):
    """
    Run autoregressive forecast and store results.

    Args:
        model: A torch.nn.Module containing the model to use to perform the forecasts.
        data_loader: A data loader to use to load the input data.
        output_path: The path to which to write the results.
        dtype: The dtype to use for the forecast.
        device: The device to run the forecast on.
        post_process_fn: A post-processing function to use to turn the raw model outputs into
            an xarray.Dataset containing the results.
    """
    model = model.to(device=device, dtype=dtype).eval()
    results = []

    for inpt in tqdm(iter(data_loader), total=len(data_loader)):

        init_times, valid_times, batch = inpt
        batch = {
            name: tnsr.to(device=device, dtype=dtype) for name, tnsr in batch.items()
        }

        with torch.no_grad():
            res = post_process_fn(batch, model(batch)).rename(step="valid_time")
            res["valid_time"] = (("batch", "valid_time"), valid_times)
            res["initialization_time"] = (("batch",), init_times)

        for batch_ind in range(res.batch.size):
            res_b = res[{"batch": batch_ind}]
            init_time = res_b.initialization_time.data
            date = init_time.astype("datetime64[s]").item()

            fname = date.strftime("forecast_%Y%m%d%H%M.nc")
            output_file = output_path / fname
            res_b.to_netcdf(output_file)
