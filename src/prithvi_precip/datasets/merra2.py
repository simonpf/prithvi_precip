"""
prithvi_precip.datasets.merra
=============================

Provides datasets for loading MERRA-2 data.
"""
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Iterator, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import xarray as xr

from ..utils import (
    find_input_files,
    load_climatology,
    load_static_input,
    load_dynamic_input,
    to_datetime,
    to_datetime64,
)
from ..data.merra2 import SURFACE_VARS, VERTICAL_VARS


LOGGER = logging.getLogger(__name__)


class MERRAInputData(Dataset):
    """
    A PyTorch Dataset for loading 3-hourly MERRA2 data organized as input for the Prithvi-WxC FM.
    """
    def __init__(
            self,
            training_data_path: Union[Path, str],
            input_time: int = 3,
            lead_times: Optional[List[int]] = None,
            climatology: bool = True,
            center_meridionally: bool = True
    ):

        """
        Args:
            training_data_path (str): Path pointing to the directory containing the dynamic MERRA2
                input data in year/month/day folders.
            input_time: The time step in hours between the two input steps.
            climatology: Whether or not to include climatology data in the input.
            center_meridionally: Whether to center input grids meridionally instad of removing the last row
                 (which is the default for the original Prithvi-WxC)
        """
        self.training_data_path = Path(training_data_path)
        self.data_path = self.training_data_path.parent
        self.times, self.input_files = find_input_files(self.training_data_path, source="merra2")
        self.climatology = climatology

        self.input_time = input_time
        self.time_step = lead_times[0]
        self.lead_times = lead_times

        self.input_indices, self.output_indices = self.calculate_valid_samples()
        self._pos_sig = None
        self.center_meridionally = center_meridionally


    def calculate_valid_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates a tuple of index arrays containing pointing to the input- and output files
        for all training data samples satifying the requested input and lead time combination.

        Return: A tuple '(input_indices, output_indices)' with `input_indices` of shape
            '(n_samples, n_input_times)' containing the indices of all the input files for each data
            samples. Similarly, 'output_indices' is a numpy.ndarray of shape '(n_samples, n_lead_times)'
            containing the corresponding file indices to load for the output data.
        """
        input_indices = []
        output_indices = []
        for ind, sample_time in enumerate(self.times):
            input_times = [sample_time + np.timedelta64(t_i, "h") for t_i in [-self.input_time, 0]]
            lead_times = [sample_time + np.timedelta64(t_l, "h") for t_l in self.lead_times]
            valid = (
                all([t_i in self.times for t_i in input_times]) and
                all([t_l in self.times for t_l in lead_times])
            )
            if valid:
                input_indices.append([ind + t_i // 3 for t_i in [-self.input_time, 0]])
                output_indices.append([ind + t_l // 3 for t_l in self.lead_times])
        return np.array(input_indices), np.array(output_indices)

    def has_input(self, time: np.datetime64) -> bool:
        """
        Determine whether dynamic input for the given time stamp is available.
        """
        return time in self.times

    def get_lonlats(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return longitude and latitude coordinates of MERRA2 data.
        """
        static = load_static_input(self.data_path)
        lats = static.latitude.data
        lons = static.longitude.data
        return lons, lats

    def get_forecast_input_static(
            self,
            initialization_time: np.datetime64,
            forecast_steps: int
    ):
        """
        Get static forecast input.

        Returns static forecast input for all forecast steps.

        Args:
            initialization_time: The forecast initialization time.
            forecast_steps: The number of forecast steps.

        """
        time_steps = (
            initialization_time + (np.arange(forecast_steps) * self.lead_times[0]).astype("timedelta64[h]")
        )
        # Removes one row along lat dimension.
        if self.center_meridionally:
            center = lambda tnsr: 0.5 * (tnsr[..., 1:, :] + tnsr[..., :-1, :])
            static_data = [center(torch.tensor(load_static_input(time, self.data_path))) for time in time_steps]
        else:
            pad = partial(nn.functional.pad, pad=((0, 0, 0, -1)))
            static_data = [pad(torch.tensor(load_static_input(time, self.data_path))) for time in time_steps]

        static_data = torch.stack(static_data)
        return static_data

    def get_forecast_input_climate(
            self,
            initialization_time: np.datetime64,
            forecast_steps: int
    ):
        """
        Get climatology input for forecast.

        Args:
            initialization_time: The forecast initialization time.
            forecast_steps: The number of forecast steps.

        """
        time_steps = (
            initialization_time + (np.arange(1, forecast_steps + 1) * self.lead_times[0]).astype("timedelta64[h]")
        )
        # Removes one row along lat dimension.
        if self.center_meridionally:
            center = lambda tnsr: 0.5 * (tnsr[..., 1:, :] + tnsr[..., :-1, :])
            climates = [center(torch.tensor(load_climatology(time, self.data_path))) for time in time_steps]
        else:
            pad = partial(nn.functional.pad, pad=((0, 0, 0, -1)))
            climates = [pad(torch.tensor(load_climatology(time, self.data_path))) for time in time_steps]

        return torch.stack(climates)

    def __len__(self) -> int:
        """
        The number of samples in the dataset.
        """
        return len(self.input_indices)

    def __getitem__(self, ind: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a single data point from the dataset.
        """
        input_files = [self.input_files[ind] for ind in self.input_indices[ind]]
        input_times = [self.times[ind] for ind in self.input_indices[ind]]
        output_files = [self.input_files[ind] for ind in self.output_indices[ind]]
        output_times = [self.times[ind] for ind in self.output_indices[ind]]

        dynamic_in = [load_dynamic_input(self.training_data_path / path) for path in input_files]
        static_in = torch.tensor(load_static_input(input_times[-1], self.data_path))

        input_time = (input_times[1] - input_times[0]).astype("timedelta64[h]").astype(np.float32)
        lead_time = (output_times[0] - input_times[1]).astype("timedelta64[h]").astype(np.float32)

        dynamic_out = [load_dynamic_input(self.training_data_path / path) for path in output_files]
        climate = [torch.tensor(load_climatology(time, self.data_path)) for time in output_times]

        # Remove one row along lat dimension.
        if self.center_meridionally:
            transform = lambda tnsr: 0.5 * (tnsr[..., 1:, :] + tnsr[..., :-1, :])
        else:
            transform = partial(nn.functional.pad, pad=((0, 0, 0, -1)))

        x = {
            "x": transform(torch.stack(dynamic_in, 0)),
            "static": transform(static_in),
            "climate": transform(torch.tensor(climate[0])),
            "input_time": torch.tensor(input_time),
            "lead_time": torch.tensor(lead_time)
        }
        y = transform(torch.tensor(dynamic_out[0]))

        return x, y

    def get_direct_forecast_input(self, init_time: np.datetime64, n_steps: int) -> Dict[str, torch.Tensor]:
        """
        Get forecast input data to perform a continuous forecast over a given number of steps
        using a direct forecasting model.

        Args:
            init_time: The initialization time of the forecast.
            n_steps: The number of steps to forecast.

        Return:
            A dictionary contraining the loaded input tensors.
        """
        input_times = [init_time + np.timedelta64(t_i * self.input_time, "h") for t_i in [-1, 0]]
        for input_time in input_times:
            if input_time not in self.input_times:
                raise ValueError(
                    "Required input data for t=%s not available.",
                    input_time
                )

        dynamic_in = []
        for input_time in input_times:
            ind = np.searchsorted(self.input_times, input_time)
            dynamic_in.append(load_dynamic_input(self.training_data_path, self.input_files[ind]))

        static_time = input_times[-1]
        static_in = load_static_input(static_time, self.data_path)

        if self.center_meridionally:
            transform = lambda tnsr: 0.5 * (tnsr[..., 1:, :] + tnsr[..., :-1, :])
        else:
            transform = partial(nn.functional.pad, pad=((0, 0, 0, -1)))

        dynamic_in = transform(torch.stack(dynamic_in, 0))[None].repeat(n_steps, 1, 1, 1, 1)
        static_in = transform(torch.tensor(static_in))[None].repeat(n_steps, 1, 1, 1)
        input_time = self.input_time * torch.ones(n_steps)
        lead_time = self.input_time * torch.arange(1, n_steps + 1).to(dtype=torch.float32)

        x = {
            "x": dynamic_in,
            "static": static_in,
            "lead_time": lead_time,
            "input_time": input_time,
        }

        if self.climate:
            output_times = [init_time + step * np.timedelta64(self.input_time, "h") for step in range(1, n_steps + 1)]
            climate = [torch.tensor(load_climatology(time, self.data_path)) for time in output_times]
            climate = transform(torch.stack(climate))
            x["climate"] = climate

        if self.obs_loader is not None:

            obs = []
            meta = []
            for time_ind, time in enumerate(input_times):
                obs_t, meta_t = self.obs_loader.load_observations(time, offset=len(input_times) - time_ind - 1)
                obs.append(obs_t)
                meta.append(meta_t)
            obs = torch.stack(obs, 0)
            obs_mask = torch.zeros_like(obs) #obs < -2.9
            obs = torch.nan_to_num(obs, nan=-3.0)
            meta = torch.stack(meta, 0)

            x["obs"] = obs[None].repeat_interleave(n_steps, 0)
            x["obs_mask"] = obs_mask[None].repeat_interleave(n_steps, 0)
            x["obs_meta"] = meta[None].repeat_interleave(n_steps, 0)

        return x

    def get_batched_direct_forecast_input(
            self,
            init_time: np.datetime64,
            n_steps: int,
            batch_size: int
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Same as get_direct_forecast_input but returns an iterator over the batched input.

        Args:
            init_time: The initialization time of the forecast.
            n_steps: The number of steps to forecast.
            batch_size: The size of each batch.

        Return:
            An iterator yielding the input data in batches of the requested size.
        """
        x = self.get_direct_forecast_input(init_time, n_steps=n_steps)
        batch_start = 0
        n_samples = x["x"].shape[0]
        while batch_start < n_samples:
            batch_end = batch_start + batch_size
            yield {name: tnsr[batch_start:batch_end] for name, tnsr in x.items()}
            batch_start = batch_end
