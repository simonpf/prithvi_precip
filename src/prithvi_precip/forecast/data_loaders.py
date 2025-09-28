"""
prithvi_precip.forecast
=======================

Functionality for running forecasts with the Prithvi Precip model.
"""
from datetime import datetime
from functools import partial
from math import ceil
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
import xarray as xr

from ..datasets.observations import ObservationLoader
from ..utils import (
    find_input_files,
    load_climatology,
    load_dynamic_input,
    load_static_input,
)


class DirectForecastLoader:
    """
    Class to load input data for direct forecasts.
    """
    def __init__(
            self,
            input_data_path: Path,
            init_times: np.ndarray,
            n_steps: int,
            input_time: int = 3,
            source: str = "merra2",
            batch_size: Optional[int] = None,
            center_meridionally: bool = True,
            observation_layers: Optional[int] = None,
            n_tiles: Tuple[int, int] = (12, 18),
            tile_size: Tuple[int, int] = (30, 32),
    ):
        """
        Args:
            input_data_path: Path pointing to the input dat.a
            init_times: Array specifying the initialization times
            n_steps: The number of forecast steps to perform.
            input_time: The tmie difference between input steps in hours.
            source: Which dataset the input is derived from ('merra2' or 'geos')
            batch_size: The batch size to use.
            center_meridionally: Set to True to averaeg input instead of cropping.
            observation_layers: Set to a positive number specifying the observation layers
                 to load to enable observation loader.
            n_tiles: A tuple specifying the number of meridional and zonal observation tiles,
                 respectively.
            tile_size: A  tuple specifying the zonal and meridional size of the observation tiles.
        """
        self.input_data_path = Path(input_data_path)
        self.input_time = input_time

        self.init_times = init_times
        self.input_times, self.input_files = find_input_files(self.input_data_path, source=source)
        self.input_indices = self.calculate_valid_samples()
        self.center_meridionally = center_meridionally

        if batch_size is None:
            self.batch_size = 1
        else:
            self.batch_size = batch_size

        self.n_steps = n_steps
        self.batches_per_input = ceil(n_steps / self.batch_size)

        if observation_layers is not None:
            self.obs_loader = ObservationLoader(
                Path(input_data_path) / "obs",
                n_tiles=n_tiles,
                tile_size=tile_size
            )
        else:
            self.obs_loader = None

    def __len__(self) -> int:
        return len(self.input_indices) * self.batches_per_input

    def calculate_valid_samples(self) -> np.ndarray:
        """
        Calculate the indices of valid forecasts input files.
        """
        input_indices = []
        for ind, sample_time in enumerate(self.input_times):
            if sample_time not in self.init_times:
                continue
            input_times = [sample_time + np.timedelta64(t_i * self.input_time, "h") for t_i in [-1, 0]]
            valid = all([t_i in self.input_times for t_i in input_times])
            if valid:
                prev_ind = np.searchsorted(self.input_times, input_times[0])
                input_indices.append([prev_ind, ind])
        return np.array(input_indices)


    def __getitem__(self, ind: int) -> Tuple[np.datetime64, np.datetime64, Dict[str, torch.tensor]]:
        """
        Load batch of input data.

        Returns:
            A tuple ``(init_time, valid_time, input_data)`` containing the initialization time (``init_time``),
            the valid forecast time (``valid_time``), and a dictionary containing the input data (``input_data``).
        """
        init_ind = ind // self.batches_per_input
        step_start = (ind % self.batches_per_input) * self.batch_size
        step_end = min(step_start + self.batch_size, self.n_steps)

        input_files = [self.input_files[ind] for ind in self.input_indices[init_ind]]
        input_times = [self.input_times[ind] for ind in self.input_indices[init_ind]]
        dynamic_in = [load_dynamic_input(self.input_data_path / path) for path in input_files]
        static = torch.tensor(load_static_input(input_times[-1], self.input_data_path.parent))

        init_time = input_times[-1]

        input_time = (input_times[1] - input_times[0]).astype("timedelta64[h]").astype(np.float32)

        # Remove one row along lat dimension.
        if self.center_meridionally:
            transform = lambda tnsr: 0.5 * (tnsr[..., 1:, :] + tnsr[..., :-1, :])
            transform_3d = lambda tnsr: 0.5 * (tnsr[..., 1:, :] + tnsr[..., :-1, :])
        else:
            transform = lambda tnsr: 0.5 * (tnsr[..., 1:, :] + tnsr[..., :-1, :])
            transform_3d = lambda tnsr: 0.5 * (tnsr[..., 1:, :] + tnsr[..., :-1, :])

        climates = []
        valid_times = []
        lead_times = []

        for output_step in range(step_start, step_end):
            lead_time = (output_step + 1) * np.timedelta64(int(input_time), "h")
            lead_times.append(torch.tensor(lead_time.astype(np.float32)))
            output_time = init_time + lead_time
            valid_times.append(output_time)
            climate = torch.tensor(load_climatology(output_time, self.input_data_path.parent))
            climates.append(transform_3d(climate))

        climate = torch.stack(climates)
        valid_times = np.array(valid_times)

        dynamic_in = torch.repeat_interleave(
            transform_3d(torch.stack(dynamic_in))[None],
            step_end - step_start,
            0
        )
        static = torch.repeat_interleave(
            transform(static)[None],
            step_end - step_start,
            0
        )

        input_data = {
            "x": dynamic_in,
            "static": static,
            "climate": climate,
            "input_time": torch.repeat_interleave(torch.tensor(input_time)[None], step_end - step_start, 0),
            "lead_time": torch.stack(lead_times)
        }

        if self.obs_loader is not None:
            obs = []
            obs_meta = []
            for time_ind, time in enumerate(input_times):
                obs_t, meta_t = self.obs_loader.load_observations(time, offset=len(input_times) - time_ind - 1)
                obs.append(obs_t)
                obs_meta.append(meta_t)

            obs = torch.stack(obs, 0)
            obs_mask = torch.zeros_like(obs) #obs < -2.9
            obs = torch.nan_to_num(obs, nan=-3.0)
            obs_meta = torch.stack(obs_meta, 0)

            input_data["obs"] = torch.repeat_interleave(obs[None], step_end - step_start, 0)
            input_data["obs_mask"] = torch.repeat_interleave(obs_mask[None], step_end - step_start, 0)
            input_data["obs_meta"] = torch.repeat_interleave(obs_meta[None], step_end - step_start, 0)

        return init_time, valid_times, input_data


class AutoregressiveForecastLoader:
    """
    Class to load input data for autoregressive forecasts.
    """
    def __init__(
            self,
            input_data_path: Path,
            init_times: np.ndarray,
            n_steps: int,
            input_time: int = 3,
            source: str = "merra2",
            batch_size: Optional[int] = None,
            center_meridionally: bool = True
    ):
        self.input_data_path = Path(input_data_path)
        self.input_time = input_time

        self.init_times = init_times
        self.input_times, self.input_files = find_input_files(self.input_data_path, source=source)
        self.input_indices = self.calculate_valid_samples()
        self.center_meridionally = center_meridionally

        if batch_size is None:
            self.batch_size = 1
        else:
            self.batch_size = batch_size

        self.n_steps = n_steps
        self.batches_per_input = ceil(n_steps / self.batch_size)

    def __len__(self) -> int:
        return ceil(len(self.input_indices) / self.batch_size)

    def calculate_valid_samples(self) -> np.ndarray:
        """
        Calculate the indices of valid forecasts input files.
        """
        input_indices = []
        for ind, sample_time in enumerate(self.input_times):
            if sample_time not in self.init_times:
                continue
            input_times = [sample_time + np.timedelta64(t_i * self.input_time, "h") for t_i in [-1, 0]]
            valid = all([t_i in self.input_times for t_i in input_times])
            if valid:
                prev_ind = np.searchsorted(self.input_times, input_times[0])
                input_indices.append([prev_ind, ind])
        return np.array(input_indices)

    def load_input(self, input_index: int) -> Tuple[np.datetime64, Dict[str, torch.Tensor]]:
        """
        Load autoregressive forecast data for a single forecast.

        Args:
            input_index: The input index identifying the forecast withing the initialization times with valid
                input data.

        Returns:
            A tuple ``init_time, input_data`` containing the initialization time as np.datetime64 object and
            a dictionary of the torch.Tensor objects containing the input data.

        """
        input_indices = self.input_indices[input_index]
        init_time = self.input_times[input_indices[-1]]

        input_time = torch.tensor(self.input_time).to(dtype=torch.float32)
        input_time = torch.repeat_interleave(input_time[None], self.n_steps, 0)
        lead_time = input_time

        dynamic = [load_dynamic_input(self.input_data_path / self.input_files[ind]) for ind in input_indices]

        static_times = [init_time + np.timedelta64(self.input_time, "h") * step for step in range(0, self.n_steps)]
        static = [
            torch.tensor(load_static_input(static_time, self.input_data_path.parent)) for static_time in static_times
        ]

        clim_times = [init_time + np.timedelta64(self.input_time, "h") * step for step in range(1, self.n_steps + 1)]
        climate = [
            torch.tensor(load_climatology(clim_time, self.input_data_path.parent)) for clim_time in clim_times
        ]

        # Remove one row along lat dimension.
        if self.center_meridionally:
            transform = lambda tnsr: 0.5 * (tnsr[..., 1:, :] + tnsr[..., :-1, :])
            transform_3d = lambda tnsr: 0.5 * (tnsr[..., 1:, :] + tnsr[..., :-1, :])
        else:
            transform = partial(nn.functional.pad, pad=(0, 0, 0, -1))
            transform_3d = partial(nn.functional.pad, pad=(0, 0, 0, -1, 0, 0), mode="constant", value=0)

        inpt = {
            "x": transform(torch.stack(dynamic)),
            "static": transform(torch.stack(static)),
            "climate": transform_3d(torch.stack(climate)),
            "input_time": input_time,
            "lead_time": lead_time
        }

        return init_time, inpt



    def __getitem__(self, ind: int) -> Tuple[np.datetime64, np.datetime64, Dict[str, torch.tensor]]:
        """
        Load batch of input data.

        Returns:
            A tuple ``(init_time, valid_time, input_data)`` containing the initialization time (``init_time``),
            the valid forecast time (``valid_time``), and a dictionary containing the input data (``input_data``).
        """
        batch_start = ind * self.batch_size
        batch_end = min(batch_start + self.batch_size, self.input_indices.shape[0])

        batch = {}
        init_times = []

        for sample_ind in range(batch_start, batch_end):
            init_time, inpt = self.load_input(sample_ind)
            init_times.append(init_time)
            for var, tnsr in inpt.items():
                batch.setdefault(var, []).append(tnsr)

        init_times = np.stack(init_times)
        valid_times = init_times[:, None] + np.arange(1, self.n_steps + 1) * np.timedelta64(self.input_time, "h")[None]

        batch = {
            name: torch.stack(tnsrs) for name, tnsrs in batch.items()
        }
        return init_times, valid_times, batch
