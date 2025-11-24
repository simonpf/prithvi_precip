"""
prithvi_precip.datasets.wtcm
============================

Provides datasets for loading WTCM winds.
"""

from datetime import datetime
from functools import cached_property, partial
import logging
from math import trunc
import os
from pathlib import Path
import shutil
from time import sleep
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
from scipy.ndimage import binary_dilation
import torch
from torch import nn
from torch.utils.data import Dataset
import xarray as xr
from PrithviWxC.dataloaders.merra2 import output_scalers

from .merra2 import MERRAInputData
from .precipitation import (
    AutoregressivePrecipForecastDataset,
    DirectPrecipForecastDataset,
    _transform_data,
    SURFACE_VARS,
    VERTICAL_VARS,
    LEVELS,
)
from ..utils import (
    find_input_files,
    load_and_interp_climatology,
    load_dynamic_input,
    load_static_input,
    to_datetime64
)


LOGGER = logging.getLogger(__name__)


def get_date_wtcm(path: Path) -> datetime:
    """
    Extract timestamp from WTCM file.

    Args:
        path: The path pointing to the WTCM file.

    Return:
        A datetime object representing the timestamp.
    """
    parts = path.name.split("_")
    return datetime.strptime(parts[3], "%Y%m%d%H")


def get_n_wtcm(path: Path) -> int:
    """
    Extract number of  WTCM files from filename.

    Args:
        path: The path pointing to the WTCM file.

    Return:
        The number of storms in the file.
    """
    parts = path.name.split("_")
    return int(parts[5:-3], "%Y%m%d%H")


class DirectWTCMForecastDataset(DirectPrecipForecastDataset):
    """
    A PyTorch Dataset for loading WTCM winds for forecasts without unrolling.
    """
    def __init__(
            self,
            training_data_path: Union[Path, str],
            input_time: Union[int, List[int]] = 3,
            lead_time: int = 3,
            accumulation_period: int = 3,
            max_steps: int = 24,
            climate: bool = True,
            sampling_rate: float = 1.0,
            center_meridionally: bool = True,
            validation: bool = False,
            local_data: Optional[Path] = None,
            augment: bool = False,
            source: str = "merra2"
    ):
        """
        Args:
            training_data_path: The directory containing the dynamic input data.
            input_time: A single int or a list of ints specifying the time different between the two
                input timesteps.
            lead_time: The lead time step.
            accumulation_period: The precipitation accumulation period.
            max_steps: The maximum number of timesteps to forecast precipitation.
            climate: Whether to include climatology data in the input.
            sampling_rate: Sub- or super-sample dataset.
            center_meridionally: If True, will use mid-point averaging to reduce the latitude dimension
                of the input data by one. If False, will use negative paddgin.
            validation: Flat indicating whether the dataset is used to load validation or training data.
            local_data: An optional path pointing to a location to which to copy the training data. This should
                typically be node-local memory that can be accessed rapidly.
            augment: Whether or not to augment the input data using random zonal rolls and meridional flips.
            source: The source of the input data: 'merra2' or 'geos'
        """
        self.training_data_path = Path(training_data_path)
        self.data_path = self.training_data_path.parent
        if not isinstance(input_time, list):
            self.input_steps = [input_time]
        else:
            self.input_steps = input_time

        if lead_time is None:
            lead_time = self.input_steps[0]
            LOGGER.info(
                "No explicit lead time provided. Falling back to input step %s h.",
                lead_time
            )
        self.lead_time = lead_time

        self.accumulation_period = accumulation_period
        self.max_steps = max_steps
        self.climate = climate
        self.sampling_rate = sampling_rate
        self.center_meridionally = center_meridionally
        self.validation = validation
        self.augment = augment

        self.local_data = None
        if local_data is not None:
            self.local_data = Path(local_data)
        self.source = source

        self.input_times, self.input_files = find_input_files(self.training_data_path, source=source)
        self.output_times, self.output_files = self.find_wtcm_files(self.training_data_path)

        self.input_indices, self.output_indices = self.calculate_valid_samples()
        self.rng = np.random.default_rng(seed=42)

        if self.local_data is not None:
            self.split_and_copy_files()


    def find_wtcm_files(
            self,
            training_data_path: Path,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find WTCM files for training.

        Args:
            training_data_path: A path object pointing to the directory containing the input data.

        Return:
            A tuple ``times, files`` containing the timestamps of the output files and the corresponding filenames.
        """
        times = []
        files = []

        for path in sorted(list(training_data_path.glob("wtcm/**/*.nc"))):
            try:
                date = get_date_wtcm(path)
                date64 = to_datetime64(date)
                n_storms = get_n_wtcm(path)
                if 0 < n_storms:
                    files.append(str(path.relative_to(training_data_path)))
                    times.append(date64)
            except ValueError:
                continue

        times = np.array(times)
        files = np.array(files)
        return times, files


    def __len__(self) -> int:
        """The number of samples in the dataset."""
        return trunc(len(self.input_indices) * self.sampling_rate)

    def load_wtcm_data(self, output_file: str) -> Dict[str, torch.tensor]:
        """
        Load WTCM winds from target-data file.

        Args:
            output_file: The relative path of the output file from which to load the data.

        Return:
            A dictionary containing the target names and corresponding tensors.
        """
        with xr.open_dataset(self.training_data_path / output_file) as data:
            mu10 = data["mu10"].data[0]
            mv10 = data["mv10"].data[0]
            target = {
                "u10": torch.tensor(mu10),
                "v10": torch.tensor(mv10),
            }
        return target

    def load_data(
            self,
            index: int,
            roll: int,
            flip_v: bool,
            flip_h: bool,
            scale: float = 1.0
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Load input and target data.

        Args:
            The index of the sample.
            roll: The number of pixels by which to roll latitudes.
            flip_v: Whether or not to flip the data meridionally.
            flip_h: Whether or not to flip the data zonally.
            scale: Apply scaling to data
        """
        input_time = self.input_steps[0]
        input_indices = self.input_indices[index]
        if self.validation:
            step_ind = int(index * self.sampling_rate) % len(self.input_steps)
            input_indices = [input_indices[step_ind], input_indices[-1]]
            input_time = self.input_steps[step_ind]
        else:
            if 1 < len(self.input_steps):
                step_ind = self.rng.integers(len(self.input_steps))
                input_indices = [input_indices[step_ind], input_indices[-1]]
                input_time = self.input_steps[step_ind]

        input_files = [self.input_files[ind] for ind in input_indices]
        input_times = [self.input_times[ind] for ind in input_indices]

        dynamic_in = [load_dynamic_input(self.training_data_path / path) for path in input_files]

        static_time = input_times[-1]
        static_in = torch.tensor(load_static_input(static_time, self.data_path))

        # Remove one row along lat dimension.
        pad = partial(nn.functional.pad, pad=((0, 0, 0, -1)))

        if self.center_meridionally:
            transform = lambda tnsr: 0.5 * (tnsr[..., 1:, :] + tnsr[..., :-1, :])
        else:
            transform = partial(nn.functional.pad, pad=((0, 0, 0, -1)))

        x = {
            "x": transform(torch.stack(dynamic_in, 0)),
            "static": transform(static_in),
            "input_time": torch.tensor(input_time).to(dtype=torch.float32),
        }

        # Apply perturbation to input
        if self.augment:
            d_x = x["x"][1] - x["x"][0]
            noise = 0.5 * torch.tensor(self.rng.normal(size=(2, d_x.shape[0], 1, 1)), dtype=np.float32)
            x["x"] += noise * d_x

        inds = self.output_indices[index]
        inds = inds[0 <= inds]

        if self.validation:
            output_ind = inds[int(index * self.sampling_rate) % len(inds)]
        else:
            output_ind = self.rng.choice(inds)
        output_file = self.output_files[output_ind]
        output_time = self.output_times[output_ind]

        lead_time = (output_time - max(input_times)).astype("timedelta64[h]").astype(np.float32)
        x["lead_time"] = torch.tensor(lead_time).to(dtype=torch.float32)

        if self.climate:
            climate = load_and_interp_climatology(output_time, self.data_path)
            x["climate"] = transform(torch.tensor(climate))
            if self.augment:
                noise = 0.5 * torch.tensor(self.rng.normal(size=(d_x.shape[0], 1, 1)), dtype=np.float32)
                x["climate"] += noise * d_x

        target = self.load_wtcm_data(output_file)

        if self.augment and output_ind < (len(self.output_files) - 1):
            next_time = self.output_times[output_ind + 1]
            next_file = self.output_files[output_ind + 1]
            diff = int((next_time - output_time).astype("timedelta64[h]").astype("int64").item())
            if diff <= self.lead_time:
                frac = self.rng.random()
                next_target = self.load_wtcm_data(next_file)
                target = {
                    name: frac * target[name] + (1.0 - frac) * next_target[name] for name in target
                }
                x["lead_time"] = x["lead_time"] + torch.tensor((1.0 - frac) * diff)

        x, target = _transform_data(x, target, roll, flip_v=flip_v, flip_h=flip_h, scale=scale)
        return x, target


class AutoregressiveWTCMForecastDataset(DirectWTCMForecastDataset):
    """
    A PyTorch Dataset for loading WTCM data for autoregressive forecasts.
    """
    def __init__(
            self,
            training_data_path: Union[Path, str],
            scaling_factors: Union[Path, str],
            input_time: int = 3,
            lead_time: Optional[int] = None,
            accumulation_period: int = 3,
            max_steps: int = 24,
            climate: bool = True,
            sampling_rate: float = 1.0,
            center_meridionally: bool = True,
            validation: bool = False,
            local_data: Optional[Path] = None,
            augment: bool = False,
            source: str = "merra2"
    ):
        """
        Args:
            training_data_path: The directory containing the dynamic input data.
            scaling_factors: Directory containing the scaling factors for the Prithvi-WxC model.
            input_time: The time difference between input samples.
            lead_time: The rollout timestep.
            accumulation_period: The precipitation accumulation period.
            max_steps: The maximum number of timesteps to forecast precipitation.
            climate: Whether to include climatology data in the input.
            sampling_rate: Sub- or super-sample dataset.
            reference_data: Name of the reference data source.
            center_meridionally: If True, will use mid-point averaging to reduce the latitude dimension
                of the input data by one. If False, will use negative paddgin.
            validation: Flat indicating whether the dataset is used to load validation or training data.
            local_data: An optional path pointing to a location to which to copy the training data. This should
                typically be node-local memory that can be accessed rapidly.
            augment: Whether or not to augment the input data using random zonal rolls and meridional flips.
            source: Name of the input dataset.
        """
        super().__init__(
            training_data_path=training_data_path,
            input_time=input_time,
            lead_time=lead_time,
            accumulation_period=accumulation_period,
            max_steps=max_steps,
            climate=climate,
            sampling_rate=sampling_rate,
            center_meridionally=center_meridionally,
            validation=validation,
            local_data=local_data,
            augment=augment,
            source=source
        )
        scaling_factors = Path(scaling_factors)
        if not scaling_factors.exists():
            raise ValueError(
                "scaling_factors must point to an existing directory and contain the PrithviWxC scaling factors."
            )
        self.output_sig = output_scalers(
            SURFACE_VARS,
            VERTICAL_VARS,
            LEVELS,
            str(scaling_factors / "anomaly_variance_surface.nc"),
            str(scaling_factors / "anomaly_variance_vertical.nc"),
        )[..., None, None] ** 0.5

    def __len__(self) -> int:
        """The number of samples in the dataset."""
        return trunc(len(self.input_indices) * self.sampling_rate)


    def load_data(
            self,
            index: int,
            roll: int,
            flip_v: bool,
            flip_h: bool,
            scale: float = 1.0
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Load training data for a specific sample.

        Args:
            index: The index of the sample.
            roll: Roll data by that many pixels in zonal direction.
            flip_v: bool,
            flip_h: bool,
            scale: float = 1.0

        Return:
            A tuple containing the input and target data.
        """
        input_time = self.input_steps[0]
        input_indices = self.input_indices[index]
        if self.validation:
            step_ind = int(index * self.sampling_rate) % len(self.input_steps)
            input_indices = [input_indices[step_ind], input_indices[-1]]
            input_time = self.input_steps[step_ind]
        else:
            if 1 < len(self.input_steps):
                step_ind = self.rng.integers(len(self.input_steps))
                input_indices = [input_indices[step_ind], input_indices[-1]]
                input_time = self.input_steps[step_ind]

        input_files = [self.input_files[ind] for ind in input_indices]
        input_times = [self.input_times[ind] for ind in input_indices]

        dynamic_in = [load_dynamic_input(self.training_data_path / path) for path in input_files]

        static_times = input_times[-1] + np.arange(0, self.max_steps) * np.timedelta64(self.lead_time, "h")
        static_in = [
            torch.tensor(load_static_input(static_time, self.data_path)) for static_time in static_times
        ]

        if self.center_meridionally:
            transform = lambda tnsr: 0.5 * (tnsr[..., 1:, :] + tnsr[..., :-1, :])
        else:
            transform = partial(nn.functional.pad, pad=(0, 0, 0, -1), mode="constant", value=0)

        x = {
            "x": transform(torch.stack(dynamic_in, 0)),
            "static": transform(torch.stack(static_in, 0)),
            "input_time": torch.tensor(input_time).to(dtype=torch.float32),
            "lead_time": torch.tensor(self.lead_time).to(dtype=torch.float32),
        }

        inds = self.output_indices[index]

        targets = {}
        climates = []
        ys = []

        available_times = [
            self.output_times[out_ind] for out_ind in self.output_indices[index]
            if 0 <= out_ind
        ]
        output_indices = [
            out_ind for out_ind in self.output_indices[index] if 0 <= out_ind
        ]

        output_time = input_times[-1]

        for step in range(1, self.max_steps + 1):

            output_time += np.timedelta64(self.lead_time, "h")

            if self.climate:
                climates.append(torch.tensor(load_and_interp_climatology(output_time, self.data_path)))

            if output_time in available_times:
                output_ind = available_times.index(output_time)
                output_file = self.output_files[output_indices[output_ind]]
                targets_s = self.load_wtcm_data(self.training_data_path / output_file)
                targets_s = {name: transform(tnsr) for name, tnsr in targets_s.items()}
            else:
                targets_s.append({
                    "u10": torch.nan * torch.zeros((1, 360, 576)),
                    "v10": torch.nan * torch.zeros((1, 360, 576))
                })

            if output_time in self.input_times:
                ind = np.searchsorted(self.input_times, output_time)
                y = load_dynamic_input(self.training_data_path / self.input_files[ind])
                if self.climate:
                    y = (y - climates[-1])
                y = y / self.output_sig
                targets_s["y"] = transform(y)
            else:
                targets_s["y"] = torch.nan * torch.zeros_like(climates[-1])

            for key, tnsr in targets_s.items():
                targets.setdefault(key, []).append(tnsr)

        if 0 < len(climates):
            x["climate"] = transform(torch.stack(climates, 0))

        #x, targets = _transform_data(x, targets, roll, flip)
        return x, targets
