"""
prithvi_precip.datasets.severe_weather
======================================

Provides datasets for training severe weather forecasts.
"""

from datetime import datetime
from functools import cache, cached_property, partial
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
from .precipitation import DirectPrecipForecastDataset, _transform_data
from ..utils import (
    find_input_files,
    load_and_interp_climatology,
    load_dynamic_input,
    load_static_input,
    to_datetime64
)


LOGGER = logging.getLogger(__name__)


@cache
def get_severe_weather_mask() -> xr.Dataset:
    """
    A mask identifying the valid domain of the training data.
    """
    return xr.load_dataset(Path(__file__).parent / "severe_weather_mask.nc").mask.data


class DirectSevereWeatherForecastDataset(DirectPrecipForecastDataset):
    """
    A PyTorch Dataset for loading precipitation and severe weahter data for forecasts without unrolling.
    """
    def __init__(
            self,
            training_data_path: Union[Path, str],
            input_time: Union[int, List[int]] = 3,
            lead_time: int = 3,
            accumulation_period: int = 24,
            max_steps: int = 24,
            climate: bool = True,
            sampling_rate: float = 1.0,
            reference_data: str = "severe_weather",
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
            reference_data: Name of the reference data source.
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
        self.reference_data = reference_data
        self.center_meridionally = center_meridionally
        self.validation = validation
        self.augment = augment

        self.local_data = None
        if local_data is not None:
            self.local_data = Path(local_data)
        self.source = source

        self.input_times, self.input_files = find_input_files(self.training_data_path, source=source)
        self.output_times, self.output_files = self.find_precip_files(
            self.training_data_path,
            reference_data=self.reference_data,
            accumulation_period=self.accumulation_period
        )

        self._pos_sig = None
        self.input_indices, self.output_indices = self.calculate_valid_samples()
        self.rng = np.random.default_rng(seed=42)

        if self.local_data is not None:
            self.split_and_copy_files()

    @cached_property
    def severe_weather_mask(self) -> xr.Dataset:
        """
        A mask identifying the valid domain of the training data.
        """
        return xr.load_dataset(Path(__file__).parent / "severe_weather_mask.nc").mask.data


    def split_and_copy_files(self) -> None:
        """
        Shards data across nodes and copies them to the location pointed to by self.local_data.
        """
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        LOGGER.info("Splitting data: %s %s %s", rank, local_rank, world_size)

        n_samples = len(self.input_indices)
        n_samples_local = n_samples // world_size
        start = rank * n_samples_local
        end = start + n_samples_local

        local_input_indices = self.input_indices[start:end]
        local_output_indices = self.output_indices[start:end]

        # Create directory for local data
        base_folder = self.training_data_path.parent.name

        if self.validation:
            training_local = self.local_data / base_folder / f"validation_data_{local_rank:02}"
        else:
            training_local = self.local_data / base_folder / f"training_data_{local_rank:02}"

        training_local.mkdir(exist_ok=True, parents=True)

        # Copy input and output samples.
        LOGGER.info(
            "Copying %s training files to local directory %s.",
            len(local_input_indices),
            training_local
        )
        input_files = []
        input_times = []
        for inds in local_input_indices:
            input_files += list(self.input_files[inds])
        input_files = set(input_files)

        output_files = []
        for inds in local_output_indices:
            out_inds = [ind for ind in inds if 0 <= ind]
            output_files += list(self.output_files[out_inds])
        output_files = set(output_files)

        all_files = input_files.union(output_files)

        for path in all_files:
            rel_path = Path(path)
            target_path = training_local / rel_path
            if not target_path.exists():
                target_path.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy2(self.training_data_path / rel_path, target_path)

        if local_rank == 0 and not self.validation:
            LOGGER.info(
                "Copying static files to temporary directory."
            )
            climatology = training_local.parent / "climatology"
            if not climatology.exists():
                shutil.copytree(self.training_data_path.parent / "climatology", climatology, dirs_exist_ok=True)
            static_data = training_local.parent / "static"
            if not static_data.exists():
                shutil.copytree(self.training_data_path.parent / "static", static_data, dirs_exist_ok=True)
        else:
            static_data = training_local.parent / "static"
            while not static_data.exists():
                sleep(0.1)

        rank = int(os.environ.get("RANK", 0))

        self.training_data_path = training_local
        self.data_path = self.training_data_path.parent
        self.input_times, self.input_files = find_input_files(self.training_data_path, source="merra2")
        self.output_times, self.output_files = self.find_precip_files(
            self.training_data_path,
            reference_data=self.reference_data,
            accumulation_period=self.accumulation_period
        )

        # Filter input and output times to ensure all processes have the same number of samples.
        input_times_new = []
        input_files_new = []
        for input_time, input_file in zip(self.input_times, self.input_files):
            if input_file in input_files:
                input_times_new.append(input_time)
                input_files_new.append(input_file)
        self.input_times = input_times_new
        self.input_files = input_files_new

        output_times_new = []
        output_files_new = []
        for output_time, output_file in zip(self.output_times, self.output_files):
            if output_file in output_files:
                output_times_new.append(output_time)
                output_files_new.append(output_file)
        self.output_times = output_times_new
        self.output_files = output_files_new

        self.input_indices, self.output_indices = self.calculate_valid_samples()
        assert len(self.input_indices) == n_samples_local

    def __len__(self) -> int:
        """The number of samples in the dataset."""
        return trunc(len(self.input_indices) * self.sampling_rate)

    def load_severe_weather_data(self, output_file: str) -> Dict[str, torch.tensor]:
        """
        Load severe weather data from target-data file.

        Args:
            output_file: The relative path of the output file from which to load the data.

        Return:
            A dictionary containing the target names and corresponding tensors.
        """
        with xr.load_dataset(self.training_data_path / output_file) as data:
            LOGGER.debug("Loading severe weather data from %s.", output_file)
            data = data.compute()
            tornado = torch.tensor(data.tornado.data)
            hail = torch.tensor(data.hail.data)
            wind = torch.tensor(data.wind.data)

            mask = ~torch.tensor(self.severe_weather_mask)
            tornado[mask] = torch.nan
            hail[mask] = torch.nan
            wind[mask] = torch.nan

            severe = tornado + wind + hail
            target = {
                "tornado": torch.clip(tornado, 0.0, 1.0),
                "tornado_weights": torch.maximum(tornado, torch.tensor(1.0)),
                "hail": torch.clip(hail, 0.0, 1.0),
                "hail_weights": torch.maximum(hail, torch.tensor(1.0)),
                "wind": torch.clip(wind, 0.0, 1.0),
                "wind_weights": torch.maximum(wind, torch.tensor(1.0)),
                "severe": torch.clip(severe, 0.0, 1.0),
                "severe_weights": torch.maximum(severe, torch.tensor(1.0)),
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
            noise = 0.05 * torch.tensor(self.rng.normal(size=(2, d_x.shape[0], 1, 1)).astype(np.float32))
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
                noise = 0.05 * torch.tensor(self.rng.normal(size=(d_x.shape[0], 1, 1)).astype(np.float32))
                x["climate"] += noise * d_x

        target = self.load_severe_weather_data(output_file)

        if self.augment and output_ind < (len(self.output_files) - 1):
            next_time = self.output_times[output_ind + 1]
            next_file = self.output_files[output_ind + 1]
            diff = int((next_time - output_time).astype("timedelta64[h]").astype("int64").item())
            if diff <= self.lead_time:
                frac = self.rng.random()
                next_target = self.load_severe_weather_data(next_file)
                target = {
                    name: frac * target[name] + (1.0 - frac) * next_target[name] for name in target
                }
                x["lead_time"] = x["lead_time"] + torch.tensor((1.0 - frac) * diff)

        x, target = _transform_data(x, target, roll, flip_v=flip_v, flip_h=flip_h, scale=scale)
        return x, target
