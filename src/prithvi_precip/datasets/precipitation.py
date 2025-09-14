"""
prithvi_precip.datasets.precipitation
=====================================

Provides datasets for loading MERRA-2 data and precipitation fields as targets.
"""
from datetime import datetime
from functools import partial
import logging
from math import trunc
import os
from pathlib import Path
import shutil
from time import sleep
from typing import Optional, Union, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import xarray as xr
from PrithviWxC.dataloaders.merra2 import output_scalers

from .merra2 import MERRAInputData
from ..data.merra2 import SURFACE_VARS, VERTICAL_VARS, LEVELS
from ..utils import (
    find_input_files,
    load_climatology,
    load_dynamic_input,
    load_static_input,
    to_datetime64
)


LOGGER = logging.getLogger(__name__)


class DirectPrecipForecastDataset(MERRAInputData):
    """
    A PyTorch Dataset for loading precipitation forecast training data for direct forecasts without
    unrolling.
    """
    def __init__(
            self,
            training_data_path: Union[Path, str],
            input_time: int = 3,
            accumulation_period: int = 3,
            max_steps: int = 24,
            climate: bool = True,
            sampling_rate: float = 1.0,
            reference_data: str = "imerg",
            center_meridionally: bool = True,
            validation: bool = False,
            local_data: Optional[Path] = None,
            weighted_sampling: bool = False,
            source: str = "merra2"
    ):
        """
        Args:
            training_data_path: The directory containing the dynamic input data.
            input_time: The time difference between input samples.
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
            weighted_sampling: Whether or not to weigh longer-range forecasts inverserly to the lead time.
            source: The source of the input data: 'merra2' or 'geos'
        """
        self.training_data_path = Path(training_data_path)
        self.data_path = self.training_data_path.parent
        self.input_time = input_time
        self.accumulation_period = accumulation_period
        self.max_steps = max_steps
        self.climate = climate
        self.sampling_rate = sampling_rate
        self.reference_data = reference_data
        self.center_meridionally = center_meridionally
        self.validation = validation
        self.local_data = None
        if local_data is not None:
            self.local_data = Path(local_data)
        self.weighted_sampling = weighted_sampling
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
        self.input_times, self.input_files = self.find_input_files(self.training_data_path, source="merra2")
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

    def find_precip_files(
            self,
            training_data_path: Path,
            reference_data: str,
            accumulation_period: int
    ) -> np.ndarray:
        """
        Find precip files for training.
        """
        times = []
        files = []

        prefix = f"{reference_data.lower()}"
        pattern = f"{self.reference_data.lower()}_{accumulation_period}/**/{prefix}*.nc"
        date_pattern = f"{reference_data.lower()}_%Y%m%d%H%M%S.nc"

        for path in sorted(list(training_data_path.glob(pattern))):
            try:
                date = datetime.strptime(path.name, date_pattern)
                date64 = to_datetime64(date)
                files.append(str(path.relative_to(training_data_path)))
                times.append(date64)
            except ValueError:
                continue

        times = np.array(times)
        files = np.array(files)
        return times, files


    def worker_init_fn(self, w_id: int) -> None:
        """
        Seeds the dataset loader's random number generator.
        """
        seed = int.from_bytes(os.urandom(4), "big") + w_id
        self.rng = np.random.default_rng(seed)


    def calculate_valid_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        A tuple of index arrays containing the indices of input- and output files for all training data
        samples satifying the requested input and lead time combination.

        Return: A tuple '(input_indices, output_indices)' with `input_indices` of shape
            '(n_samples, n_input_times)' containing the indices of all the input files for each data
            samples. Similarly, 'output_indices' is a numpy.ndarray of shape '(n_samples, n_lead_times)'
            containing the corresponding file indices to load for the output data.
        """
        input_indices = []
        output_indices = []
        for ind, sample_time in enumerate(self.input_times):
            input_times = [sample_time + np.timedelta64(t_i * self.input_time, "h") for t_i in [-1, 0]]

            output_times = [
                sample_time + np.timedelta64(t_i * self.input_time, "h") for t_i in np.arange(1, self.max_steps + 1)
            ]
            output_times = [t_o for t_o in output_times if t_o in self.output_times]
            valid = all([t_i in self.input_times for t_i in input_times])
            if valid and len(output_times) > 0:

                prev_ind = np.searchsorted(self.input_times, input_times[0])
                input_indices.append([prev_ind, ind])

                output_inds = []
                for output_time in output_times:
                    output_ind = np.searchsorted(self.output_times, output_time)
                    output_inds.append(output_ind)
                output_indices.append(output_inds + [-1] * (self.max_steps - len(output_inds)))

        return np.array(input_indices), np.array(output_indices)

    def __len__(self) -> int:
        """The number of samples in the dataset."""
        return trunc(len(self.input_indices) * self.sampling_rate)

    def __getitem__(self, ind: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a single data point from the dataset.
        """
        lower = trunc(ind / self.sampling_rate)
        upper = min(trunc((ind + 1) / self.sampling_rate), len(self.input_indices) - 1)
        if lower < upper:
            ind = self.rng.integers(lower, upper)
        else:
            ind = lower

        try:
            input_files = [self.input_files[ind] for ind in self.input_indices[ind]]
            input_times = [self.input_times[ind] for ind in self.input_indices[ind]]
            dynamic_in = [load_dynamic_input(self.training_data_path / path) for path in input_files]

            static_time = input_times[-1]
            static_in = torch.tensor(load_static_input(static_time, self.data_path))

            input_time = self.input_time

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

            inds = self.output_indices[ind]
            inds = inds[0 <= inds]

            deltas = np.array([(self.output_times[output_ind] - input_times[-1]) for output_ind in inds])
            if self.weighted_sampling:
                deltas = deltas.astype("datetime64[s]")
                delta_min = deltas.min()
                delta_max = deltas.max()
                weights = 0.5 + 0.5 * (delta - delta_min) / (delta_max - delta_min)
                weights = weights.astype(np.float32)
            else:
                weights = np.ones_like(deltas).astype(np.float32)
            weights /= weights.sum()

            output_ind = self.rng.choice(inds, p=weights)
            output_file = self.output_files[output_ind]
            output_time = self.output_times[output_ind]

            lead_time = (output_time - max(input_times)).astype("timedelta64[h]").astype(np.float32)
            x["lead_time"] = torch.tensor(lead_time).to(dtype=torch.float32)

            if self.climate:
                climate = load_climatology(output_time, self.data_path)
                x["climate"] = transform(torch.tensor(climate))

            with xr.load_dataset(self.training_data_path / output_file) as data:
                LOGGER.debug("Loading precip data from %s.", output_file)
                precip = torch.tensor(data.surface_precip.data.astype(np.float32))
                if self.reference_data.startswith("era5"):
                    precip = 1e3 * precip

            coords = x["static"][:2]

            return x, precip

        except Exception as exc:
            raise exc
            LOGGER.exception(
                "Encountered an error when load training sample %s. Falling back to another "
                " randomly-chosen sample.",
                ind
            )
            new_ind = np.random.randint(0, len(self))
            return self[new_ind]


    def get_forecast_input(self, init_time: np.datetime64, n_steps: int):

        input_times = [init_time - np.timedelta64(self.input_time, "h"), init_time]

        input_ind = np.searchsorted(self.input_times, input_times[0])
        input_time = self.input_times[input_ind]
        if input_time != input_times[0]:
            raise ValueError(
                "Missing required input for time %s.",
                input_times[0]
            )
        dynamic_in = [load_dynamic_input(self.training_data_path / self.input_files[input_ind])]
        input_ind = np.searchsorted(self.input_times, input_times[1])
        input_time = self.input_times[input_ind]
        if input_time != input_times[1]:
            raise ValueError(
                "Missing required input for time %s.",
                input_times[1]
            )
        dynamic_in += [load_dynamic_input(self.trainign_data_path / self.input_files[input_ind])]


        static_time = input_times[-1]
        static_in = torch.tensor(load_static_input(static_time, self.data_path))

        if self.center_meridionally:
            transform = lambda tnsr: 0.5 * (tnsr[..., 1:, :] + tnsr[..., :-1, :])
        else:
            transform = partial(nn.functional.pad, pad=((0, 0, 0, -1)))

        inpt = {
            "x": (transform(torch.stack(dynamic_in, 0))[None]).repeat_interleave(n_steps, dim=0),
            "static": (transform(static_in)[None]).repeat_interleave(n_steps, dim=0),
            "input_time": ((torch.tensor(self.input_time).to(dtype=torch.float32))[None]).repeat_interleave(n_steps, dim=0),
        }

        output_times = init_time + np.timedelta64(self.input_time, "h") * np.arange(1, n_steps + 1)
        climates = []
        if self.climate:
            for output_time in output_time:
                climates.append(transform(load_climatology(output_time, self.data_path)))

        climates = torch.stack(climates)
        x["climate"] = climates

        return inpt


class AutoregressivePrecipForecastDataset(DirectPrecipForecastDataset):
    """
    A PyTorch Dataset for loading precipitation forecast training data for autoregressive forecasts.
    """
    def __init__(
            self,
            training_data_path: Union[Path, str],
            scaling_factors: Union[Path, str],
            input_time: int = 3,
            accumulation_period: int = 3,
            max_steps: int = 24,
            climate: bool = True,
            sampling_rate: float = 1.0,
            reference_data: str = "imerg",
            center_meridionally: bool = True,
            validation: bool = False,
            local_data: Optional[Path] = None,
            weighted_sampling: bool = False,
            source: str = "merra2"
    ):
        """
        Args:
            training_data_path: The directory containing the dynamic input data.
            scaling_factors: Directory containing the scaling factors for the Prithvi-WxC model.
            input_time: The time difference between input samples.
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
            weighted_sampling: Whether or not to weigh longer-range forecasts inverserly to the lead time.
        """
        super().__init__(
            training_data_path=training_data_path,
            input_time=input_time,
            accumulation_period=accumulation_period,
            max_steps=max_steps,
            climate=climate,
            sampling_rate=sampling_rate,
            reference_data=reference_data,
            center_meridionally=center_meridionally,
            validation=validation,
            local_data=local_data,
            weighted_sampling=weighted_sampling,
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

    def __getitem__(self, sample_ind: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a single data point from the dataset.
        """
        lower = trunc(sample_ind / self.sampling_rate)
        upper = min(trunc((sample_ind + 1) / self.sampling_rate), len(self.input_indices) - 1)
        if lower < upper:
            sample_ind = self.rng.integers(lower, upper)
        else:
            sample_ind = lower

        try:
            input_files = [self.input_files[ind_in] for ind_in in self.input_indices[sample_ind]]
            input_times = [self.input_times[ind_in] for ind_in in self.input_indices[sample_ind]]
            dynamic_in = [load_dynamic_input(self.training_data_path / path) for path in input_files]

            static_times = input_times[-1] + np.arange(0, self.max_steps) * np.timedelta64(self.input_time, "h")
            static_in = [
                torch.tensor(load_static_input(static_time, self.data_path)) for static_time in static_times
            ]

            input_time = self.input_time

            if self.center_meridionally:
                transform = lambda tnsr: 0.5 * (tnsr[..., 1:, :] + tnsr[..., :-1, :])
            else:
                transform = partial(nn.functional.pad, pad=(0, 0, 0, -1), mode="constant", value=0)

            x = {
                "x": transform(torch.stack(dynamic_in, 0)),
                "static": transform(torch.stack(static_in, 0)),
                "input_time": torch.tensor(input_time).to(dtype=torch.float32),
                "lead_time": torch.tensor(input_time).to(dtype=torch.float32),
            }

            inds = self.output_indices[sample_ind]

            precip = []
            climates = []
            ys = []

            available_times = [
                self.output_times[out_ind] for out_ind in self.output_indices[sample_ind]
                if 0 <= out_ind
            ]
            output_indices = [
                out_ind for out_ind in self.output_indices[sample_ind] if 0 <= out_ind
            ]

            output_time = input_times[-1]

            for step in range(1, self.max_steps + 1):

                output_time += np.timedelta64(self.input_time, "h")

                if self.climate:
                    climates.append(torch.tensor(load_climatology(output_time, self.data_path)))

                if output_time in available_times:
                    output_ind = available_times.index(output_time)
                    output_file = self.output_files[output_indices[output_ind]]
                    with xr.load_dataset(self.training_data_path / output_file) as data:
                        LOGGER.debug("Loading precip data from %s.", output_file)
                        precip_s = torch.tensor(data.surface_precip.data.astype(np.float32))
                        if self.reference_data.startswith("era5"):
                            precip_s = 1e3 * precip_s
                        precip.append(precip_s)
                else:
                    precip.append(torch.nan * torch.zeros((1, 360, 576)))


                if output_time in self.input_times:
                    ind = np.searchsorted(self.input_times, output_time)
                    y = load_dynamic_input(self.training_data_path / self.input_files[ind])
                    if self.climate:
                        y = (y - climates[-1])
                    y = y / self.output_sig
                    ys.append(transform(y))
                else:
                    ys.append(torch.nan * torch.zeros_like(climates[-1]))

            if 0 < len(climates):
                x["climate"] = transform(torch.stack(climates, 0))

            targets = {
                "surface_precip": precip,
                "y": ys
            }
            return x, targets

        except Exception as exc:
            raise exc
            LOGGER.exception(
                "Encountered an error when load training sample %s. Falling back to another "
                " randomly-chosen sample.",
                sample_ind
            )
            new_ind = np.random.randint(0, len(self))
            return self[new_ind]
