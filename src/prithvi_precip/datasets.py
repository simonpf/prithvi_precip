"""
prithvi_precip.datasets
=======================

Provides datasets to load training data for the Prithvi-WxC model.
"""
from datetime import datetime
from functools import cache, cached_property, partial
import logging
from math import trunc
import os
from pathlib import Path
import re
import shutil
from time import sleep
from typing import Dict, Iterator, List, Optional, Tuple, Union

from filelock import FileLock
import numpy as np
from prithvi_precip.utils import load_static_input, load_climatology
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
import xarray as xr

from prithvi_precip.data.merra2 import (
    SURFACE_VARS,
    VERTICAL_VARS,
    STATIC_SURFACE_VARS,
)
from .utils import to_datetime, to_datetime64


LOGGER = logging.getLogger(__name__)


def get_position_signal(lons: np.ndarray, lats:np.ndarray, kind: str) -> np.ndarray:
    """
    Calculate the position encoding.

    Args:
        lons: An array containing the longitude coordinates.
        lats: An array containing the latitude coordiantes.
        kind: A string defining the kind of the encoding. Currely supported are:
            - 'absolute': Returns the sine of the latitude coordinates and the cosine and sine
               of the longitude coordaintes stacked along the first dimensions
            - anything else: Simply returns the latitudes and longitudes in degree stacked along
              the first dimenion.
    """
    lons = lons.astype(np.float32)
    lats = lats.astype(np.float32)
    lons ,lats = np.meshgrid(lons, lats, indexing="xy")
    if kind == "absolute":
        lats_rad = np.deg2rad(lats_rad)
        lons_rad = np.deg2rad(lons_rad)
        static = np.stack([
            np.sin(lats_rad),
            np.cos(lons_rad),
            np.sin(lons_rad)
        ])
    return np.stack([lats, lons], axis=0).astype(np.float32)


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
            observation_layers: Optional[int] = None,
            center_meridionally: bool = True
    ):

        """
        Args:
            training_data_path (str): Path pointing to the directory containing the dynamic MERRA2
                input data in year/month/day folders.
            input_time: The input time.
            climatology: Whether or not to include climatology data in the input.
            observation_layers:
            center_meridionally: Whether to center input grids meridionally instad of removing the last row
                 (which is the default for the original Prithvi-WxC)
        """
        self.training_data_path = Path(training_data_path)
        self.data_path = self.training_data_path.parent
        self.times, self.input_files = self.find_merra_files(self.training_data_path)
        self.climatology = climatology

        self.input_time = input_time
        self.time_step = lead_times[0]
        self.lead_times = lead_times

        self.input_indices, self.output_indices = self.calculate_valid_samples()
        self._pos_sig = None
        self.center_meridionally = center_meridionally

    def find_merra_files(self, training_data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gather all available MERRA2 files paths and extract available times.

        Args:
            training_data_path: Path object pointing to the directory containing the training data.

        Return:
            A tuple containing arrays of available inputs times and corresponding file
            paths.
        """
        times = []
        files = []
        pattern = re.compile(r"merra_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\.nc")

        for path in sorted(list(training_data_path.glob("dynamic/**/merra2_*.nc"))):
            try:
                date = datetime.strptime(path.name, "merra2_%Y%m%d%H%M%S.nc")
                date64 = to_datetime64(date)

                files.append(str(path.relative_to(training_data_path)))
                times.append(date64)
            except ValueError:
                continue

        times = np.array(times)
        files = np.array(files)
        return times, files

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

    def load_dynamic_data(self, path: Path, slcs: Optional[Dict[str, slice]] = None) -> torch.Tensor:
        """
        Load all dynamic data from a given input file and return the data.

        Args:
            path: A path object pointing to the file to load.

        Return:
            A torch.Tensor containing all dynamic data for the given input file in the shape
            [var + levels (channels), lat, lon].
        """
        LOGGER.debug(
            "Loading dynamic input from file %s.",
            path
        )
        all_data = []
        if slcs is None:
            slcs = {}
        with xr.open_dataset(self.training_data_path / path) as data:
            for var in SURFACE_VARS:
                all_data.append(data[var].__getitem__(slcs).data[None].astype(np.float32))
            for var in VERTICAL_VARS:
                all_data.append(data[var].__getitem__(slcs).astype(np.float32))
        all_data = torch.tensor(np.concatenate(all_data, axis=0))
        return all_data

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

        dynamic_in = [self.load_dynamic_data(path) for path in input_files]
        static_in = torch.tensor(load_static_input(input_times[-1], self.data_path))

        input_time = (input_times[1] - input_times[0]).astype("timedelta64[h]").astype(np.float32)
        lead_time = (output_times[0] - input_times[1]).astype("timedelta64[h]").astype(np.float32)

        dynamic_out = [self.load_dynamic_data(path) for path in output_files]
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
        input_times = [init_time + np.timedelta64(t_i * self.time_step, "h") for t_i in [-1, 0]]
        for input_time in input_times:
            if input_time not in self.times:
                raise ValueError(
                    "Required input data for t=%s not available.",
                    input_time
                )

        dynamic_in = []
        for input_time in input_times:
            ind = np.searchsorted(self.times, input_time)
            dynamic_in.append(self.load_dynamic_data(self.input_files[ind]))

        static_time = input_times[-1]
        static_in = load_static_input(static_time, self.data_path)

        if self.center_meridionally:
            transform = lambda tnsr: 0.5 * (tnsr[..., 1:, :] + tnsr[..., :-1, :])
        else:
            transform = partial(nn.functional.pad, pad=((0, 0, 0, -1)))

        dynamic_in = transform(torch.stack(dynamic_in, 0))[None].repeat(n_steps, 1, 1, 1, 1)
        static_in = transform(torch.tensor(static_in))[None].repeat(n_steps, 1, 1, 1)
        input_time = self.input_times[0] * torch.ones(n_steps)
        lead_time = self.time_step * torch.arange(1, n_steps + 1).to(dtype=torch.float32)

        x = {
            "x": dynamic_in,
            "static": static_in,
            "lead_time": lead_time,
            "input_time": input_time,
        }

        if self.climate:
            output_times = [init_time + step * np.timedelta64(self.time_step, "h") for step in range(1, n_steps + 1)]
            climate = [torch.tensor(load_climatology(time, self.data_path)) for time in output_times]
            climate = transform(torch.stack(torch.tensor(climate)))
            x["climate"] = climate

        #if self.obs_loader is not None:
        #    obs = []
        #    meta = []
        #    for time_ind, time in enumerate(input_times):
        #        obs_t, meta_t = self.obs_loader.load_observations(time, offset=len(input_times) - time_ind - 1)
        #        obs.append(obs_t)
        #        meta.append(meta_t)
        #    obs = torch.stack(obs, 0)
        #    obs_mask = obs < -2.9
        #    obs = torch.nan_to_num(obs, nan=-3.0)
        #    meta = torch.stack(meta, 0)

        #    x["obs"] = obs[None].repeat_interleave(n_steps, 0)
        #    x["obs_mask"] = obs_mask[None].repeat_interleave(n_steps, 0)
        #    x["obs_meta"] = meta[None].repeat_interleave(n_steps, 0)

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
        x = self.get_forecast_input(init_time, n_steps=n_steps)
        batch_start = 0
        n_samples = x["x"].shape[0]
        while batch_start < n_samples:
            batch_end = batch_start + batch_size
            yield {name: tnsr[batch_start:batch_end] for name, tnsr in x.items()}
            batch_start = batch_end


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
            weighted_sampling: bool = False
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

        self.input_times, self.input_files = self.find_merra_files(self.training_data_path)
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
        for ind in local_input_indices:
            input_files += list(self.input_files[ind])
        input_files = set(input_files)

        output_files = []
        for ind in local_output_indices:
            output_files += list(self.output_files[ind])
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
        self.input_times, self.input_files = self.find_merra_files(self.training_data_path)
        self.output_times, self.output_files = self.find_precip_files(
            self.training_data_path,
            reference_data=self.reference_data,
            accumulation_period=self.accumulation_period
        )
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
        pattern = f"**/{prefix}*.nc"
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

    def __len__(self):
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
            dynamic_in = [self.load_dynamic_data(path) for path in input_files]

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


class ObservationLoader(Dataset):
    """
    PyTorch dataset for loading satellite observations as input for
    PrithviWxC precipitation forecasts.
    """
    def __init__(
            self,
            observation_path: Path,
            observation_layers: int = 32,
            n_tiles: Tuple[int, int] = (12, 18),
            tile_size: Tuple[int, int] = (30, 32)
    ):
        """
        Args:
            observation_path: Path containing the observations.
            observation_layers: The number of observation layers to load.
            n_tiles: A tuple specifying containing the number of meridional
                and zonal tiles, respectively.
            tile_size: The size of the observation tiles.
        """
        self.observation_path = Path(observation_path)
        self.observation_layers = observation_layers
        self.n_tiles = n_tiles
        self.tile_size = tile_size
        self.rng = np.random.default_rng(seed=42)
        self.time_step = 3
        self.freq_min = 1.0
        self.freq_max = 30e3
        self.file_regexp = None

    def worker_init_fn(self, w_id: int) -> None:
        """
        Seeds the dataset loader's random number generator.
        """
        #tracemalloc.start()
        seed = int.from_bytes(os.urandom(4), "big") + w_id
        self.rng = np.random.default_rng(seed)

    @cached_property
    def start_time(self):
        """
        The first time for which observations are available.
        """
        year_folders = sorted([path for path in self.observation_path.iterdir() if path.is_dir()])
        month_folders = sorted([path for path in year_folders[0].iterdir() if path.is_dir()])
        day_folders = sorted([path for path in month_folders[0].iterdir() if path.is_dir()])
        year = int(year_folders[0].name)
        month = int(month_folders[0].name)
        day = int(day_folders[0].name)
        return np.datetime64(f"{year}-{month:02}-{day:02}")

    @cached_property
    def end_time(self):
        """
        The last time for which observations are available.
        """
        year_folders = sorted([path for path in self.observation_path.iterdir() if path.is_dir()])
        month_folders = sorted([path for path in year_folders[-1].iterdir() if path.is_dir()])
        day_folders = sorted([path for path in month_folders[-1].iterdir() if path.is_dir()])
        year = int(year_folders[-1].name)
        month = int(month_folders[-1].name)
        day = int(day_folders[-1].name)
        return np.datetime64(f"{year}-{month:02}-{day:02}")

    @cached_property
    def stats_data(self):
        """
        The observation bulk statistics from the stats.nc file in the top-level directory.
        """
        stats_file = self.observation_path / "stats.nc"
        lock_file = FileLock(stats_file.with_suffix(".lock"))
        with lock_file:
            stats = xr.load_dataset(stats_file)
        return stats

    def get_observation_mask(self) -> xr.Dataset:
        """
        Calculate an xarray.Dataset indicating the availability of different observations.
        """
        stats = self.stats_data
        all_vars = stats.attrs["variables"].split(",")
        n_vars = len(all_vars)

        start_time = self.start_time
        end_time = self.end_time

        times = np.arange(start_time, end_time + np.timedelta64(3, "h"), np.timedelta64(3, "h"))
        obs_times = []
        obs_tiles = []
        obs_bins = np.arange(n_vars + 1) - 0.5

        for time in tqdm(times):
            date = to_datetime(time)
            path = self.observation_path / date.strftime("%Y/%m/%d/obs_%Y%m%d%H%M%S.nc")
            if path.exists():
                try:
                    with xr.open_dataset(path) as obs_data:
                        obs_times.append(time)
                        obs_tiles.append(np.zeros((obs_bins.size - 1,) + self.n_tiles))

                        for meridional_index in range(self.n_tiles[0]):
                            for zonal_index in range(self.n_tiles[1]):
                                tilename = f"obs_id_{meridional_index:02}_{zonal_index:02}"
                                if tilename in obs_data:
                                    obs_ids = obs_data[tilename].data
                                    obs_tiles[-1][:, meridional_index, zonal_index] += np.histogram(obs_ids, bins=obs_bins)[0]
                except Exception:
                    LOGGER.exception(
                        "Encountered an error when observation file %s.",
                        path
                    )

        platforms = []
        sensors = []
        for var in all_vars:
            platform_sensor = var.split("_")[0]
            parts = platform_sensor.split(".")
            platform = parts[0]
            sensor = ""
            if len(parts) > 1:
                sensor = parts[1]
            platforms.append(platform)
            sensors.append(sensor)

        times = np.array(obs_times)
        mask = np.stack(obs_tiles)
        obs_mask = xr.Dataset({
            "time": (("time",), times),
            "observations": (("observations",), all_vars),
            "platforms" : (("observations"), platforms),
            "sensors" : (("observations"), sensors),
            "mask": (("time", "observations", "lat_tiles", "lon_tiles"), mask)
        })
        return obs_mask


    @cached_property
    def obs_vars(self):
        return self.stats_data.attrs["variables"].split(",")

    @cache
    def get_minmax(self, obs_id: int):
        var_name = self.obs_vars[min(obs_id, len(self.obs_vars) - 1)]
        min_val = self.stats_data[f"{var_name}_min"].data
        max_val = self.stats_data[f"{var_name}_max"].data
        return 0, 300

    def has_obs(self, time: np.datetime64):
        date = to_datetime(time)
        path = self.observation_path / date.strftime("%Y/%m/%d/obs_%Y%m%d%H%M%S.nc")
        return path.exists()

    def load_observations(
            self,
            time: np.datetime64,
            offset: Optional[int] = None,
            randomize: bool = True
    ):
        """
        Load observations for a given time.
        """
        date = to_datetime(time)
        path = self.observation_path / date.strftime("%Y/%m/%d/obs_%Y%m%d%H%M%S.nc")

        observations = -1.5 * torch.ones(self.n_tiles + (self.observation_layers, 1) + self.tile_size)
        meta_data = --1.5 * torch.ones(self.n_tiles + (self.observation_layers, 8) + self.tile_size)

        if not path.exists():
            LOGGER.warning(
                "No observations for time %s.", time
            )
            return observations, meta_data

        layer_ind = np.zeros(self.n_tiles, dtype=np.int64)

        try:
            data = xr.load_dataset(path)
        except Exception:
            return observations, meta_data


        for row_ind in range(self.n_tiles[0]):
            for col_ind in range(self.n_tiles[1]):

                obs_name = f"observations_{row_ind:02}_{col_ind:02}"
                if obs_name not in data:
                    continue

                try:
                    obs = data[obs_name].data
                    if randomize:
                        inds = np.random.permutation(obs.shape[0])
                    else:
                        inds = np.arange(obs.shape[0])

                    tiles = min(obs.shape[0], self.observation_layers)

                    obs_ids = f"obs_id_{row_ind:02}_{col_ind:02}"
                    obs_ids = data[obs_ids].data[inds[:tiles]]
                    minmax = np.array([self.get_minmax(obs_id) for obs_id in obs_ids])
                    minmax = minmax[..., None, None]

                    obs = obs[inds[:tiles]]
                    invalid = np.isnan(obs)
                    obs_n = -1.0 + 2.0 * (obs - minmax[:, 0]) / (minmax[:, 1] - minmax[:, 0])
                    obs_n[invalid] = -1.5
                    observations[row_ind, col_ind, :tiles, 0]  = torch.tensor(obs_n)

                    freq = np.log10(data[f"frequency_{row_ind:02}_{col_ind:02}"].data[inds[:tiles]])
                    freq = -1.0 + 2.0 * (freq - np.log10(self.freq_max)) / (np.log10(self.freq_max) - np.log10(self.freq_min))
                    offs = data[f"offset_{row_ind:02}_{col_ind:02}"].data[inds[:tiles]]
                    offs = np.minimum(offs, 10) / 10
                    pol = torch.nn.functional.one_hot(
                        torch.tensor(data[f"polarization_{row_ind:02}_{col_ind:02}"].data[inds[:tiles]]).to(dtype=torch.int64),
                        num_classes=5
                    )

                    time_offset = data[f"time_offset_{row_ind:02}_{col_ind:02}"].data[inds[:tiles]] / 180.0
                    if offset is not None:
                        time_offset = time_offset + offset

                    meta_data[row_ind, col_ind, :tiles, 0] = torch.tensor(freq)[..., None, None]
                    meta_data[row_ind, col_ind, :tiles, 1] = torch.tensor(offs)[..., None, None]
                    meta_data[row_ind, col_ind, :tiles, 2] = torch.tensor(time_offset)
                    meta_data[row_ind, col_ind, :tiles, 3:] = pol[..., None, None]
                except Exception as exc:
                    raise exc

                    LOGGER.warning(
                        "Encountered an error when loading observations from file '%s'.",
                        path
                    )

        observations = torch.nan_to_num(observations, nan=-3.0)
        meta_data = torch.nan_to_num(meta_data, nan=-3.0)
        return observations, meta_data


class DirectPrecipForecastWithObsDataset(DirectPrecipForecastDataset):
    """
    A PyTorch Dataset for loading precipitation forecast training data with global satellite
    observations.
    """
    def __init__(
            self,
            root_dir: Union[Path, str],
            time_step: int = 3,
            max_steps: int = 24,
            sentinel: float = -3.0,
            n_tiles: Tuple[int, int] = (12, 18),
            tile_size: Tuple[int, int] = (30, 32),
            climate: bool = False,
            sampling_rate: float = 1.0,
            reference_data: str = "IMERG"
    ):
        """
        Args:
            root_dir (str): Root directory containing year/month/day folders.
            time_step: The forecast time step.
            max_steps: The maximum number of timesteps to forecast precipitation.
        """
        root_dir = Path(root_dir)
        self.obs_loader = ObservationLoader(
            root_dir / "obs",
            n_tiles=n_tiles,
            tile_size=tile_size,
            observation_layers=32
        )
        super().__init__(
            root_dir=root_dir,
            time_step=time_step,
            max_steps=max_steps,
            climate=climate,
            sampling_rate=1.0,
            reference_data=reference_data,
            augment=False
        )
        self._sampling_rate = sampling_rate
        self.rng = np.random.default_rng(seed=42)


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
        for ind in range(self.input_times.size):
            sample_time = self.input_times[ind]
            input_times = [sample_time + np.timedelta64(t_i * self.time_step, "h") for t_i in [-1, 0]]
            output_times = [
                sample_time + np.timedelta64(t_i * self.time_step, "h") for t_i in np.arange(1, self.max_steps + 1)
            ]
            output_times = [t_o for t_o in output_times if t_o in self.output_times]
            valid = all([t_i in self.input_times for t_i in input_times])
            if valid and len(output_times) > 0:
                input_indices.append([ind - self.time_step // 3, ind])
                output_inds = []
                for output_time in output_times:
                    output_ind = np.searchsorted(self.output_times, output_time)
                    output_inds.append(output_ind)
                output_indices.append(output_inds + [-1] * (self.max_steps - len(output_inds)))
        return np.array(input_indices), np.array(output_indices)


    def __len__(self):
        return trunc(len(self.input_indices) * self._sampling_rate)


    def __getitem__(self, ind: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a single data point from the dataset.
        """
        lower = trunc(ind / self._sampling_rate)
        upper = min(trunc((ind + 1) / self._sampling_rate), len(self.input_indices) - 1)
        if lower < upper:
            ind = self.rng.integers(lower, upper)
        else:
            ind = lower

        x, y = super().__getitem__(ind)
        input_times = [self.input_times[ind] for ind in self.input_indices[ind]]
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

        x["obs"] = obs
        x["obs_mask"] = obs_mask
        x["obs_meta"] = meta

        return x, y
