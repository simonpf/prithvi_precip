"""
prithvi_precip.mlp
==================

Provides dataloaders to load sever weather and precipitation accumulation statistics similar
to CSU MLP system.
"""
from datetime import datetime
import logging
from functools import partial, cached_property
from math import trunc
import os
from pathlib import Path
import shutil
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import xarray as xr

from .datasets import MERRAInputData
from .utils import load_static_input, load_climatology, to_datetime64


LOGGER = logging.getLogger(__name__)


class SevereWeatherForecastDataset(MERRAInputData):
    """
    A PyTorch Dataset for forecasting sever weather occurrence.
    """
    def __init__(
            self,
            training_data_path: Union[Path, str],
            max_steps: int = 12,
            climate: bool = True,
            sampling_rate: float = 1.0,
            center_meridionally: bool = True,
            validation: bool = False,
            local_data: Optional[Path] = None,
            cleanup: bool = False
    ):
        """
        Args:
            training_data_path: The directory containing the dynamic input data.
            max_steps: The maximum number of timesteps to forecast precipitation.
            climate: Whether to include climatology data in the input.
            sampling_rate: Sub- or super-sample dataset.
            center_meridionally: Whether to center the data meridionally instead of cropping.
            validation: Boolean flag inidicating whether that dataset contains validation data.
            local_data: Optional Path pointing to a local directory to use
            cleanup: Whether to cleanup copied training samples.
        """
        self.training_data_path = Path(training_data_path)
        self.data_path = self.training_data_path.parent
        self.input_time = 24
        self.accumulation_period = 24
        self.max_steps = max_steps
        self.climate = climate
        self.sampling_rate = sampling_rate
        self.center_meridionally = center_meridionally
        self.validation = validation

        self.local_data = None
        if local_data is not None:
            self.local_data = Path(local_data)
        self.cleanup = cleanup

        self.input_times, self.input_files = self.find_merra_files(self.training_data_path)
        self.output_times, self.output_files = self.find_target_files(self.training_data_path)
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
            static_data = training_local.parent / "static"
            if not static_data.exists():
                shutil.copytree(self.training_data_path.parent / "static", static_data, dirs_exist_ok=True)
            climatology = training_local.parent / "climatology"
            if not climatology.exists():
                shutil.copytree(self.training_data_path.parent / "climatology", climatology, dirs_exist_ok=True)


        rank = int(os.environ.get("RANK", 0))

        self.training_data_path = training_local
        self.data_path = self.training_data_path.parent
        self.input_times, self.input_files = self.find_merra_files(self.training_data_path)
        self.output_times, self.output_files = self.find_target_files(self.training_data_path)
        self.input_indices, self.output_indices = self.calculate_valid_samples()
        assert len(self.input_indices) == n_samples_local


    def __del__(self) -> None:
        """
        Clean up temporary directory if it exists.
        """
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0 and self.local_data is not None and self.cleanup:
            if self.local_data.exists():
                shutil.rmtree(self.local_data)

    @cached_property
    def conus_slices(self) -> Dict[str, slice]:
        with xr.open_dataset(self.training_data_path / self.input_files[0]) as data:
            lons = data.longitude.data
            lats = data.latitude.data
        data.close()
        del data

        col_start = np.where(lons > -125)[0][0]
        col_end = col_start + 3 * 32
        row_start = np.where(lats > 25)[0][0]
        row_end = row_start + 2 * 30
        return {
            "latitude": slice(row_start, row_end),
            "longitude": slice(col_start, col_end),
        }

    def find_target_files(self, training_data_path: Path,) -> np.ndarray:
        """
        Find target files for training.
        """
        times = []
        files = []

        reference_data = "severe_weather_24"

        prefix = f"{reference_data.lower()}"
        pattern = f"**/{prefix}*.nc"
        date_pattern = f"severe_weather_%Y%m%d%H%M%S.nc"
        pattern = "severe_weather_24/**/severe_weather_*"

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
                sample_time + np.timedelta64(int(t_i * self.input_time), "h") for t_i in np.arange(0, self.max_steps) + 0.5
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

            inpt = {
                "x": transform(torch.stack(dynamic_in, 0)),
                "static": transform(static_in),
                "input_time": torch.tensor(input_time / 24.0).to(dtype=torch.float32),
            }

            inds = self.output_indices[ind]
            inds = inds[0 <= inds]
            output_ind = self.rng.choice(inds)
            output_file = self.output_files[output_ind]
            output_time = self.output_times[output_ind]

            lead_time = (output_time - max(input_times)).astype("timedelta64[h]").astype(np.float32)
            inpt["lead_time"] = torch.tensor(lead_time / 24.0).to(dtype=torch.float32)

            if self.climate:
                climate = load_climatology(output_time, self.data_path)
                inpt["climate"] = transform(torch.tensor(climate))

            with xr.open_dataset(self.training_data_path / output_file) as data:
                LOGGER.debug("Loading sever weather mask from %s.", output_file)
                data = data[self.conus_slices].compute()
                tornado = torch.tensor(data.tornado.data)
                hail = torch.tensor(data.hail.data)
                wind = torch.tensor(data.wind.data)
                tornado = (tornado > 0)
                hail = (hail > 0)
                wind = (wind > 0)
                severe = tornado + hail + wind

                target = {
                    "tornado": tornado.to(dtype=torch.float32),
                    "hail": hail.to(dtype=torch.float32),
                    "wind": wind.to(dtype=torch.float32),
                    "severe": severe.to(dtype=torch.float32)
                }


            data.close()
            del data

            lat_bounds = self.conus_slices["latitude"]
            lon_bounds = self.conus_slices["longitude"]

            inpt.update({
                "x_regional": inpt["x"][..., lat_bounds, lon_bounds],
                "climate_regional": inpt["climate"][..., lat_bounds, lon_bounds],
                "static_regional": inpt["static"][..., lat_bounds, lon_bounds],
            })

            return inpt, target

        except Exception as exc:
            raise exc
            LOGGER.exception(
                "Encountered an error when load training sample %s. Falling back to another "
                " randomly-chosen sample.",
                ind
            )
            new_ind = np.random.randint(0, len(self))
            return self[new_ind]


    def get_direct_forecast_input(self, init_time: np.datetime64, target_step: int) -> Dict[str, torch.Tensor]:
        """
        Get forecast input data to perform a continuous forecast over a given number of steps
        using a direct forecasting model.

        Args:
            init_time: The initialization time of the forecast.
            target_step: The time step to forecast.

        Return:
            A dictionary contraining the loaded input tensors.
        """
        input_times = [init_time + np.timedelta64(t_i * 24, "h") for t_i in [-1, 0]]
        for input_time in input_times:
            if input_time not in self.input_times:
                raise ValueError(
                    "Required input data for t=%s not available.",
                    input_time
                )

        dynamic_in = []
        for input_time in input_times:
            ind = np.searchsorted(self.input_times, input_time)
            dynamic_in.append(self.load_dynamic_data(self.input_files[ind]))

        static_time = input_times[-1]
        static_in = load_static_input(static_time, self.data_path)

        if self.center_meridionally:
            transform = lambda tnsr: 0.5 * (tnsr[..., 1:, :] + tnsr[..., :-1, :])
        else:
            transform = partial(nn.functional.pad, pad=((0, 0, 0, -1)))

        dynamic_in = transform(torch.stack(dynamic_in, 0))[None]
        static_in = transform(torch.tensor(static_in))[None]
        input_time = torch.tensor(240.0)
        lead_time = torch.tensor(12.0 + target_step * 24.0)

        x = {
            "x": dynamic_in,
            "static": static_in,
            "lead_time": lead_time,
            "input_time": input_time,
        }

        target_time = init_time + np.timedelta64(int(24 * (target_step + 0.5)), "h")

        if self.climate:
            climate = torch.tensor(load_climatology(target_time, self.data_path)).to(dtype=torch.float32)
            x["climate"] = transform(climate)[None]

        lat_bounds = self.conus_slices["latitude"]
        lon_bounds = self.conus_slices["longitude"]

        x.update({
            "x_regional": x["x"][..., lat_bounds, lon_bounds],
            "climate_regional": x["climate"][..., lat_bounds, lon_bounds],
            "static_regional": x["static"][..., lat_bounds, lon_bounds],
        })


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
