"""
prithvi_precip.datasets.observations
====================================

Provides dataset to load satellite data together with the MERRA-2 input data.
"""
from filelock import FileLock
from functools import cache, cached_property
import logging
from math import trunc
import os
from pathlib import Path
import re
import shutil
from time import sleep
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

from .precipitation import (
    DirectPrecipForecastDataset,
    AutoregressivePrecipForecastDataset
)
from ..utils import to_datetime


LOGGER = logging.getLogger(__name__)



def untile(tnsr: torch.Tensor) -> torch.Tensor:
    """
    Untile tensor.

    Args:
        tnsr: The tensor to untile.
    """
    return torch.permute(tnsr, (2, 3, 0, 4, 1, 5)).flatten(-2, -1).flatten(-3, -2)


def tile(tnsr: torch.Tensor, tile_size: Tuple[int, int]) -> torch.Tensor:
    """
    Split global tensor into tiles and move tile dimensions to front.

    Args:
        tnsr: The tensor to tile.
        tile_size: A tuple specifying the size of the tiles.

    Return:
        The reshaped and transposed tensor.
    """
    TY, TX = tile_size
    L, C, Y, X = tnsr.shape
    tnsr = tnsr.reshape(L, C, Y // TY, TY, X // TX, TX)
    return torch.permute(tnsr, (2, 4, 0, 1, 3, 5))


def transform_observations(
        observations: Dict[str, torch.Tensor],
        meta_data: torch.Tensor,
        tile_size: Tuple[int, int],
        roll: int,
        flip_v: bool,
        flip_h: bool,
        scale: float = 1.0
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Transform data by applying zonal roll and meridional flipping.

    Args:
        inpt: The dictionary containing the input data.
        targets: A single tensor containing the target or a dictionary containing the targets
        roll: The number of pixels by which to roll the data zonally.
        flip_v: Whether or not to flip the data meridionally.
        flip_h: Whether or not to flip the data zonally.
        scale: Apply scaling to data

    Return:
        A tuple ``(inpt, target)`` containing the transformed input data.
    """
    observations = untile(observations)
    meta_data = untile(meta_data)
    if 0 < roll:
        observations = torch.roll(observations, roll, dims=-1)
        meta_data = torch.roll(meta_data, roll, dims=-1)

    if flip_v:
        observations = torch.flip(observations, dims=(-2,))
        meta_data = torch.flip(meta_data, dims=(-2,))

    if flip_h:
        observations = torch.flip(observations, dims=(-1,))
        meta_data = torch.flip(meta_data, dims=(-1,))

    if scale != 1.0:
        from torchvision.transforms.v2.functional import affine
        observations = affine(observations, angle=0, translate=[0, 0], shear=0, scale=scale)
        meta_data = affine(meta_data, angle=0, translate=[0, 0], shear=0, scale=scale)

    observations = tile(observations, tile_size)
    meta_data = tile(meta_data, tile_size)

    return observations, meta_data


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
            n_tiles: A tuple specifying the number of meridional and zonal tiles,
                 respectively.
            tile_size: A  tuple specifying the size of the observation tiles.
        """
        self.observation_path = Path(observation_path)
        self.observation_layers = observation_layers
        self.n_tiles = n_tiles
        self.tile_size = tile_size
        self.rng = np.random.default_rng(seed=42)
        self.time_step = 3
        self.freq_min = 1.0
        self.freq_max = 30e3
        self._obs_regexp = None


    @property
    def obs_regexp(self) -> Union[re.Pattern, None]:
        return self._obs_regexp

    @obs_regexp.setter
    def obs_regexp(self, regexp: str) -> None:
        self._obs_regexp = re.compile(regexp)


    def copy_files(
            self,
            input_times: np.ndarray,
            input_steps: int,
            local_data: Path,
            validation: bool = False
    ) -> None:
        """
        Shards data across nodes and copies them to the location pointed to by self.local_data.

        Args:
            input_times: An array containing the time stamps for which observations are required.
            input_steps: The time differences between input steps in hours.
            local_data: The local data path of the machine.
            validation: If 'True' data will be copied to local validation data folder.
        """
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        base_folder = self.observation_path.parent.parent.name
        if validation:
            obs_local = local_data / base_folder / f"validation_data_{local_rank:02}" / "obs"
        else:
            obs_local = local_data / base_folder / f"training_data_{local_rank:02}" / "obs"
        obs_local.mkdir(exist_ok=True, parents=True)

        obs_files = []

        for time in input_times:

            date = to_datetime(time)
            path = self.observation_path / date.strftime("%Y/%m/%d/obs_%Y%m%d%H%M%S.nc")
            if path.exists():
                obs_files.append(path.relative_to(self.observation_path))

            for step in input_steps:
                date = to_datetime(time - np.timedelta64(step, "h"))
                path = self.observation_path / date.strftime("%Y/%m/%d/obs_%Y%m%d%H%M%S.nc")
                if path.exists():
                    obs_files.append(path.relative_to(self.observation_path))

        obs_files = list(set(obs_files))
        # Copy input and output samples.
        LOGGER.info(
            "Copying %s observations files to local directory %s.",
            len(obs_files),
            obs_local
        )

        for path in obs_files:
            rel_path = Path(path)
            target_path = obs_local / rel_path
            if not target_path.exists():
                target_path.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy2(self.observation_path / rel_path, target_path)

        if local_rank == 0:
            LOGGER.info(
                "Copying stats file to temporary directory."
            )
            stats = obs_local / "stats.nc"
            if not stats.exists():
                shutil.copy2(self.observation_path / "stats.nc", obs_local / "stats.nc")
        else:
            stats = obs_local / "stats.nc"
            while not stats.exists():
                sleep(0.1)

        rank = int(os.environ.get("RANK", 0))

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
            with xr.open_dataset(stats_file, engine="h5netcdf", chunks=None, cache=False) as data:
                stats = data.load()
            del data
        return stats

    def get_observation_mask(
            self,
            start_time: np.datetime64,
            end_time: np.datetime64,
            step: np.timedelta64 = np.timedelta64(3, "h")
    ) -> xr.Dataset:
        """
        Calculate an xarray.Dataset indicating the availability of different observations.

        Args:
            start_time: The beginning of the time interval during which to look for observations.
            end_time: The end of the time interval.

        Return:
            An xarray dataset
        """
        from tqdm import tqdm
        stats = self.stats_data
        all_vars = stats.attrs["variables"].split(",")
        n_vars = len(all_vars)

        times = np.arange(start_time, end_time + np.timedelta64(3, "h"), step)
        obs_times = []
        obs_tiles = []
        obs_bins = np.arange(n_vars + 1) - 0.5

        for time in tqdm(times):
            date = to_datetime(time)
            path = self.observation_path / date.strftime("%Y/%m/%d/obs_%Y%m%d%H%M%S.nc")
            if path.exists():
                try:
                    with xr.open_dataset(path, engine="h5netcdf", chunks=None, cache=False) as obs_data:
                        obs_times.append(time)
                        obs_tiles.append(np.zeros((obs_bins.size - 1,) + self.n_tiles))

                        for meridional_index in range(self.n_tiles[0]):
                            for zonal_index in range(self.n_tiles[1]):
                                tilename = f"obs_id_{meridional_index:02}_{zonal_index:02}"
                                if tilename in obs_data:
                                    obs_ids = obs_data[tilename].data
                                    obs_ids[obs_ids < 0] += 256
                                    obs_tiles[-1][:, meridional_index, zonal_index] += np.histogram(obs_ids, bins=obs_bins)[0]
                except Exception as exc:
                    LOGGER.exception(
                        "Encountered an error when observation file %s.",
                        path
                    )
                    path.unlink()

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

    def get_available_observations(self, time: np.datetime64) -> List[str]:
        """
        Return list of observation layers available for a given time.

        Args:
            time: A numpy.datetime64 object defining the time.

        Returns:
            A list of the observation layers names available that data.
        """
        date = to_datetime(time)
        path = self.observation_path / date.strftime("%Y/%m/%d/obs_%Y%m%d%H%M%S.nc")
        if not path.exists():
            return []

        stats = self.stats_data
        all_vars = stats.attrs["variables"].split(",")
        n_vars = len(all_vars)

        obs_ids = []

        with xr.open_dataset(path, engine="h5netcdf", chunks=None, cache=False) as obs_data:
            for meridional_index in range(self.n_tiles[0]):
                for zonal_index in range(self.n_tiles[1]):
                    tilename = f"obs_id_{meridional_index:02}_{zonal_index:02}"
                    if tilename in obs_data:
                        obs_ids_tile = obs_data[tilename].data.astype(np.int64)
                        obs_ids_tile[obs_ids_tile < 0] += 256
                        obs_ids += list(obs_data[tilename].data)

        obs_ids = set(obs_ids)
        available = [all_vars[obs_id] for obs_id in obs_ids]
        return available

    def get_observations(self, time: np.datetime64, obs: str) -> np.ndarray:
        """
        Return single observation layer.

        Args:
            time: A numpy.datetime64 object defining the time.
            obs: The name of the observation layer to load.

        Returns:
            A list of the observation layers names available that data.
        """
        stats = self.stats_data
        date = to_datetime(time)
        path = self.observation_path / date.strftime("%Y/%m/%d/obs_%Y%m%d%H%M%S.nc")
        if not path.exists():
            return []

        all_vars = stats.attrs["variables"].split(",")
        obs_id = all_vars.index(obs)

        obs_ids = []
        observations = np.nan * np.ones(self.n_tiles + self.tile_size)

        with xr.open_dataset(path, engine="h5netcdf", chunks=None, cache=False) as obs_data:
            for meridional_index in range(self.n_tiles[0]):
                for zonal_index in range(self.n_tiles[1]):
                    obs_id_name = f"obs_id_{meridional_index:02}_{zonal_index:02}"
                    obs_name = f"observations_{meridional_index:02}_{zonal_index:02}"
                    if obs_id_name not in obs_data:
                        continue
                    obs_ids = obs_data[obs_id_name].data.astype(np.int64)
                    obs_ids[obs_ids < 0] += 256
                    if obs_id in obs_ids:
                        obs_ind = list(obs_ids).index(obs_id)
                        observations[meridional_index, zonal_index, :, :] = obs_data[obs_name].data[obs_ind]

        n_y_g, n_x_g, n_y_l, n_x_l = observations.shape
        observations = np.transpose(observations, (0, 2, 1, 3)).reshape(n_y_g * n_y_l, n_x_g * n_x_l)
        return observations

    @cached_property
    def obs_vars(self):
        return self.stats_data.attrs["variables"].split(",")

    @cache
    def get_minmax(self, obs_id: int):
        #var_name = self.obs_vars[min(obs_id, len(self.obs_vars) - 1)]
        #min_val = self.stats_data[f"{var_name}_min"].data
        #max_val = self.stats_data[f"{var_name}_max"].data
        return 0, 300

    def has_obs(self, time: np.datetime64):
        date = to_datetime(time)
        path = self.observation_path / date.strftime("%Y/%m/%d/obs_%Y%m%d%H%M%S.nc")
        return path.exists()

    def load_observations(
            self,
            time: np.datetime64,
            offset: Optional[int] = None,
            randomize: bool = True,
            roll: int = 0,
            flip_v: bool = False,
            flip_h: bool = False,
            scale: float = 1.0
    ):
        """
        Load observations for a given time.
        """
        date = to_datetime(time)
        path = self.observation_path / date.strftime("%Y/%m/%d/obs_%Y%m%d%H%M%S.nc")

        observations = -1.5 * torch.ones(self.n_tiles + (self.observation_layers, 1) + self.tile_size)
        meta_data = -1.5 * torch.ones(self.n_tiles + (self.observation_layers, 8) + self.tile_size)

        if not path.exists():
            LOGGER.warning(
                "No observations for time %s.", time
            )
            return observations, meta_data

        layer_ind = np.zeros(self.n_tiles, dtype=np.int64)

        data = None
        try:
            with xr.open_dataset(path, engine="h5netcdf", chunks=None, cache=False) as data:

                for row_ind in range(self.n_tiles[0]):
                    for col_ind in range(self.n_tiles[1]):

                        obs_name = f"observations_{row_ind:02}_{col_ind:02}"
                        if obs_name not in data:
                            continue

                        try:
                            obs = data[obs_name].data

                            if self.obs_regexp is not None:
                                obs_ids = f"obs_id_{row_ind:02}_{col_ind:02}"
                                obs_ids = data[obs_ids].data
                                obs_ids[obs_ids < 0] += 256
                                names = [self.obs_vars[obs_id] for obs_id in obs_ids if self.obs_regexp.match(self.obs_vars[obs_id])]
                                inds = [ind for ind, obs_id in enumerate(obs_ids) if self.obs_regexp.match(self.obs_vars[obs_id])]
                                inds = np.array(inds)
                                if len(inds) == 0:
                                    continue
                            else:
                                if randomize:
                                    inds = np.random.permutation(obs.shape[0])
                                else:
                                    inds = np.arange(obs.shape[0])

                            tiles = min(obs.shape[0], self.observation_layers, len(inds))

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
                            freq = -1.0 + 2.0 * (freq - np.log10(self.freq_min)) / (np.log10(self.freq_max) - np.log10(self.freq_min))
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
                            LOGGER.exception(
                                "Encountered an error when loading observations for tile [%s, %s] from file '%s'.",
                                row_ind, col_ind, path
                            )
        except Exception:
            return observations, meta_data
        finally:
            del data

        observations = torch.nan_to_num(observations, nan=-1.5)
        meta_data = torch.nan_to_num(meta_data, nan=-1.5)

        if (0 < roll) or flip_h or flip_v or scale != 1.0:
            observations, meta_data = transform_observations(
                observations,
                meta_data,
                self.tile_size,
                roll=roll,
                flip_v=flip_v,
                flip_h=flip_h,
                scale=scale
            )

        return observations, meta_data


class ObsDatasetBase():
    def __init__(
            self,
            training_data_path: Path,
            observation_layers: int = 32,
            n_tiles: Tuple[int, int] = (12, 18),
            tile_size: Tuple[int, int] = (30, 32),
    ):
        training_data_path = Path(training_data_path)
        self.obs_loader = ObservationLoader(
            training_data_path / "obs",
            n_tiles=n_tiles,
            tile_size=tile_size,
            observation_layers=observation_layers
        )

        if self.local_data:
            self.obs_loader.copy_files(
                self.input_times,
                self.input_steps,
                self.local_data,
                validation=self.validation
            )
            self.obs_loader.observation_path = self.training_data_path / "obs"


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
            roll = 0
            flip_v = False
            flip_h = False
            scale = 1.0
            if not self.validation and self.augment:
                roll = self.rng.integers(0, 576)
                flip_v = 0.5 < self.rng.random()
                flip_h = 0.5 < self.rng.random()
                scale = self.rng.uniform(1.0, 1.2)

            x, y = self.load_data(ind, roll, flip_v=flip_v, flip_h=flip_h, scale=scale)

            input_times = [self.input_times[ind] for ind in self.input_indices[ind]]
            obs = []
            meta = []
            input_time = int(x["input_time"].item())
            for time_ind, time in enumerate(input_times):
                obs_t = []
                meta_t = []
                n_steps = input_time // 3
                for step in range(-input_time + 3, 1, 3):
                    obs_s, meta_s = self.obs_loader.load_observations(
                        time + np.timedelta64(step, "h"),
                        offset=step // 3,
                        roll=roll,
                        flip_v=flip_v,
                        flip_h=flip_h,
                        scale=scale
                    )
                    obs_t.append(obs_s)
                    meta_t.append(meta_s)
                obs.append(torch.cat(obs_t, 2))
                meta.append(torch.cat(meta_t, 2))
            obs = torch.stack(obs, 0)
            obs_mask = obs < -1.4
            obs = torch.nan_to_num(obs, nan=-1.5)
            meta = torch.stack(meta, 0)

            x["obs"] = obs
            x["obs_mask"] = obs_mask
            x["obs_meta"] = meta

            return x, y

        except Exception as exc:
            LOGGER.exception(
                "Encountered an error when load training sample %s. Falling back to another "
                " randomly-chosen sample.",
                ind
            )
            new_ind = np.random.randint(0, len(self))
            return self[new_ind]

        return x, y


class DirectPrecipForecastWithObsDataset(ObsDatasetBase, DirectPrecipForecastDataset):
    """
    A PyTorch Dataset for loading precipitation forecast training data with global satellite
    observations.
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
            reference_data: str = "imerg",
            center_meridionally: bool = True,
            validation: bool = False,
            local_data: Optional[Path] = None,
            augment: bool = False,
            source: str = "merra2",
            observation_layers: int = 32,
            n_tiles: Tuple[int, int] = (12, 18),
            tile_size: Tuple[int, int] = (30, 32),
    ):
        """
        Args:
            training_data_path: The directory containing the dynamic input data.
            input_time: The time difference between input samples.
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
            observation_layers: The number of observation layers to load.
            n_tiles: The number of global observations tiles.
            tile_size: The size of each tile.
        """
        DirectPrecipForecastDataset.__init__(
            self,
            training_data_path=training_data_path,
            input_time=input_time,
            lead_time=lead_time,
            accumulation_period=accumulation_period,
            max_steps=max_steps,
            climate=climate,
            sampling_rate=sampling_rate,
            reference_data=reference_data,
            center_meridionally=center_meridionally,
            validation=validation,
            local_data=local_data,
            augment=augment,
            source=source,
        )
        ObsDatasetBase.__init__(
            self,
            training_data_path,
            observation_layers=observation_layers,
            n_tiles=n_tiles,
            tile_size=tile_size
        )


class AutoregressivePrecipForecastWithObsDataset(ObsDatasetBase, AutoregressivePrecipForecastDataset):
    """
    A PyTorch Dataset for loading precipitation forecast training data with global satellite
    observations.
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
            reference_data: str = "imerg",
            center_meridionally: bool = True,
            validation: bool = False,
            local_data: Optional[Path] = None,
            augment: bool = False,
            source: str = "merra2",
            observation_layers: int = 32,
            n_tiles: Tuple[int, int] = (12, 18),
            tile_size: Tuple[int, int] = (30, 32),
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
            observation_layers: The number of observation layers to load.
            n_tiles: The number of global observations tiles.
            tile_size: The size of each tile.
        """
        AutoregressivePrecipForecastDataset.__init__(
            self,
            training_data_path=training_data_path,
            scaling_factors=scaling_factors,
            input_time=input_time,
            lead_time=lead_time,
            accumulation_period=accumulation_period,
            max_steps=max_steps,
            climate=climate,
            sampling_rate=sampling_rate,
            reference_data=reference_data,
            center_meridionally=center_meridionally,
            validation=validation,
            local_data=local_data,
            augment=augment,
            source=source,
        )
        ObsDatasetBase.__init__(
            self,
            training_data_path,
            n_tiles=n_tiles,
            tile_size=tile_size,
            observation_layers=observation_layers
        )
