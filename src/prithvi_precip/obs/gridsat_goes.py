"""
prithvi_precip.obs.gridsat_goes
===============================

This module provides functionality to extract satellite observations from the GridSat GOES dataset.
"""
from calendar import monthrange
import click
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from filelock import FileLock
import logging
from pathlib import Path
import re
from typing import Tuple

import click
from filelock import FileLock
from scipy.constants import speed_of_light
import numpy as np
from pansat import TimeRange
from pansat.file_record import FileRecord
from pansat.products.satellite.ncei import gridsat_goes
from rich.progress import Progress
import xarray as xr

from .. import domains
from .utils import (
    get_output_path,
    round_time,
    track_stats,
    split_tiles,
    combine_tiles
)


LOGGER = logging.getLogger(__name__)

WAVELENGTHS = {
    1: 0.6,
    2: 3.9,
    3: 6.7,
    4: 11,
    5: 12,
    6: 13,
}


def extract_observations(
        base_path: Path,
        goes_file: FileRecord,
        domain: str,
        tile_dims: Tuple[int, int] = (32, 32),
        time_step = np.timedelta64(3, "h"),
) -> None:
    """
    Extract observations from a GPM L1C file.

    Args:
        base_path: Path object pointing to the directory to which to write the extracted observations.
        gpm_file: A pansat FileRecord pointing to a GPM input file.
        domain: The name of the target domain.
        platform_name: The name of the satellite platform the sensor is on.
        sensor_name: The name of the sensor
        radius_of_influence: Radius of influence to use for resampling.
        tile_dims: A tuple defining the number of tiles.
        time_step: The time interval over which to integrate observations.
    """
    lons, lats = domains.get_lonlats(domain)
    lons = lons[0]
    lats = lats[:, 0]

    with xr.open_dataset(goes_file) as data:
        data = data[[f"ch{ch_ind}" for ch_ind in range(1, 7)]].compute()
        data = data.interp(lat=lats, lon=lons).rename(lat="latitude", lon="longitude")

        data = data.coarsen({"latitude": tile_dims[0], "longitude": tile_dims[1]})
        data = data.construct(
            {"longitude": ("tiles_zonal", "lon_tile"),
             "latitude": ("tiles_meridional", "lat_tile")}
        )
        data = data.stack(tiles=("tiles_meridional", "tiles_zonal"))
        data = data.transpose("tiles", "time", "lat_tile", "lon_tile")

        for ch_ind in range(1, 7):

            obs = data[[f"ch{ch_ind}"]][{"time": 0}].rename({f"ch{ch_ind}": "observations"})
            if ch_ind == 1:
                obs["observations"] = 100.0 * obs["observations"]

            time = obs.time.data

            ref_time = round_time(time, time_step)
            rel_time = time - ref_time

            valid = np.isfinite(obs[f"observations"].data)
            if valid.sum() == 0:
                continue

            obs_name = f"gridsat_goes_ch{ch_ind}"
            obs_id = track_stats(base_path, obs_name, obs[f"observations"].data)
            obs.attrs["obs_name"] = obs_name

            valid_tiles = np.isfinite(obs[f"observations"]).mean(("lon_tile", "lat_tile")) > 0.5
            obs = obs[{"tiles": valid_tiles}].reset_index("tiles")

            n_tiles = obs.tiles.size
            ones = np.ones(n_tiles)
            height = obs.lat_tile.size
            width = obs.lon_tile.size

            wavelength = WAVELENGTHS[ch_ind]
            frequency = speed_of_light / (wavelength / 1e6) / 1e9

            obs["wavelength"] = (("tiles",), wavelength * ones)
            obs["frequency"] = (("tiles",), frequency * ones)
            obs["offset"] = (("tiles",), frequency * ones)
            obs["polarization"] = (("tiles",), 0 * ones.astype(np.int8))
            obs["obs_id"] = (("tiles",), obs_id * ones.astype(np.int16))
            obs["time_offset"] = (
                ("tiles", "lat_tile", "lon_tile"), rel_time.astype("timedelta64[m]").astype("float32") * np.ones((n_tiles, height, width))
            )

            filename = ref_time.astype("datetime64[s]").item().strftime("obs_%Y%m%d%H%M%S.nc")
            output_path = base_path / get_output_path(ref_time) / filename

            new_data = split_tiles(obs)

            lock = FileLock(output_path.with_suffix(".lock"))
            with lock:

                if output_path.exists():
                        existing = xr.load_dataset(output_path)
                        new_data = combine_tiles(existing, new_data)

                fill_value = 2 ** 15 - 1
                encoding = {}
                for var in new_data:
                    if var.startswith("observations") or var.startswith("time_offset"):
                        encoding[var] = {
                                "zlib": True,
                                "dtype": "int16",
                                "scale_factor": 0.1,
                                "_FillValue": fill_value
                        }

                if not output_path.parent.exists():
                        output_path.parent.mkdir(parents=True)

                new_data.to_netcdf(output_path, encoding=encoding)


def extract_observations_day(
        year: int,
        month: int,
        day: int,
        output_path: Path,
        domain: str,
        tile_dims: Tuple[int, int] = (30, 32),
        interval: int = 1
):
    """
    Extract observation data from all CPCIR files for a given day.

    Args:
        year: The year.
        month: The month.
        day: The day.
        output_path: The path to which to write the extracted observation data.
        domain: The name of the target region.
        tile_dims: The dimension of the tiles to use for the target region.
        interval: The interval length in hours at which to extract observations.
    """
    start = datetime(year, month, day)
    end = start + timedelta(days=1)
    time_steps = np.arange(start, end, np.timedelta64(interval, "h"))
    recs = []
    for step in time_steps:
        recs += gridsat_goes.get(TimeRange(step, step + np.timedelta64(30, "m")))
    for rec in recs:
        try:
            extract_observations(output_path, rec.local_path, domain=domain, tile_dims=tile_dims)
        except:
            LOGGER.exception(
                "Encountered an error when processing input file %s.",
                rec.local_path
            )


@click.argument('year', type=int)
@click.argument('month', type=int)
@click.argument('output_path', type=click.Path())
@click.option(
    '--domain',
    type=str,
    default="MERRA",
    help="The domain over which to extract the data."

)
@click.option(
    '--tile_dims',
    type=str,
    default="30,32",
    help="The dimension of the spatial tiles."
)
@click.option(
    '--n_processes',
    type=int,
    default=1,
    help="The number of processes to use to parallelize the extraction"
)
@click.option(
    '--interval',
    type=int,
    default=3,
    help="The interval at which to extract observations."
)
def extract_goes_observations(
        year: int,
        month: int,
        output_path: str,
        domain: str,
        tile_dims: Tuple[int, int] = (30, 32),
        n_processes: int = 1,
        interval: int = 1
):
    """
    Extract GOES observations for given year and month.

    YEAR: Year of the data to process (integer)
    MONTH: Month of the data to process (integer)
    OUTPUT_PATH: Path to save the processed data (string/path)
    """
    if month < 1 or month > 12:
        click.echo("Error: Month must be between 1 and 12.")
        return

    _, n_days = monthrange(year, month)
    days = list(range(1, n_days + 1))

    tile_dims = tuple(map(int, tile_dims.split(",")))

    if n_processes > 1:
        LOGGER.info(f"[bold blue]Using {n_processes} processes for downloading data.[/bold blue]")
        tasks = [(year, month, d, output_path, domain, tile_dims, interval) for d in days]

        with (
                ProcessPoolExecutor(max_workers=n_processes) as executor,
                Progress() as progress
        ):
            task_id = progress.add_task("Extracting data:", total=len(tasks))
            future_to_task = {
                executor.submit(extract_observations_day, *task): task for task in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    future.result()
                except Exception as e:
                    LOGGER.exception(f"Task {task} failed with error: {e}")
                finally:
                    progress.update(task_id, advance=1)
    else:
        with Progress() as progress:
            task_id = progress.add_task("Extracting data:", total=len(days))
            for d in days:
                try:
                    extract_observations_day(year, month, d, output_path, domain, tile_dims, interval)
                except Exception as e:
                    LOGGER.exception(f"Error processing day {d}: {e}")
                finally:
                    progress.update(task_id, advance=1)
