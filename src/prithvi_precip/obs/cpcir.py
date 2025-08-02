"""
prithvi_precip.obs.cpcir
========================

This module provides functionality to extract geostationary 11 um satellite observations.
"""
from calendar import monthrange
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
import logging
from pathlib import Path
import re
from typing import Tuple

import click
from filelock import FileLock
from scipy.constants import speed_of_light
import numpy as np
from pansat import FileRecord
from pansat.time import to_datetime64, TimeRange
from pansat.utils import resample_data
from pansat.products.satellite.gpm import merged_ir
from pyresample.geometry import AreaDefinition
from rich.progress import Progress
from tqdm import tqdm
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



def extract_observations(
        base_path: Path,
        cpcir_file: FileRecord,
        n_tiles: Tuple[int, int] = (12, 18),
        time_step = np.timedelta64(3, "h"),
        domain: str = "MERRA",
        tile_dims: Tuple[int, int] = (30, 32)
) -> xr.Dataset:
    """
    Extract observations from a GPM L1C file.

    Args:
        base_path: Path object pointing to the directory to which to write the extracted observations.
        gpm_file: A pansat FileRecord pointing to a GPM input file.
        radius_of_influence: Radius of influence to use for resampling.
    """
    cpcir_obs = cpcir_file.product.open(cpcir_file).rename(Tb="observations")
    lons, lats = domains.get_lonlats(domain)
    lons = lons[0]
    lats = lats[:, 0]

    cpcir_obs = cpcir_obs.interp(latitude=lats, longitude=lons)

    wavelength = 11.0
    frequency = speed_of_light / (wavelength / 1e6) / 1e9

    cpcir_obs = cpcir_obs.coarsen({"latitude": tile_dims[0], "longitude": tile_dims[1]})
    cpcir_obs = cpcir_obs.construct(
        {"longitude": ("tiles_zonal", "lon_tile"),
         "latitude": ("tiles_meridional", "lat_tile")}
    )
    cpcir_obs = cpcir_obs.stack(tiles=("tiles_meridional", "tiles_zonal"))
    cpcir_obs = cpcir_obs.transpose("tiles", "time", "lat_tile", "lon_tile")

    obs = cpcir_obs.observations.data
    obs[obs < 101] = np.nan
    obs[obs > 350] = np.nan

    for time_ind in range(cpcir_obs.time.size):

        cpcir_obs_t = cpcir_obs[{"time": time_ind}]
        time = cpcir_obs_t.time.data

        ref_time = round_time(time, time_step)
        rel_time = time - ref_time

        valid = np.isfinite(cpcir_obs_t.observations.data)
        if valid.sum() == 0:
            continue
        obs_id = track_stats(base_path, "cpcir", cpcir_obs_t.observations.data)
        cpcir_obs_t.attrs["obs_name"] = "cpcir"

        valid_tiles = np.isfinite(cpcir_obs_t.observations).mean(("lon_tile", "lat_tile")) > 0.5
        cpcir_obs_t = cpcir_obs_t[{"tiles": valid_tiles}].reset_index("tiles")

        n_tiles = cpcir_obs_t.tiles.size
        ones = np.ones(n_tiles)
        height = cpcir_obs_t.lat_tile.size
        width = cpcir_obs_t.lon_tile.size

        cpcir_obs_t["wavelength"] = (("tiles",), wavelength * ones)
        cpcir_obs_t["frequency"] = (("tiles",), frequency * ones)
        cpcir_obs_t["offset"] = (("tiles",), frequency * ones)
        cpcir_obs_t["polarization"] = (("tiles",), 0 * ones.astype(np.int8))
        cpcir_obs_t["obs_id"] = (("tiles",), obs_id * ones.astype(np.int8))
        cpcir_obs_t["time_offset"] = (
            ("tiles", "lat_tile", "lon_tile"), rel_time.astype("timedelta64[m]").astype("float32") * np.ones((n_tiles, height, width))
        )

        rel_minutes = rel_time.astype("timedelta64[m]").astype("int64")

        filename = ref_time.astype("datetime64[s]").item().strftime("obs_%Y%m%d%H%M%S.nc")
        output_path = base_path / get_output_path(ref_time) / filename

        new_data = split_tiles(cpcir_obs_t)

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

            lock = FileLock(output_path.with_suffix(".lock"))
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
        interval: int = 1
    """
    start = datetime(year, month, day)
    end = start + timedelta(days=1)
    time_steps = np.arange(start, end, np.timedelta64(interval, "h"))
    recs = []
    for step in time_steps:
        recs += merged_ir.get(TimeRange(step + np.timedelta64(30, "m")))

    for rec in recs:
        try:
            extract_observations(output_path, rec, domain=domain, tile_dims=tile_dims)
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
def extract_cpcir_observations(
        year: int,
        month: int,
        output_path: str,
        domain: str,
        tile_dims: Tuple[int, int] = (30, 32),
        n_processes: int = 1,
        interval: int = 1
):
    """
    Extract global geostationary observations from the CPCIR dataset for given year and month.

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
