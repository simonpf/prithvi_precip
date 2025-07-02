"""
prithvi_precip.data.stage4
==========================

Functionality to extract Prithvi Precip training data derived from Stage4 radar data.
"""
from calendar import monthrange
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import cache
import logging
from pathlib import Path

import click
import numpy as np
from pansat.time import TimeRange, to_datetime
from pansat.products.ground_based import stage4
from rich.progress import Progress
from scipy.stats import binned_statistic_2d
import xarray as xr

from precipfm.definitions import LAT_BINS, LON_BINS


LOGGER = logging.getLogger(__name__)

@cache
def load_stage4_data(year, month) -> xr.Dataset:
    """
    Load Stage IV data for given year and month.

    Args:
        year: The year
        month: The month

    Return:
        The loaded Stage IV data.
    """
    time_range = TimeRange(datetime(year, month, 10))
    rec = stage4.surface_precip.get(time_range)[0]
    return stage4.surface_precip.open(rec)



def extract_stage4_precip(
        year: int,
        month: int,
        day: int,
        accumulate: int,
        granularity: int,
        output_path: Path,
) -> None:
    """
    Extract reference preciptation fields from Stage IV data and resample to MERRA grid.

    Args:
        year: int specifying the year.
        month: int specifying the month
        day: int specifying the day.
        granularity: The frequency in hours at which to extract samples.
        accumulate: The period over which precipitation is accumulated.
        output_path: The path to which to write the extracted files.
    """
    output_path = Path(output_path)

    time = datetime(year=year, month=month, day=day)
    time_before = time - timedelta(days=1)
    time_now = datetime(year, month, day)
    time_after = time + timedelta(days=1)

    yearmonths = set([(time.year, time.month) for time in [time_before, time_now, time_after]])

    surface_precip = []
    for year, month in yearmonths:
        surface_precip.append(load_stage4_data(year, month))
    surface_precip = xr.concat(surface_precip, dim="time")

    time_shifted = surface_precip.time[:-accumulate]
    surface_precip = surface_precip.rolling(time=accumulate).mean()[{"time": slice(0, -accumulate)}]
    surface_precip = surface_precip.assign_coords(time=time_shifted)

    lons = surface_precip.longitude.data
    lats = surface_precip.latitude.data
    lons = lons
    lons[180 < lons] -= 360
    lats_r = 0.5 * (LAT_BINS[1:] + LAT_BINS[:-1])
    lons_r = 0.5 * (LON_BINS[1:] + LON_BINS[:-1])

    start_time = np.datetime64(f"{year}-{month:02}-{day:02}T00:00:00")
    end_time = start_time + np.timedelta64(1, "D")
    for time in np.arange(start_time, end_time, np.timedelta64(granularity, "h")):
        surface_precip_t = surface_precip.interp(time=time.astype("datetime64[ns]"), method="nearest")
        valid = 0.0 <= surface_precip_t.surface_precip.data
        surface_precip_r = binned_statistic_2d(
            lons[valid],
            lats[valid],
            surface_precip_t.surface_precip.data[valid],
            bins=(LON_BINS, LAT_BINS)
        )[0].T

        surface_precip_t = xr.Dataset({
            "latitude": (("latitude",), lats_r),
            "longitude": (("longitude",), lons_r),
            "surface_precip": (("latitude", "longitude"), surface_precip_r),
        })

        date = to_datetime(time)
        fname = date.strftime(f"stage4_{accumulate}/%Y/%m/%d/stage4_precip_%Y%m%d%H%M.nc")
        output_file = output_path / fname
        output_file.parent.mkdir(exist_ok=True, parents=True)
        encoding = {"surface_precip": {"dtype": np.float32, "zlib": True}}
        surface_precip_t.to_netcdf(output_file, encoding=encoding)


@click.argument('accumulate', type=int)
@click.argument('granularity', type=int)
@click.argument('year', type=int)
@click.argument('month', type=int)
@click.argument('days', nargs=-1, type=list, required=False)
@click.argument('output_path', type=click.Path(writable=True))
@click.option('--n_processes', default=1, type=int, help="Number of processes to use for downloading data.")
def extract_precip(
        accumulate: int,
        granularity: int,
        year: int,
        month: int,
        days: int,
        output_path: Path,
        n_processes: int = 1
) -> None:
    """
    Extract Stage IV precipitation data.

    Args:
        : The interval over which to average precipitation.
        year: The year
        month: The month
        day: The day
        output_path: A path object pointing to the directory to which to download the data.
    """
    if len(days) == 0:
        _, n_days = monthrange(year, month)
        days = list(range(1, n_days + 1))

    output_path = Path(output_path)
    output_folder = output_path
    output_folder.mkdir(exist_ok=True, parents=True)

    if n_processes > 1:
        LOGGER.info(f"Using {n_processes} processes for data extraction.")
        tasks = [(year, month, d, accumulate, granularity, output_path) for d in days]

        with ProcessPoolExecutor(max_workers=n_processes) as executor, Progress() as progress:
            task_id = progress.add_task("Extracting data:", total=len(tasks))
            future_to_task = {executor.submit(extract_stage4_precip, *task): task for task in tasks}
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
                    extract_stage4_precip(year, month, d, accumulate, granularity, output_path)
                except Exception as e:
                    LOGGER.exception(f"Error processing day {d}: {e}")
                finally:
                    progress.update(task_id, advance=1)
