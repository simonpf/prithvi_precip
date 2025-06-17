"""
prithvi_precip.data.mrms
=============

Functionality to extract reference precipitation estimates from MRMS measurements.
"""
from calendar import monthrange
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import List

import click
import numpy as np
from pansat.time import TimeRange, to_datetime
from pansat.products.ground_based import mrms
from rich.progress import Progress
from scipy.stats import binned_statistic_2d
import xarray as xr

from precipfm.definitions import LAT_BINS, LON_BINS


LOGGER = logging.getLogger(__name__)


def extract_mrms_precip(
        year: int,
        month: int,
        day: int,
        accumulate: int,
        granularity: int,
        output_path: Path
) -> None:
    """
    Download MRMS data and resample to MERRA grid and desired accumulations.

    Args:
        year: int specifying the year.
        month: int specifying the month
        day: int specifying the day.
        granularity: The time interval at which to extract files in hours.
        accumulate: The accumulation period in hours.
        output_path: The path to which to write the extracted files.
    """
    output_path = Path(output_path)

    start_time = datetime(year, month, day, minute=1) - timedelta(hours=accumulate)
    end_time = start_time + timedelta(hours=23, minutes=59) + timedelta(hours=accumulate)
    time_range = TimeRange(start_time, end_time)

    recs = mrms.precip_1h_ms.get(time_range)
    recs += mrms.precip_1h_gc.get(time_range)

    precip_fields = []
    time = []

    for rec in recs:

        rec_ro = mrms.precip_1h.get(rec.central_time)
        if len(rec_ro) > 0:
            data_ro = mrms.precip_1h.open(rec_ro[0])
            mask = (0.0 <= data_ro.precip_1h.data)
        else:
            mask = None


        if mrms.precip_1h_ms.matches(rec):
            data = mrms.precip_1h_ms.open(rec)
            surface_precip = data.precip_1h_ms.data
            if mask is not None:
                surface_precip[~mask] = np.nan
        else:
            data = mrms.precip_1h_gc.open(rec)
            surface_precip = data.precip_1h_gc.data
            if mask is not None:
                surface_precip[~mask] = np.nan

        lons = data.longitude.data
        lats = data.latitude.data
        lons, lats = np.meshgrid(lons, lats, indexing="xy")
        lons = lons
        lats = lats
        valid = 0.0 <= surface_precip
        surface_precip_r = binned_statistic_2d(
            lons[valid],
            lats[valid],
            surface_precip[valid],
            bins=(LON_BINS, LAT_BINS)
        )[0].T
        precip_fields.append(surface_precip_r)
        time.append(data.time.data - np.timedelta64(1, "h"))

    data = xr.Dataset({
        "latitude": 0.5 * (LAT_BINS[1:] + LAT_BINS[:-1]),
        "longitude": 0.5 * (LON_BINS[1:] + LON_BINS[:-1]),
        "time": np.stack(time),
        "surface_precip": (("time", "latitude", "longitude"), np.stack(precip_fields))
    })
    data = data.sortby("time")

    time_shifted = data.time[:-accumulate]
    data = data.rolling(time=accumulate, center=False).mean()[accumulate:]
    data = data.assign_coords(time=time_shifted)

    encoding = {"surface_precip": {"dtype": np.float32, "zlib": True}}
    for time_ind in range(data.time.size):
        data_t = data[{'time': time_ind}]
        time = to_datetime(data_t["time"].data)
        fname = time.strftime(f"{time.year:04}/{time.month:02}/mrms_%Y%m%d%H%M.nc")
        output_file = output_path / fname
        output_file.parent.mkdir(parents=True, exist_ok=True)
        data_t.to_netcdf(output_path / fname, encoding=encoding)


@click.argument('granularity', type=int)
@click.argument('accumulate', type=int)
@click.argument('year', type=int)
@click.argument('month', type=int)
@click.argument('days', nargs=-1, type=int, required=False)
@click.argument('output_path', type=click.Path(writable=True))
@click.option('--n_processes', default=1, type=int, help="Number of processes to use for extracting data.")
def extract_precip(
        granularity: int,
        accumulate: int,
        year: int,
        month: int,
        days: List[int],
        output_path: Path,
        n_processes: int
) -> None:
    """
    Extract MRMS data for evaluating the PrithviWxC.

    YEAR and MONTH are required. DAY is optional and defaults to extracting data for
    all days of the month.
    """
    if days:
        LOGGER.info(f"Extracting MRMS data for {year}-{month:02d} on days {', '.join(map(str, days))} to {output_path}.")
    else:
        LOGGER.info(f"Extracting MRMS data for all days in {year}-{month:02d} to {output_path}.")

    if len(days) == 0:
        _, n_days = monthrange(year, month)
        days = list(range(1, n_days + 1))

    if n_processes > 1:
        LOGGER.info(f"Using {n_processes} processes for downloading data.")
        tasks = [(year, month, d, granularity, accumulate, output_path) for d in days]

        with ProcessPoolExecutor(max_workers=n_processes) as executor, Progress() as progress:
            task_id = progress.add_task("Extracting data:", total=len(tasks))
            future_to_task = {executor.submit(extract_mrms_precip, *task): task for task in tasks}
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
                    extract_mrms_precip(year, month, d, granularity, accumulate, output_path)
                except Exception as e:
                    LOGGER.exception(f"Error processing day {d}: {e}")
                finally:
                    progress.update(task_id, advance=1)
