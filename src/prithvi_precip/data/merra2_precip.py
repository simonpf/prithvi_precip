"""
prithvi_precip.data.merra2_precip
=================================

Provides an interface to extract surface precipitation estimates from MERRA2.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from calendar import monthrange
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import List, Union

import click
import numpy as np
from pansat import FileRecord, Geometry, TimeRange
from pansat.products.reanalysis.merra import MERRA2
from pansat.time import to_datetime, to_datetime64
from rich.progress import Progress
import xarray as xr


LOGGER = logging.getLogger(__name__)


m2t1nxflx = MERRA2(
    collection="m2t1nxflx",
)


def extract_merra_precip(
        year: int,
        month: int,
        day: int,
        accumulate: int,
        granularity: int,
        output_path: Path
) -> None:
    """
    Extract MERRA precip fields for a date given by year, month, and day.

    Args:
        year: The year
        month: The month
        day: The day
        accumulate: The period over which to accumulate results.
        ganularity: The frequency in hours at which to extract samples.
        output_path: A path object pointing to the directory to which to download the data.
    """
    output_path = Path(output_path)

    start_time = datetime(year, month, day) - timedelta(hours=accumulate)
    end_time = start_time + timedelta(days=1, hours=accumulate)
    time_range = TimeRange(start_time, end_time)
    merra_recs = m2t1nxflx.get(time_range)

    all_data = []
    for rec in merra_recs:
        with xr.open_dataset(rec.local_path) as data:
            data = data[["PRECTOT"]].load().rename(PRECTOT="surface_precip")
            data["surface_precip"].data *= 3.6e3
            all_data.append(data)

    start_time = to_datetime64(datetime(year, month, day))
    end_time = start_time + np.timedelta64(1, "D")
    time_steps = np.arange(start_time, end_time, np.timedelta64(3, "h"))


    data = xr.concat(all_data, "time").sortby("time")
    time_shifted = data.time[:-accumulate]
    data = data.rolling(time=accumulate, center=False).mean()[{"time": slice(None, -accumulate)}]
    data = data.assign_coords(time=time_shifted)

    encoding = {"surface_precip": {"dtype": np.float32, "zlib": True}}

    start_time = np.datetime64(f"{year}-{month:02}-{day:02}T00:00:00")
    end_time = start_time + np.timedelta64(1, "D")
    for time in np.arange(start_time, end_time, np.timedelta64(granularity, "h")):
        data_t = data.interp(time=time.astype("datetime64[ns]"), method="nearest")
        time = to_datetime(data_t["time"].data)
        fname = time.strftime(f"merra2_precip_{accumulate}/{time.year:04}/{time.month:02}/merra2_precip_%Y%m%d%H%M.nc")
        output_file = output_path / fname
        output_file.parent.mkdir(parents=True, exist_ok=True)
        data_t.to_netcdf(output_path / fname, encoding=encoding)


@click.argument('accumulate', type=int)
@click.argument('granularity', type=int)
@click.argument('year', type=int)
@click.argument('month', type=int)
@click.argument('days', nargs=-1, type=int, required=False)
@click.argument('output_path', type=click.Path(writable=True))
@click.option('--n_processes', default=1, type=int, help="Number of processes to use for extracting data.")
def extract_precip(
        accumulate: int,
        granularity: int,
        year: int,
        month: int,
        days: List[int],
        output_path: Path,
        n_processes: int
) -> None:
    """
    Extract MERRA2 surface precipitation data for finetuning the PrithviWxC.

    YEAR and MONTH are required. DAY is optional and defaults to extracting data for
    all days of the month.
    """
    if days:
        LOGGER.info(f"Extracting MERRA2 precipitation data for {year}-{month:02d} on days {', '.join(map(str, days))} to {output_path}.")
    else:
        LOGGER.info(f"Extracting MERRA2 precipitation data for all days in {year}-{month:02d} to {output_path}.")

    if len(days) == 0:
        _, n_days = monthrange(year, month)
        days = list(range(1, n_days + 1))

    if n_processes > 1:
        LOGGER.info(f"Using {n_processes} processes for downloading data.")
        tasks = [(year, month, d, accumulate, granularity,  output_path) for d in days]

        with ProcessPoolExecutor(max_workers=n_processes) as executor, Progress() as progress:
            task_id = progress.add_task("Extracting data:", total=len(tasks))
            future_to_task = {executor.submit(extract_merra_precip, *task): task for task in tasks}
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
                    extract_merra_precip(year, month, d, accumulate, granularity,  output_path)
                except Exception as e:
                    LOGGER.exception(f"Error processing day {d}: {e}")
                finally:
                    progress.update(task_id, advance=1)
