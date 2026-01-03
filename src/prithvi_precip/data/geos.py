"""
prithvi_precip.data.geos
========================

Provides an interface to download GEOS reanalysis and forecast data.
"""
from calendar import monthrange
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import List

import click
import numpy as np
from pansat.time import TimeRange, to_datetime64, to_datetime
from pansat.products.model.geos import (
    inst3_3d_asm_nv,
    inst3_2d_asm_nx,
    tavg1_2d_lnd_nx,
    tavg1_2d_flx_nx,
    tavg1_2d_rad_nx,
    tavg1_2d_flx_nx_fc,
)
from rich.progress import Progress
import xarray as xr

from .merra2 import LEVELS, SURFACE_VARS, VERTICAL_VARS, NAN_VALS


LOGGER = logging.getLogger(__name__)


DYNAMIC_PRODUCTS = [
    inst3_3d_asm_nv,
    inst3_2d_asm_nx,
    tavg1_2d_lnd_nx,
    tavg1_2d_flx_nx,
    tavg1_2d_rad_nx,
]


def download_dynamic(
        year: int, month: int, day: int, output_path: Path) -> None:
    """
    Download dynamic GEOS input data for a date given by year, month, and day.

    Args:
        year: The year
        day: The day
        output_path: A path object pointing to the directory to which to download the data.
    """
    start_time = datetime(year, month, day)
    time_range = TimeRange(start_time, start_time + timedelta(hours=23, minutes=59))
    geos_recs = []
    for prod in DYNAMIC_PRODUCTS:
        prod_recs = prod.get(time_range)
        geos_recs.append(prod_recs)

    start_time = to_datetime64(datetime(year, month, day))
    end_time = start_time + np.timedelta64(1, "D")
    time_steps = np.arange(start_time, end_time, np.timedelta64(3, "h"))

    vars_req = VERTICAL_VARS + SURFACE_VARS

    all_data = []
    for recs in geos_recs:
        data_combined = []
        for rec in recs:
            with xr.open_dataset(rec.local_path) as data:
                vars = [
                    var for var in vars_req if var in data.variables
                ]
                data = data[vars + ["time"]]
                if "lev" in data:
                    data = data.loc[{"lev": np.array(LEVELS)}]
                data_combined.append(data.load())
        data = xr.concat(data_combined, "time").sortby("time")

        for var in data:
            if var in NAN_VALS:
                nan = NAN_VALS[var]
                data[var].data[:] = np.nan_to_num(data[var].data, nan=nan)


        if (data.time.data[0] - data.time.data[0].astype("datetime64[h]")) > 0:
            for var in data:
                data[var].data[1:] = 0.5 * (data[var].data[1:] + data[var].data[:-1])
            new_time = data.time.data - 0.5 * (data.time.data[1] -  data.time.data[0])
            data = data.assign_coords(time=new_time)

        times = list(data.time.data)
        time_steps = [step for step in time_steps if step in times]
        inds = [times.index(t_s) for t_s in time_steps]
        data_t = data[{"time": inds}]

        all_data.append(data_t)


    data = xr.merge(all_data, compat="override")
    data = data.rename(
        lat="latitude",
        lon="longitude"
    )
    data = data.coarsen({"longitude": 2}).mean()
    data_n = data[{"latitude": 720}]
    data = data[{"latitude": slice(0, -1)}].coarsen({"latitude": 2}).mean()
    data = xr.concat((data, data_n), "latitude")

    output_path = Path(output_path) / "dynamic" / f"{year:04}/{month:02}/{day:02}"
    output_path.mkdir(exist_ok=True, parents=True)

    encoding = {name: {"zlib": True} for name in data}

    for time_ind in range(data.time.size):
        data_t = data[{"time": time_ind}]
        date = to_datetime(data_t.time.data)
        output_file = date.strftime("geos_%Y%m%d%H%M%S.nc")
        data_t.to_netcdf(output_path / output_file, encoding=encoding)


def download_precip(
        year: int,
        month: int,
        day: int,
        output_path: Path
) -> None:
    """
    Download GEOS precipitation analysis for a date given by year, month, and day.

    Args:
        year: The year
        day: The day
        output_path: A path object pointing to the directory to which to download the data.
    """
    start_time = datetime(year, month, day)
    time_range = TimeRange(start_time, start_time + timedelta(hours=23, minutes=59))
    recs = tavg1_2d_flx_nx.get(time_range)

    start_time = to_datetime64(datetime(year, month, day))
    end_time = start_time + np.timedelta64(1, "D")
    time_steps = np.arange(start_time, end_time, np.timedelta64(3, "h"))

    vars_req = ["PRECTOT"]

    all_data = []
    data_combined = []
    for rec in recs:
        with xr.open_dataset(rec.local_path) as data:
            vars = [
                var for var in vars_req if var in data.variables
            ]
            data = data[vars + ["time"]]
            data_combined.append(data.load())
    data = xr.concat(data_combined, "time").sortby("time")

    if (data.time.data[0] - data.time.data[0].astype("datetime64[h]")) > 0:
        for var in data:
            data[var].data[1:] = 0.5 * (data[var].data[1:] + data[var].data[:-1])
        new_time = data.time.data - 0.5 * (data.time.data[1] -  data.time.data[0])
        data = data.assign_coords(time=new_time)

    times = list(data.time.data)
    time_steps = [step for step in time_steps if step in times]
    inds = [times.index(t_s) for t_s in time_steps]
    data_t = data[{"time": inds}]

    all_data.append(data_t)


    data = xr.merge(all_data, compat="override")
    data = data.rename(
        lat="latitude",
        lon="longitude"
    )
    data = data.coarsen({"longitude": 2}).mean()
    data_n = data[{"latitude": 720}]
    data = data[{"latitude": slice(0, -1)}].coarsen({"latitude": 2}).mean()
    data = xr.concat((data, data_n), "latitude")

    output_path = Path(output_path) / "geos_precip" / f"{year:04}/{month:02}/{day:02}"
    output_path.mkdir(exist_ok=True, parents=True)

    encoding = {name: {"zlib": True} for name in data}

    for time_ind in range(data.time.size):
        data_t = data[{"time": time_ind}]
        date = to_datetime(data_t.time.data)
        output_file = date.strftime("geos_precip_%Y%m%d%H%M%S.nc")
        data_t.to_netcdf(output_path / output_file, encoding=encoding)



def download_geos_forecast(
        init_time: np.datetime64,
        output_path: Path
        ):
    """
    Download GEOS forecast for a given initialization time.

    Args:
        init_time: The initialization time.

    """
    LOGGER.info(
        "Extracting forecasts for initialization time %s.",
        init_time
    )
    geos_recs = tavg1_2d_flx_nx_fc.find_files(
        TimeRange(init_time + np.timedelta64(3, "h"))
    )

    if len(geos_recs) == 0:
        LOGGER.info(
            "No forecasts found for initialization time %s.",
            init_time
        )
        return None

    geos_data = []
    for rec in geos_recs:
        try:
            rec = rec.get()
        except Exception as exc:
            LOGGER.warning(
                "Error downloading file record %s", rec
            )
            return None
        with xr.open_dataset(rec.local_path) as data:
            data = data[["PRECTOT"]].compute().rename({
                "PRECTOT": "surface_precip",
                "lon": "longitude",
                "lat": "latitude"
            })
            data["surface_precip"].data *= 3.6e3
            data = data.coarsen({"longitude": 2}).mean()
            data_n = data[{"latitude": 720}]
            data = data[{"latitude": slice(0, -1)}].coarsen({"latitude": 2}).mean()
            geos_data.append(data)

    geos_data = xr.concat(geos_data, dim="time")
    geos_data = geos_data.sortby("time")
    geos_data = geos_data.resample(time="3h").mean()

    init_time = to_datetime(init_time)
    filename = init_time.strftime("geos_forecast_%Y%m%d_%H.nc")
    geos_data.to_netcdf(output_path / filename)


@click.argument('year', type=int)
@click.argument('month', type=int)
@click.argument('days', nargs=-1, type=int, required=False)
@click.argument('output_path', type=click.Path(writable=True))
def download_geos_forecasts(
        year: int,
        month: int,
        days: List[int],
        output_path: Path
) -> None:
    """
    Download GEOS precipitation forecasts results for a given day and year.

    Args:
        year: The year, if set to negative value will download forecasts from the previous day.
        day: The day
        output_path: A path object pointing to the directory to which to download the data.
    """
    if year < 1000:
        today = datetime.today() - timedelta(days=1)
        year = today.year
        month = today.month
        day = [today.day]

    if days:
        LOGGER.info(f"Extracting data for {year}-{month:02d} on days {', '.join(map(str, days))} to {output_path}.")
    else:
        LOGGER.info(f"Extracting data for all days in {year}-{month:02d} to {output_path}.")


    if len(days) == 0:
        _, n_days = monthrange(year, month)
        days = list(range(1, n_days + 1))


    init_times = []
    for day in days:
        start_time = datetime(year, month, day)
        end_time = start_time + timedelta(hours=23, minutes=59)
        start_time = to_datetime64(start_time)
        end_time = to_datetime64(end_time)
        init_times.append(np.arange(start_time, end_time, np.timedelta64(12, "h")))

    init_times = np.concatenate(init_times, axis=0)


    for init_time in init_times:

        date = init_time.astype("datetime64[s]").item()
        output_path = Path(output_path)
        output_folder = output_path / f"{date.year:04}" / f"{date.month:02}" / f"{date.day:02}"
        output_folder.mkdir(exist_ok=True, parents=True)

        try:
            download_geos_forecast(init_time, output_folder)
        except Exception:
            LOGGER.exception(
                "Encoutered an error when processing initialization time %s.",
                init_time
            )


@click.argument('year', type=int)
@click.argument('month', type=int)
@click.argument('days', nargs=-1, type=int, required=False)
@click.argument('output_path', type=click.Path(writable=True))
@click.option('--n_processes', default=1, type=int, help="Number of processes to use for downloading data.")
def extract_geos_data(
        year: int,
        month: int,
        days: List[int],
        output_path: Path,
        n_processes: int = 1
) -> None:
    """
    Extract data for a given YEAR, MONTH, and optional DAY, and write output to to OUTPUT_PATH.

    YEAR and MONTH are required. DAY is optional and defaults to extracting data for
    all days of the month.
    """
    if days:
        LOGGER.info(f"Extracting data for {year}-{month:02d} on days {', '.join(map(str, days))} to {output_path}.")
    else:
        LOGGER.info(f"Extracting data for all days in {year}-{month:02d} to {output_path}.")


    if len(days) == 0:
        _, n_days = monthrange(year, month)
        days = list(range(1, n_days + 1))


    if n_processes > 1:
        LOGGER.info(f"Using {n_processes} processes for downloading data.")
        tasks = [(year, month, d, output_path) for d in days]

        with ProcessPoolExecutor(max_workers=n_processes) as executor, Progress() as progress:
            task_id = progress.add_task("Extracting data:", total=len(tasks))
            future_to_task = {executor.submit(download_dynamic, *task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    future.result()
                except Exception as e:
                    logger.exception(f"Task {task} failed with error: {e}")
                finally:
                    progress.update(task_id, advance=1)
    else:
        with Progress() as progress:
            task_id = progress.add_task("Extracting data:", total=len(days))
            for d in days:
                try:
                    download_dynamic(year, month, d, output_path)
                except Exception as e:
                    LOGGER.exception(f"Error processing day {d}: {e}")
                finally:
                    progress.update(task_id, advance=1)


@click.argument('year', type=int)
@click.argument('month', type=int)
@click.argument('days', nargs=-1, type=int, required=False)
@click.argument('output_path', type=click.Path(writable=True))
@click.option('--n_processes', default=1, type=int, help="Number of processes to use for downloading data.")
def extract_geos_precip_data(
        year: int,
        month: int,
        days: List[int],
        output_path: Path,
        n_processes: int = 1
) -> None:
    """
    Extract data for a given YEAR, MONTH, and optional DAY, and write output to to OUTPUT_PATH.

    YEAR and MONTH are required. DAY is optional and defaults to extracting data for
    all days of the month.
    """
    if days:
        LOGGER.info(f"Extracting data for {year}-{month:02d} on days {', '.join(map(str, days))} to {output_path}.")
    else:
        LOGGER.info(f"Extracting data for all days in {year}-{month:02d} to {output_path}.")


    if len(days) == 0:
        _, n_days = monthrange(year, month)
        days = list(range(1, n_days + 1))


    if n_processes > 1:
        LOGGER.info(f"Using {n_processes} processes for downloading data.")
        tasks = [(year, month, d, output_path) for d in days]

        with ProcessPoolExecutor(max_workers=n_processes) as executor, Progress() as progress:
            task_id = progress.add_task("Extracting data:", total=len(tasks))
            future_to_task = {executor.submit(download_precip, *task): task for task in tasks}
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
                    download_precip(year, month, d, output_path)
                except Exception as e:
                    LOGGER.exception(f"Error processing day {d}: {e}")
                finally:
                    progress.update(task_id, advance=1)
