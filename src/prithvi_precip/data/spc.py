"""
prithvi_precip.data.spc
=======================

Functionality to extract Prithvi Precip training data derived from the severe weather
database of the Storm Prediction Center.
"""
from calendar import monthrange
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
from io import BytesIO, StringIO
import logging
from pathlib import Path
import zipfile

import click
import numpy as np
import pandas as pd
from pansat.time import to_datetime
import requests
from rich.progress import Progress
from scipy.stats import binned_statistic_2d
import xarray as xr

from .merra2 import LAT_BINS, LON_BINS


LOGGER = logging.getLogger(__name__)


@cache
def get_data(url: str) -> pd.DataFrame:
    """
    Download and load CSV data.

    Parameters:
        url (str): URL to the potentially zipped CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    response = requests.get(url)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")

    if "zip" in content_type or url.endswith(".zip"):
        with zipfile.ZipFile(BytesIO(response.content)) as zfile:
            for filename in zfile.namelist():
                with zfile.open(filename) as f:
                    data = pd.read_csv(f)
                    break
    else:
        csv_content = StringIO(response.text)
        data = pd.read_csv(csv_content)

    valid = (data.tz == 3)
    data = data[valid]
    time_cst = pd.to_datetime(data['date'] + ' ' + data['time'])
    time_cst = time_cst.dt.tz_localize(
        'America/Chicago',
        ambiguous="NaT",
        nonexistent="NaT"
        )
    data['time'] = time_cst.dt.tz_convert('UTC')
    return data

@cache
def get_tornado_data() -> pd.DataFrame:
    """
    Downloads th SPC tornado database from the given URL and returns a pandas DataFrame.

    Parameters:
        url (str): URL to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    return get_data("https://www.spc.noaa.gov/wcm/data/1950-2024_torn.csv.zip")


@cache
def get_hail_data() -> pd.DataFrame:
    """ Downloads the SPC hail database from the given URL and returns a pandas DataFrame.

    Parameters:
        url (str): URL to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    return get_data("https://www.spc.noaa.gov/wcm/data/1955-2024_hail.csv.zip")


@cache
def get_wind_data() -> pd.DataFrame:
    """
    Downloads the SPC damaging wind database from the given URL and returns a pandas DataFrame.

    Parameters:
        url (str): URL to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    return get_data("https://www.spc.noaa.gov/wcm/data/1955-2024_wind.csv.zip")


def extract_severe_weather(
        year: int,
        month: int,
        day: int,
        accumulate: int,
        granularity: int,
        output_path: Path,
) -> None:
    """
    Extract sever weather data and resample to MERRA grid.

    Args:
        year: int specifying the year.
        month: int specifying the month
        day: int specifying the day.
        granularity: The frequency in hours at which to extract samples.
        accumulate: The period over which precipitation is accumulated.
        output_path: The path to which to write the extracted files.
    """
    start_times = np.arange(
        np.datetime64(f"{year}-{month:02}-{day:02}T00:00:00"),
        np.datetime64(f"{year}-{month:02}-{day:02}T00:00:00") + np.timedelta64(24, "h"),
        np.timedelta64(granularity, "h")
    )

    for start_time in start_times:

        start_time = pd.Timestamp(start_time, tz='UTC')
        end_time = start_time + np.timedelta64(accumulate, "h")

        tornado = get_tornado_data().copy()
        hail = get_hail_data().copy()
        wind = get_wind_data().copy()
        wind['mag'] = wind['mag'] * 1.15


        mask = (start_time <= tornado.time) * (tornado.time < end_time)
        tornado = tornado[mask]
        mask = (start_time <= hail.time) * (hail.time < end_time)
        hail = hail[mask]
        mask = (start_time <= wind.time) * (wind.time < end_time)
        wind = wind[mask]

        tornado_mask = 0 <= tornado.mag
        hail_mask = 1 <= hail.mag
        wind_mask = 58 <= wind.mag

        tornado = tornado.loc[tornado_mask]
        hail = hail.loc[hail_mask]
        wind = wind.loc[wind_mask]

        bins = (LON_BINS, LAT_BINS)
        lons = tornado.slon
        lats = tornado.slat
        if 0 < lons.size:
            tornado_mask_r = binned_statistic_2d(lons, lats, tornado_mask.astype(np.float32), "count", bins=bins)[0].T
        else:
            tornado_mask_r = np.zeros((LAT_BINS.size - 1, LON_BINS.size - 1))
        lons = hail.slon
        lats = hail.slat
        if 0 < lats.size:
            hail_mask_r = binned_statistic_2d(lons, lats, hail_mask.astype(np.float32), "count", bins=bins)[0].T
        else:
            hail_mask_r = np.zeros((LAT_BINS.size - 1, LON_BINS.size - 1))

        lons = wind.slon
        lats = wind.slat
        if 0 < lats.size:
            wind_mask_r = binned_statistic_2d(lons, lats, wind_mask.astype(np.float32), "count", bins=bins)[0].T
        else:
            wind_mask_r = np.zeros((LAT_BINS.size - 1, LON_BINS.size - 1))

        lons = 0.5 * (LON_BINS[1:] + LON_BINS[:-1])
        lats = 0.5 * (LAT_BINS[1:] + LAT_BINS[:-1])

        result = xr.Dataset({
            "latitude": (("latitude",), lats),
            "longitude": (("longitude",), lons),
            "tornado": (("latitude", "longitude"), tornado_mask_r),
            "hail": (("latitude", "longitude"), hail_mask_r),
            "wind": (("latitude", "longitude"), wind_mask_r),
        })

        date = to_datetime(start_time)
        fname = date.strftime(f"severe_weather_{accumulate}/%Y/%m/%d/severe_weather_%Y%m%d%H%M.nc")
        output_file = output_path / fname
        output_file.parent.mkdir(exist_ok=True, parents=True)
        encoding = {
            "hail": {"dtype": "int8", "zlib": True, "_FillValue": -1},
            "tornado": {"dtype": "int8", "zlib": True, "_FillValue": -1},
            "wind": {"dtype": "int8", "zlib": True, "_FillValue": -1},
        }
        result.to_netcdf(fname, encoding=encoding)


@click.argument('accumulate', type=int)
@click.argument('granularity', type=int)
@click.argument('year', type=int)
@click.argument('month', type=int)
@click.argument('days', nargs=-1, type=list, required=False)
@click.argument('output_path', type=click.Path(writable=True))
@click.option('--n_processes', default=1, type=int, help="Number of processes to use for downloading data.")
def extract_training_data(
        accumulate: int,
        granularity: int,
        year: int,
        month: int,
        days: int,
        output_path: Path,
        n_processes: int = 1
) -> None:
    """
    Extract SPC report data.

    Args:
        accumulate: The interval over which to average precipitation.
        granularity: The interval for which to extract samples.
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
            future_to_task = {executor.submit(extract_severe_weather, *task): task for task in tasks}
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
                    extract_severe_weather(year, month, d, accumulate, granularity, output_path)
                except Exception as e:
                    LOGGER.exception(f"Error processing day {d}: {e}")
                finally:
                    progress.update(task_id, advance=1)
