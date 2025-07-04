"""
prithvi_precip.data.merra2
==========================

Provides functionality to extract MERRA-2 training data for the Prithvi Precip model.
"""
import click
from calendar import monthrange
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import List, Union

import numpy as np
try:
    from pansat import FileRecord, Geometry, TimeRange
    from pansat.products.reanalysis.merra import MERRA2, MERRA2Constant
    from pansat.time import to_datetime, to_datetime64
except ImportError:
    pass
from rich.progress import Progress
import xarray as xr

from prithvi_precip import domains


LOGGER = logging.getLogger(__name__)

LON_BINS = np.linspace(-180.3125, 179.6875, 577)
LAT_BINS = np.linspace(-90, 90, 361)

LEVELS = [
    34.0, 39.0, 41.0, 43.0, 44.0, 45.0, 48.0, 51.0, 53.0, 56.0, 63.0, 68.0, 71.0, 72.0
]
SURFACE_VARS = [
    "EFLUX",
    "GWETROOT",
    "HFLUX",
    "LAI",
    "LWGAB",
    "LWGEM",
    "LWTUP",
    "PS",
    "QV2M",
    "SLP",
    "SWGNT",
    "SWTNT",
    "T2M",
    "TQI",
    "TQL",
    "TQV",
    "TS",
    "U10M",
    "V10M",
    "Z0M"
]
STATIC_SURFACE_VARS = [
    "FRACI",
    "FRLAND",
    "FROCEAN",
    "PHIS"
]
VERTICAL_VARS = [
    "CLOUD",
    "H",
    "OMEGA",
    "PL",
    "QI",
    "QL",
    "QV",
    "T",
    "U",
    "V"
]
NAN_VALS = {
    "GWETROOT": 1.0,
    "LAI": 0.0,
}


try:
    m2i3nvasm = MERRA2(
        collection="m2i3nvasm",
    )
    m2i1nxasm = MERRA2(
        collection="m2i1nxasm",
    )
    m2t1nxlnd = MERRA2(
        collection="m2t1nxlnd",
        variables=[
            "GWETROOT", "LAI"
        ]
    )
    m2t1nxflx = MERRA2(
        collection="m2t1nxflx",
    )
    m2t1nxrad = MERRA2(
        collection="m2t1nxrad",
    )

    DYNAMIC_PRODUCTS = [
        m2i3nvasm,
        m2i1nxasm,
        m2t1nxlnd,
        m2t1nxflx,
        m2t1nxrad
    ]
except NameError:
    pass



def download_dynamic(
        year: int,
        month: int,
        day: int,
        output_path: Path,
        domain: str = "MERRA"
) -> None:
    """
    Download dynamic MERRA input data for a date given by year, month, and day.

    Args:
        year: The year
        day: The day
        output_path: A path object pointing to the directory to which to download the data.
        domain: Name of the domain for which to extract the data.
    """
    time_range = TimeRange(datetime(year, month, day, 12))
    merra_recs = []
    for prod in DYNAMIC_PRODUCTS:
        prod_recs = prod.get(time_range)
        assert len(prod_recs) == 1
        merra_recs.append(prod_recs)

    start_time = to_datetime64(datetime(year, month, day))
    end_time = start_time + np.timedelta64(1, "D")
    time_steps = np.arange(start_time, end_time, np.timedelta64(3, "h"))

    vars_req = VERTICAL_VARS + SURFACE_VARS

    all_data = []
    for recs in merra_recs:
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

        if not "time" in data:
            continue

        if (data.time.data[0] - data.time.data[0].astype("datetime64[h]")) > 0:
            for var in data:
                data[var].data[1:] = 0.5 * (data[var].data[1:] + data[var].data[:-1])
            new_time = data.time.data - 0.5 * (data.time.data[1] -  data.time.data[0])
            data = data.assign_coords(time=new_time)

        times = list(data.time.data)
        inds = [times.index(t_s) for t_s in time_steps]
        data_t = data[{"time": inds}]

        all_data.append(data_t)


    data = xr.merge(all_data, compat="override")
    data = data.rename(
        lat="latitude",
        lon="longitude"
    )

    if domain.upper() != "MERRA":
        lons, lats = domains.get_lonlats(domain)
        data = data.interp(longitude=lons, latitude=lats)

    output_path = Path(output_path) / f"dynamic/{year:04}/{month:02}/{day:02}"
    output_path.mkdir(exist_ok=True, parents=True)

    encoding = {name: {"zlib": True} for name in data}

    for time_ind in range(data.time.size):
        data_t = data[{"time": time_ind}]
        date = to_datetime(data_t.time.data)
        output_file = date.strftime("merra2_%Y%m%d%H%M%S.nc")
        data_t.to_netcdf(output_path / output_file, encoding=encoding)


try:
    m2conxasm = MERRA2Constant(
        collection="m2conxasm",
    )
    m2conxctm = MERRA2Constant(
        collection="m2conxctm",
    )
    STATIC_PRODUCTS = [
        m2conxasm,
        m2conxctm
    ]
except NameError:
    pass

@click.argument('year', type=int)
@click.argument('month', type=int)
@click.argument('days', nargs=-1, type=int, required=False)
@click.argument('output_path', type=click.Path(writable=True))
@click.option('--domain', type=str, default="MERRA")
@click.option('--n_processes', default=1, type=int, help="Number of processes to use for extracting data.")
def extract_merra_data(
        year: int,
        month: int,
        days: List[int],
        output_path: Path,
        domain: str,
        n_processes: int
) -> None:
    """
    Extract MERRA2 input data fo the PrithviWxC model.

    YEAR and MONTH are required. DAY is optional and defaults to extracting data for
    all days of the month.
    """
    if days:
        LOGGER.info(f"Extracting MERRA2  data for {year}-{month:02d} on days {', '.join(map(str, days))} to {output_path}.")
    else:
        LOGGER.info(f"Extracting MERRA2  data for all days in {year}-{month:02d} to {output_path}.")

    if len(days) == 0:
        _, n_days = monthrange(year, month)
        days = list(range(1, n_days + 1))

    if n_processes > 1:
        LOGGER.info(f"Using {n_processes} processes for downloading data.")
        tasks = [(year, month, day, output_path, domain) for d in days]

        with ProcessPoolExecutor(max_workers=n_processes) as executor, Progress() as progress:
            task_id = progress.add_task("Extracting data:", total=len(tasks))
            future_to_task = {executor.submit(download_dynamic, *task): task for task in tasks}
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
                    download_dynamic(year, month, d, output_path, domain)
                except Exception as e:
                    LOGGER.exception(f"Error processing day {d}: {e}")
                finally:
                    progress.update(task_id, advance=1)


def download_static(
        output_path: Path,
        domain: str = "MERRA"
) -> None:
    """
    Download static MERRA input data.

    Args:
        output_path: A path object pointing to the directory to which to download the data.
        domain: Name of the domain for which to extract the data.
    """
    time_range = datetime(2020, 1, 1)
    merra_recs = []
    for prod in STATIC_PRODUCTS:
        prod_recs = prod.get(time_range)
        assert len(prod_recs) == 1
        merra_recs += prod_recs

    vars_req = STATIC_SURFACE_VARS

    all_data = []
    for rec in merra_recs:
        with xr.open_dataset(rec.local_path) as data:
            vars = [
                var for var in vars_req if var in data.variables
            ]
            all_data.append(data.load())
    data = xr.merge(all_data)
    data = data.rename(
        lat="latitude",
        lon="longitude"
    )

    if domain.upper() != "MERRA":
        lons, lats = domains.get_lonlats(domain)
        data = data.interp(longitude=lons, latitude=lats)

    output_path = Path(output_path) / "static"
    output_path.mkdir(exist_ok=True)

    output_file = output_path / "merra2_static.nc"
    encoding = {name: {"zlib": True} for name in data}
    data.to_netcdf(output_file, encoding=encoding)


@click.argument('output_path', type=click.Path(writable=True))
@click.option('--domain', type=str, default="MERRA")
def extract_static_merra_data(
        output_path: Path,
        domain: str,
) -> None:
    """
    Extract MERRA2 input data fo the PrithviWxC model.

    YEAR and MONTH are required. DAY is optional and defaults to extracting data for
    all days of the month.
    """
    with Progress() as progress:
        task_id = progress.add_task("Extracting data:", total=len(days))
        for d in days:
            try:
                download_static(output_path, domain)
            except Exception as e:
                LOGGER.exception(f"Error processing day {d}: {e}")
            finally:
                progress.update(task_id, advance=1)
