"""
prithvi_precip.data.merra2
==========================

Provides functionality to extract MERRA-2 training data for the Prithvi Precip model.
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Union

import numpy as np
from pansat import FileRecord, Geometry, TimeRange
from pansat.products.reanalysis.merra import MERRA2, MERRA2Constant
from pansat.time import to_datetime, to_datetime64
import xarray as xr


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


def download_dynamic(year: int, month: int, day: int, output_path: Path) -> None:
    """
    Download dynamic MERRA input data for a date given by year, month, and day.

    Args:
        year: The year
        day: The day
        output_path: A path object pointing to the directory to which to download the data.
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

    output_path = Path(output_path) / f"dynamic/{year:04}/{month:02}/{day:02}"
    output_path.mkdir(exist_ok=True, parents=True)

    encoding = {name: {"zlib": True} for name in data}

    for time_ind in range(data.time.size):
        data_t = data[{"time": time_ind}]
        date = to_datetime(data_t.time.data)
        output_file = date.strftime("merra2_%Y%m%d%H%M%S.nc")
        data_t.to_netcdf(output_path / output_file, encoding=encoding)


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


def download_static(output_path: Path) -> None:
    """
    Download static MERRA input data.

    Args:
        output_path: A path object pointing to the directory to which to download the data.
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

    output_path = Path(output_path) / "static"
    output_path.mkdir(exist_ok=True)

    output_file = output_path / "merra2_static.nc"
    encoding = {name: {"zlib": True} for name in data}
    data.to_netcdf(output_file, encoding=encoding)
