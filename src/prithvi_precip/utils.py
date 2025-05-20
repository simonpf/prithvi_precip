"""
prithvi_precip.utils
====================

Shared utility functions.
"""

from datetime import datetime
from functools import cache
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr


LOGGER = logging.getLogger(__name__)


LEVELS = [
    34.0,
    39.0,
    41.0,
    43.0,
    44.0,
    45.0,
    48.0,
    51.0,
    53.0,
    56.0,
    63.0,
    68.0,
    71.0,
    72.0,
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
    "Z0M",
]


STATIC_SURFACE_VARS = ["FRACI", "FRLAND", "FROCEAN", "PHIS"]


VERTICAL_VARS = ["CLOUD", "H", "OMEGA", "PL", "QI", "QL", "QV", "T", "U", "V"]


NAN_VALS = {
    "GWETROOT": 1.0,
    "LAI": 0.0,
}


@cache
def load_static_data(data_dir: Path) -> xr.Dataset:
    """
    Load static input data from root data path.

    Args:
        data_dir: A path object containing the input data

    Return:
        An xarray.Dataset containing the static MERRA data expected by the Prithvi-WxC model.
    """
    static_file = data_dir / "static" / "static.nc"
    return xr.load_dataset(static_file)


@cache
def load_position_signal(data_dir: Path) -> xr.Dataset:
    """
    Load longitude and latitude coordinates.

    Args:
        data_dir: A path object containing the input data

    Return:
        A numpy.ndarray of shape [2 x n_lat x n_lon] containing the latitude and
        longitude coordinates expected by the Prithvi-WxC model.
    """
    static_data = load_static_data(data_dir)
    lons = np.deg2rad(static_data.longitude.data)
    lats = np.deg2rad(static_data.latitude.data)
    lons = static_data.longitude.data
    lats = static_data.latitude.data
    lons, lats = np.meshgrid(lons, lats, indexing="xy")
    return np.stack([lats, lons], axis=0).astype(np.float32)


def load_climatology(time: np.datetime64, data_dir: Path) -> np.ndarray:
    """
    Load climatology data.

    Args:
         time: A timestamp defining the time for which to load the input data.
         data_dir: The root directory containing the Prithvi-WxC training data.

    Return:
         A numpy array containing the climatology data for the given time.
    """
    date = time.astype("datetime64[s]").item()
    year = date.year
    doy = (date - datetime(year=year, month=1, day=1)).days + 1
    # No climatology for leap years :(.
    doy = min(doy, 365)
    hod = date.hour

    sfc_file = data_dir / "climatology" / f"climate_surface_doy{doy:03}_hour{hod:02}.nc"
    data_sfc = []
    with xr.open_dataset(sfc_file) as sfc_data:
        for var in SURFACE_VARS:
            data_sfc.append(sfc_data[var].data.astype(np.float32))
    data_sfc = np.stack(data_sfc)

    data_vert = []
    vert_file = (
        data_dir / "climatology" / f"climate_vertical_doy{doy:03}_hour{hod:02}.nc"
    )
    with xr.open_dataset(vert_file) as vert_data:
        for var in VERTICAL_VARS:
            data_vert.append(np.flip(vert_data[var].data.astype(np.float32), 0))
    data_vert = np.stack(data_vert, 0)
    data_vert = data_vert.reshape(-1, *data_vert.shape[2:])

    data_combined = np.concatenate((data_sfc, data_vert), 0)
    return data_combined


def load_static_input(time: np.datetime64, data_dir: Path) -> np.ndarray:
    """
    Load all dynamic data from a given input file and return the data.

    Args:
        time: The time for which to load the static data.
        data_dir: The directory containing the training data.

    Return:
        A numpy ndarray containing all dynamic data for the given
    """
    LOGGER.debug("Loading static input from for time %s.", time)
    rel_time = time - time.astype("datetime64[Y]").astype(time.dtype)
    rel_time = np.datetime64("1980-01-01T00:00:00") + rel_time
    static_data = load_static_data(data_dir)
    static_data = static_data.interp(
        time=rel_time.astype("datetime64[ns]"),
        method="nearest",
        kwargs={"fill_value": "extrapolate"},
    )

    pos_sig = load_position_signal(data_dir)

    n_time = 4
    n_pos = pos_sig.shape[0]
    n_lat = pos_sig.shape[1]
    n_lon = pos_sig.shape[2]
    n_static_vars = len(STATIC_SURFACE_VARS)

    data = np.zeros((n_time + n_pos + n_static_vars, n_lat, n_lon))

    doy = time - time.astype("datetime64[Y]").astype(time.dtype)
    doy = doy.astype("timedelta64[D]").astype(int) + 1
    assert 0 <= doy <= 366

    hod = time - time.astype("datetime64[D]").astype(time.dtype)
    hod = hod.astype("timedelta64[h]").astype(int)
    assert 0 <= hod <= 24

    data[0:n_pos] = pos_sig
    data[n_pos + 0] = np.cos(2 * np.pi * doy / 366)
    data[n_pos + 1] = np.sin(2 * np.pi * doy / 366)
    data[n_pos + 2] = np.cos(2 * np.pi * hod / 24)
    data[n_pos + 3] = np.sin(2 * np.pi * hod / 24)

    for ind, var in enumerate(STATIC_SURFACE_VARS):
        data[n_pos + 4 + ind] = static_data[var].data

    return data
