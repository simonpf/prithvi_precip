"""
prithvi_precip.obs.utils
========================

Utility functions for extracting observations.
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

from filelock import FileLock
import numpy as np
from pansat.time import to_datetime, to_datetime64
import xarray as xr
import yaml


POLARIZATIONS = {
    "NONE": 0,
    "H": 1,
    "V": 2,
    "QH": 3,
    "QV": 4
}


def encode_polarization(pol: str) -> int:
    """
    Encode polarization as integer.
    """
    return POLARIZATIONS[pol.upper()]


def update_tile_list(
        output_folder: Path,
        n_tiles: Tuple[int, int],
        row_index: int,
        col_index: int,
        filename: str
):
    """
    Add observation filename to tile list in folder.

    Args:
        output_folder: A Path object pointing to the directory to which to write the tile list.
        row_index: The row index of the tile.
        col_index: The column index of the tile.
        filename: The filename of the observation file.
    """
    output_folder = Path(output_folder)

    tile_file = output_folder / "tiles.yml"
    lock = FileLock(tile_file.with_suffix(".lock"))
    with lock:
        tiles = []
        if not tile_file.exists():
            for i_r in range(n_tiles[0]):
                t_c = []
                for i_c in range(n_tiles[1]):
                    t_c.append([])
                tiles.append(t_c)
        else:
            with open(tile_file, "r") as inpt:
                tiles = yaml.safe_load(inpt)

        tiles[row_index][col_index].append(filename)

        with open(tile_file, "w") as output:
            yaml.dump(tiles, output, default_flow_style=False, allow_unicode=True)


def round_time(time: np.datetime64 | datetime , step: np.timedelta64 | datetime) -> np.datetime64:
    """
    Round time to given time step.

    Args:
        time: A numpy.datetime64 object representing the time to round.
        step: A numpy.timedelta64 object representing the time step to
            which to round the results.
    """
    if isinstance(time, datetime):
        time = to_datetime64(time)
    if isinstance(step, timedelta):
        step = to_timedelta64(step)
    time = time.astype("datetime64[s]")
    step = step.astype("timedelta64[s]")
    rounded = (
        np.datetime64(0, "s")
        + time.astype(np.int64) // step.astype(np.int64) * step
    )
    return rounded


def lla_to_ecef(coords_lla: np.ndarray):
    """
    Converts latitude-longitude-altitude (LLA) coordinates to
    earth-centric earth-fixed coordinates (ECEF)

    Params:
        coords_lla: A numpy.ndarray containing the three coordinates oriented along the last axis.

    Return:
        coords_ecef: An array of the same shape as 'coords_lla' but containing the x, y, and z
             coordinates along the last axis.
    """
    SEM_A = 6_378_137.0
    SEM_B = 6_356_752.0
    ECC2 = 1.0 - (SEM_B ** 2 / SEM_A ** 2)

    lon = np.radians(coords_lla[..., 0])
    lat = np.radians(coords_lla[..., 1])
    alt = coords_lla[..., 2]

    roc = SEM_A / np.sqrt(1 - ECC2 * np.sin(lat)**2)

    x = (roc + alt) * np.cos(lat) * np.cos(lon)
    y = (roc + alt) * np.cos(lat) * np.sin(lon)
    z = (roc * (1 - ECC2) + alt) * np.sin(lat)

    return np.stack((x, y, z), -1)


def calculate_angles(
        fp_lons: np.ndarray,
        fp_lats: np.ndarray,
        sensor_lons: np.ndarray,
        sensor_lats: np.ndarray,
        sensor_alts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate zenith and azimuth angles describing the observation geometry.

    Args:
        fp_lons: Array containing the longitude coordinates of the observation
            footprints.
        fp_lats: Array containing the latitude coordinates of the observation
            footprints.
        sensor_lons: The longitude coordinates of the sensor.
        sensor_lats: The latitude coordinates of the sensor.
        sensor_alts: The altitude coordinates of the sensor.

    Return:
        A tuple ``zenith, azimuth, viewing_angle`` containing the zenith, azimuth, and
        sensor viewing angles in degree for all lines of sights.
    """
    sensor_lla = np.stack((sensor_lons, sensor_lats, sensor_alts), -1)
    sensor_ecef = lla_to_ecef(sensor_lla)
    sensor_down = np.stack((sensor_lons, sensor_lats, np.zeros_like(sensor_lons)), -1) - sensor_ecef
    sensor_down /= np.linalg.norm(sensor_down, axis=-1, keepdims=True)

    fp_lla = np.stack((fp_lons, fp_lats, np.zeros_like(fp_lons)), -1)
    fp_ecef = lla_to_ecef(fp_lla)
    local_up = fp_ecef / np.linalg.norm(fp_ecef, axis=-1, keepdims=True)
    fp_west = fp_lla.copy()
    fp_west[..., 0] -= 0.1
    fp_west = lla_to_ecef(fp_west) - fp_ecef
    fp_west /= np.linalg.norm(fp_west, axis=-1, keepdims=True)
    fp_north = fp_lla.copy()
    fp_north[..., 1] += 0.1
    fp_north = lla_to_ecef(fp_north) - fp_ecef
    fp_north /= np.linalg.norm(fp_north, axis=-1, keepdims=True)

    if sensor_ecef.ndim < fp_lla.ndim:
        sensor_ecef = np.broadcast_to(sensor_ecef[..., None, :], fp_lla.shape)
        sensor_down = np.broadcast_to(sensor_down[..., None, :], fp_lla.shape)

    los = sensor_ecef - fp_ecef
    los /= np.linalg.norm(los, axis=-1, keepdims=True)

    zenith = np.arccos((local_up * los).sum(-1))
    viewing_angle = np.arccos(-(los * sensor_down).sum(-1))

    azimuth = np.arctan2((los * fp_west).sum(-1), (los * fp_north).sum(-1))
    azimuth = np.nan_to_num(azimuth, nan=0.0)

    return np.rad2deg(zenith), np.rad2deg(azimuth), np.rad2deg(viewing_angle)


def get_output_path(time: np.datetime64):
    """
    Determine relative output path for a given time stamp.

    Args:
        time: A rounded time stamp for which to determine the relative sample folder.

    Return:
        A pathlib.Path object pointing to the in which to store the output file.
    """
    date = to_datetime(time)
    path = Path(date.strftime("%Y/%m/%d/"))
    return path


def track_stats(
        base_path: Path,
        variable_name: str,
        observations: np.ndarray
) -> int:
    """
    Track stats for a given observation source.

    Args:
        base_path: The base path to which observation tiles are extracted.
        variable_name: The name of the variable being tracked.
        observations: The observation data to track.

    Return:
        A variable that can be used to load the stats for the tracked variable.
    """
    base_path = Path(base_path)
    stats_file = base_path / "stats.nc"
    lock = FileLock(stats_file.with_suffix(".lock"))
    valid = np.isfinite(observations)
    obs = observations[valid]

    with lock:
        if not stats_file.exists():
            data = xr.Dataset()
        else:
            data = xr.load_dataset(stats_file)

        all_vars = data.attrs.get("variables", "").split(",")
        n_vars = len(all_vars)

        if variable_name not in all_vars:
            data[variable_name + "_sum"] = obs.sum()
            data[variable_name + "_squared_sum"] = (obs ** 2).sum()
            data[variable_name + "_counts"] = obs.size
            data[variable_name + "_min"] = obs.min()
            data[variable_name + "_max"] = obs.max()
            var_ind = n_vars
            all_vars.append(variable_name)
        else:
            try:
                data[variable_name + "_sum"] += obs.sum()
                data[variable_name + "_squared_sum"] += (obs ** 2).sum()
                data[variable_name + "_counts"] += obs.size
                data[variable_name + "_min"] = min(data[variable_name + "_min"], obs.min())
                data[variable_name + "_max"] = max(data[variable_name + "_max"], obs.max())
            except Exception:
                data[variable_name + "_sum"] = obs.sum()
                data[variable_name + "_squared_sum"] = (obs ** 2).sum()
                data[variable_name + "_counts"] = obs.size
                data[variable_name + "_min"] = obs.min()
                data[variable_name + "_max"] = obs.max()
            var_ind = all_vars.index(variable_name)

        data.attrs["variables"] = ",".join(all_vars)
        data.to_netcdf(stats_file)

        return var_ind


def split_tiles(data: xr.Dataset) -> xr.Dataset:
    """
    Split tiles into difference variables.
    """
    tiles_zonal = data.tiles_zonal
    tiles_meridional = data.tiles_meridional

    split = {}
    for tile_zonal, tile_meridional in zip(tiles_zonal, tiles_meridional):
        dim_name = f"tiles_{tile_meridional:02}_{tile_zonal:02}"
        tile_mask = (data.tiles_zonal == tile_zonal) * (data.tiles_meridional == tile_meridional)
        for var in ["observations", "frequency", "wavelength", "offset", "polarization", "obs_id", "time_offset"]:
            var_name = f"{var}_{tile_meridional:02}_{tile_zonal:02}"
            tiles = data[var][{"tiles": tile_mask}].rename(tiles=dim_name)

            if tiles.ndim == 3:
                tiles = tiles.transpose(dim_name, "lat_tile", "lon_tile")
                split[var_name] = ((dim_name, "lat_tile", "lon_tile"), tiles.data)
            else:
                split[var_name] = ((dim_name,), tiles.data)


    return xr.Dataset(split)


def combine_tiles(data: xr.Dataset, new_data: xr.Dataset) -> xr.Dataset:
    """
    Split tiles into difference variables.
    """
    variables = list(data.variables) + list(new_data.variables)

    full = {}
    for var in variables:
        if var in data:
            if var in new_data:
                full[var] = xr.concat(
                        (data[var], new_data[var]),
                        dim=data[var].dims[0]
                    )
            else:
                full[var] = data[var]
        else:
            if var in new_data:
                full[var] = new_data[var]

    return xr.Dataset(full)
