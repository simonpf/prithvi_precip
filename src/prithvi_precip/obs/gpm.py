"""
prithvi_precip.obs.gpm
======================

This module provides functionality to extract satellite observations from the sensors of the
GPM constellation.
"""
from calendar import monthrange
import click
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from filelock import FileLock
import logging
from pathlib import Path
import re
from typing import Tuple

from scipy.constants import speed_of_light
import numpy as np
from pansat import FileRecord
from pansat.time import to_datetime64, TimeRange
from pansat.utils import resample_data
from pansat.products.satellite.gpm import (
    l1c_xcal2021v_f16_ssmis_v07a,
    l1c_xcal2021v_f16_ssmis_v07b,
    l1c_xcal2021v_f17_ssmis_v07a,
    l1c_xcal2021v_f17_ssmis_v07b,
    l1c_xcal2019v_noaa20_atms_v07a,
    l1c_xcal2019v_npp_atms_v07a,
    l1c_xcal2016v_noaa19_mhs_v07a,
    l1c_xcal2016v_noaa18_mhs_v07a,
    l1c_xcal2019v_metopc_mhs_v07a,
    l1c_xcal2016v_metopb_mhs_v07a,
    l1c_r_xcal2016c_gpm_gmi_v07a,
    l1c_r_xcal2016c_gpm_gmi_v07b,
    l1c_xcal2016v_gcomw1_amsr2_v07a
)
from pyresample.geometry import AreaDefinition
from rich.progress import Progress
from tqdm import tqdm
import xarray as xr

from .. import domains
from .utils import (
    encode_polarization,
    calculate_angles,
    get_output_path,
    round_time,
    track_stats,
    split_tiles,
    combine_tiles
)


LOGGER = logging.getLogger(__name__)


_CHANNEL_REGEXP = re.compile("([\d\.]+)\s*(?:GHz)?(?:\+-)?\s*(?:\+\/-)?\s*([\d\.]*)\s*(?:GHz)?\s*(\w+)-Pol")


SSMIS_PRODUCTS = [
    l1c_xcal2021v_f16_ssmis_v07a,
    l1c_xcal2021v_f16_ssmis_v07b,
    l1c_xcal2021v_f17_ssmis_v07a,
    l1c_xcal2021v_f17_ssmis_v07b,
]

SENSORS = {
    "ssmis": [
        ("f16", 60e3, [l1c_xcal2021v_f16_ssmis_v07a, l1c_xcal2021v_f16_ssmis_v07b]),
        ("f17", 60e3, [l1c_xcal2021v_f17_ssmis_v07a, l1c_xcal2021v_f17_ssmis_v07b]),
    ],
    "atms": [
        ("noaa20", 60e3, [l1c_xcal2019v_noaa20_atms_v07a]),
        ("snpp", 60e3, [l1c_xcal2019v_npp_atms_v07a]),
    ],
    "mhs": [
        ("noaa19", 60e3, [l1c_xcal2016v_noaa19_mhs_v07a]),
        ("noaa18", 60e3, [l1c_xcal2016v_noaa18_mhs_v07a]),
        ("metop_b", 60e3, [l1c_xcal2016v_metopb_mhs_v07a]),
        ("metop_c", 60e3, [l1c_xcal2019v_metopc_mhs_v07a]),
    ],
    "gmi": [
        ("gpm", 20e3, [l1c_r_xcal2016c_gpm_gmi_v07a, l1c_r_xcal2016c_gpm_gmi_v07b]),
    ],
    "amsr2": [
        ("gcomw1", 20e3, [l1c_xcal2016v_gcomw1_amsr2_v07a]),
    ],
}


def extract_observations(
        base_path: Path,
        gpm_file: FileRecord,
        domain: str,
        platform_name: str,
        sensor_name: str,
        radius_of_influence: float,
        tile_dims: Tuple[int, int] = (12, 18),
        time_step = np.timedelta64(3, "h"),
) -> None:
    """
    Extract observations from a GPM L1C file.

    Args:
        base_path: Path object pointing to the directory to which to write the extracted observations.
        gpm_file: A pansat FileRecord pointing to a GPM input file.
        domain: The name of the target domain.
        platform_name: The name of the satellite platform the sensor is on.
        sensor_name: The name of the sensor
        radius_of_influence: Radius of influence to use for resampling.
        tile_dims: A tuple defining the number of tiles.
        time_step: The time interval over which to integrate observations.
    """
    target_grid = getattr(domains, domain.upper())

    l1c_obs = gpm_file.product.open(gpm_file)
    if "latitude" in l1c_obs:
        vars = [
            "latitude", "longitude", "tbs", "channels"
        ]
        l1c_obs = l1c_obs.rename(dict([(name, name + "_s1") for name in vars]))

    swath_ind = 1
    while f"latitude_s{swath_ind}" in l1c_obs:
        freqs = []
        offsets = []
        pols = []

        for match in _CHANNEL_REGEXP.findall(l1c_obs[f"tbs_s{swath_ind}"].attrs["LongName"]):
            freq, offs, pol = match
            freqs.append(float(freq))
            if offs == "":
                offsets.append(0.0)
            else:
                offsets.append(float(offs))
            pols.append(pol)

        swath_data = l1c_obs[[
            f"longitude_s{swath_ind}",
            f"latitude_s{swath_ind}",
            f"tbs_s{swath_ind}",
            f"channels_s{swath_ind}",
            "scan_time"
        ]].reset_coords("scan_time")

        fp_lons = swath_data[f"longitude_s{swath_ind}"].data
        fp_lats = swath_data[f"latitude_s{swath_ind}"].data
        sensor_lons = l1c_obs["spacecraft_longitude"].data
        sensor_lats = l1c_obs["spacecraft_latitude"].data
        sensor_alt = l1c_obs["spacecraft_altitude"].data * 1e3
        zenith, azimuth, viewing_angle = calculate_angles(
            fp_lons,
            fp_lats,
            sensor_lons,
            sensor_lats,
            sensor_alt
        )
        sensor_alt = np.broadcast_to(sensor_alt[..., None], zenith.shape) / 100e3

        swath_data = swath_data.rename({
            f"longitude_s{swath_ind}": "longitude",
            f"latitude_s{swath_ind}": "latitude"
        })
        swath_data["sensor_alt"] = (("scans", "pixels"), sensor_alt)
        swath_data["zenith"] = (("scans", "pixels"), zenith)
        swath_data["azimuth"] = (("scans", "pixels"), azimuth)
        swath_data["viewing_angle"] = (("scans", "pixels"), viewing_angle)

        try:
            swath_data_r = resample_data(
                swath_data,
                target_grid,
                radius_of_influence=radius_of_influence,
            ).transpose("latitude", "longitude", ...)
        except ValueError:
            return None
        sensor_alt = swath_data_r.sensor_alt.data
        zenith = swath_data_r.zenith.data
        azimuth = swath_data_r.azimuth.data
        viewing_angle = swath_data_r.viewing_angle.data

        start_time = swath_data.scan_time.min().data
        end_time = swath_data.scan_time.max().data
        start_time = round_time(start_time, time_step)
        end_time = round_time(end_time, time_step)
        times = np.arange(start_time, end_time + time_step, time_step)

        otime = swath_data_r.scan_time.data
        valid = np.isfinite(otime)

        for time in times:
            for chan_ind, (freq, offset, pol) in enumerate(zip(freqs, offsets, pols)):

                mask = (
                    (time <= swath_data_r.scan_time.data) *
                    (swath_data_r.scan_time.data <= (time + time_step))
                )
                if mask.sum() == 0:
                    continue

                obs = swath_data_r[f"tbs_s{swath_ind}"].data[..., chan_ind].copy()
                obs[obs < 0] = np.nan
                obs[obs > 400] = np.nan

                # Calculate relative time in seconds
                rel_time = (swath_data_r.scan_time.data - time).astype("timedelta64[m]").astype("float32")

                output = xr.Dataset({
                    "observations": (("y", "x"), obs),
                    "time_offset": (("y", "x"), rel_time),
                })

                output["observations"].data[~mask] = np.nan
                #output["observation_relative_time"].data[~mask] = np.datetime64("NAT")
                #output["observation_zenith_angle"].data[~mask] = np.nan
                #output["observation_azimuth_angle"].data[~mask] = np.nan

                uint16_max = 2 ** 16 - 1
                encoding = {
                    "observations": {"dtype": "uint16", "scale_factor": 0.01, "_FillValue": uint16_max, "zlib": True},
                    "observation_relative_time": {"dtype": "uint16", "_FillValue": uint16_max, "zlib": True},
                    "observation_zenith_angle": {"dtype": "uint16", "scale_factor": 0.01, "_FillValue": uint16_max, "zlib": True},
                    "observation_azimuth_angle": {"dtype": "uint16", "scale_factor": 0.01, "_FillValue": uint16_max, "zlib": True},
                }

                n_rows, n_cols = output.observations.data.shape

                obs_name = f"{platform_name}_{sensor_name}_{freq:.02f}_{offset:.02}_{pol.lower()}"

                valid = np.isfinite(output.observations.data)
                if valid.sum() == 0:
                    continue
                obs_id = track_stats(base_path, obs_name, output.observations.data)

                output = output.coarsen({"x": tile_dims[0], "y": tile_dims[1]})
                output = output.construct({
                    "x": ("tiles_zonal", "lon_tile"),
                    "y": ("tiles_meridional", "lat_tile")
                })
                output = output.stack(tiles=("tiles_meridional", "tiles_zonal"))
                output = output.transpose("tiles", "lat_tile", "lon_tile")
                valid_tiles = np.isfinite(output.observations).mean(("lon_tile", "lat_tile")) > 0.25
                output = output[{"tiles": valid_tiles}].reset_index("tiles")

                tot_tiles = output.tiles.size
                ones = np.ones(tot_tiles)
                height = output.lat_tile.size
                width = output.lon_tile.size

                frequency = freq
                wavelength = speed_of_light / (freq * 1e9) * 1e6,
                polarization = encode_polarization(pol)
                output["wavelength"] = (("tiles",), wavelength * ones)
                output["frequency"] = (("tiles",), frequency * ones)
                output["offset"] = (("tiles",), offset * ones)
                output["polarization"] = (("tiles",), polarization * ones.astype(np.int8))
                output["obs_id"] = (("tiles",), obs_id * ones.astype(np.int8))

                new_data = split_tiles(output)

                filename = start_time.astype("datetime64[s]").item().strftime("obs_%Y%m%d%H%M%S.nc")
                output_path = base_path / get_output_path(start_time) / filename

                lock = FileLock(output_path.with_suffix(".lock"))
                with lock:

                    if output_path.exists():
                        existing = xr.load_dataset(output_path)
                        new_data = combine_tiles(existing, new_data)

                    new_data = split_tiles(output)
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

                    new_data.to_netcdf(output_path, encoding=encoding)

        swath_ind += 1


def extract_observations_day(
        sensor: str,
        year: int,
        month: int,
        day: int,
        output_path: Path,
        domain: str,
        tile_dims: Tuple[int, int]
) -> None:
    """
    Extract GPM observations for a given day.

    Args:
        sensor: The name of the sensor.
        year: Integer specfiying the year.
        month: Integer specfiying the month.
        day: Integer specfiying the day.
        output_path: A path object pointing to the directory to which to write the extracted observations.
        domain: The name of the target domain.
        tile_dims: A tuple defining the number of tiles in the domain.
    """
    sensors = SENSORS[sensor]
    for platform_name, roi, pansat_products in sensors:
        for pansat_product in pansat_products:
            start = datetime(year, month, day)
            end = start + timedelta(days=1)
            recs = pansat_product.get(TimeRange(start, end))
            for rec in recs:
                try:
                    extract_observations(output_path, rec, domain, platform_name, sensor, roi, tile_dims=tile_dims)
                except:
                    LOGGER.exception(
                        "Encountered an error when processing input file %s.",
                        rec.local_path
                    )


@click.argument('sensor_name', type=str)
@click.argument('year', type=int)
@click.argument('month', type=int)
@click.argument('output_path', type=click.Path())
@click.option("--n_processes", help="The number of process to use for the data extraction.", default=1)
@click.option("--domain", help="The target domain.", default="MERRA")
@click.option("--tile_dims", help="The number of tiles", default="32,32")
def process_sensor_data(
        sensor_name,
        year,
        month,
        output_path,
        n_processes:int = 1,
        domain: str = "MERRA",
        tile_dims: str = "30,32"
):
    """
    Process sensor data for a given sensor, year, and month, and save the result to the specified output path.

    SENSOR_NAME: Name of the sensor (string)
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
        LOGGER.info(f"Using {n_processes} processes for downloading data.")
        tasks = [(sensor_name, year, month, d, output_path, domain, tile_dims) for d in days]

        with ProcessPoolExecutor(max_workers=n_processes) as executor, Progress() as progress:
            task_id = progress.add_task("Extracting data:", total=len(tasks))
            future_to_task = {executor.submit(extract_observations_day, *task): task for task in tasks}
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
                    extract_observations_day(sensor_name, year, month, d, output_path, domain, tile_dims)
                except Exception as e:
                    LOGGER.exception(f"Error processing day {d}: {e}")
                finally:
                    progress.update(task_id, advance=1)
