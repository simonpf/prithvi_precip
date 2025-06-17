"""
Tests for the prithvi_precip.datasets module.
"""
from datetime import datetime
import os
from pathlib import Path

import numpy as np
import pytest

from PrithviWxC.dataloaders.merra2 import (
    Merra2Dataset,
    preproc
)
import torch
import xarray as xr


from prithvi_precip.datasets import (
    MERRAInputData,
    DirectPrecipForecastDataset
)

from prithvi_precip.data.merra2 import (
    SURFACE_VARS,
    VERTICAL_VARS,
    STATIC_SURFACE_VARS,
    VERTICAL_VARS,
    LEVELS
)


MERRA_DATA_PATH = os.environ.get("MERRA_DATA", None)
HAS_MERRA_DATA = MERRA_DATA_PATH is not None

PRITHVI_DATA_PATH = os.environ.get("PRITHVI_DATA", None)
HAS_PRITHVI_DATA = PRITHVI_DATA_PATH is not None


@pytest.mark.skipif(not HAS_MERRA_DATA, reason="MERRA input data not available.")
def test_merra_input_data():
    """
    Test that available input files for MERRA data are parsed correctly.
    """
    lead_times = [3, 6]
    dataset = MERRAInputData(
        MERRA_DATA_PATH,
        input_time=3,
        lead_times=lead_times
    )

    times_in_1 = dataset.times[dataset.input_indices[:, 0]]
    times_in_2 = dataset.times[dataset.input_indices[:, 1]]
    t_d = (times_in_2 - times_in_1).astype("timedelta64[h]").astype("int64")
    assert np.all(np.isclose(t_d, 3))

    times_out_1 = dataset.times[dataset.output_indices[:, 0]]
    times_out_2 = dataset.times[dataset.output_indices[:, 1]]
    t_d = (times_in_2 - times_in_1).astype("timedelta64[h]").astype("int64")
    assert np.all(np.isclose(t_d, 3))

    times_out_1 = dataset.times[dataset.input_indices[:, 1]]
    times_out_2 = dataset.times[dataset.output_indices[:, 0]]
    t_d = (times_in_2 - times_in_1).astype("timedelta64[h]").astype("int64")
    assert np.all(np.isclose(t_d, 3))

    lead_times = [-3]
    dataset = MERRAInputData(
        MERRA_DATA_PATH,
        input_time=3,
        lead_times=lead_times
    )

    input_files = dataset.input_files[dataset.input_indices[:, 0]]
    output_files = dataset.input_files[dataset.output_indices[:, 0]]
    assert np.all(input_files == output_files)


@pytest.mark.skipif(not HAS_MERRA_DATA, reason="MERRA inptu data not available.")
def test_load_sample():
    """
    Test that available input files for MERRA data are parsed correctly.
    """
    lead_times = [3, 6]
    dataset = MERRAInputData(
        MERRA_DATA_PATH,
        input_time=3,
        lead_times=lead_times
    )
    x, y = dataset[0]

    assert "x" in x
    assert x["x"].shape == (2, 160, 360, 576)
    assert "climate" in x
    assert x["climate"].shape == (160, 360, 576)
    assert "static" in x
    assert x["static"].shape == (10, 360, 576)
    assert y.shape == (160, 360, 576)


def load_data_prithvi(data_path: Path, ind: int):
    """
    Load input data using original Prithvi implementation.
    """
    surf_dir = data_path / "merra-2"
    vert_dir = data_path / "merra-2"
    surf_clim_dir = data_path / "climatology"
    vert_clim_dir = data_path / "climatology"
    surface_vars = [
        "EFLUX", "GWETROOT", "HFLUX", "LAI", "LWGAB", "LWGEM", "LWTUP", "PS", "QV2M",
        "SLP", "SWGNT", "SWTNT", "T2M", "TQI", "TQL", "TQV", "TS", "U10M", "V10M", "Z0M",
    ]
    static_surface_vars = ["FRACI", "FRLAND", "FROCEAN", "PHIS"]
    vertical_vars = ["CLOUD", "H", "OMEGA", "PL", "QI", "QL", "QV", "T", "U", "V"]
    levels = [34.0, 39.0, 41.0, 43.0, 44.0, 45.0, 48.0, 51.0, 53.0, 56.0, 63.0, 68.0, 71.0, 72.0,]
    lead_times = [6]
    input_times = [-6]
    time_range = ("2020-01-01T00:00:00", "2020-01-01T23:59:59")
    positional_encoding = "fourier"
    dataset = Merra2Dataset(
        time_range=time_range,
        lead_times=lead_times,
        input_times=input_times,
        data_path_surface=surf_dir,
        data_path_vertical=vert_dir,
        climatology_path_surface=surf_clim_dir,
        climatology_path_vertical=vert_clim_dir,
        surface_vars=surface_vars,
        static_surface_vars=static_surface_vars,
        vertical_vars=vertical_vars,
        levels=levels,
        positional_encoding=positional_encoding,
    )
    data = dataset[ind]

    padding = {"level": [0, 0], "lat": [0, -1], "lon": [0, 0]}
    return preproc([data], padding)


@pytest.mark.skipif(not HAS_MERRA_DATA, reason="MERRA input data not available.")
@pytest.mark.skipif(not HAS_PRITHVI_DATA, reason="PRITHVI input data not available.")
def test_loaded_data():
    input_times = [-6, 0]
    lead_times = [6]
    dataset = MERRAInputData(
        MERRA_DATA_PATH,
        input_time=6,
        lead_times=lead_times,
        center_meridionally=False
    )
    x, y = dataset[0]

    inpt_ref = load_data_prithvi(Path(PRITHVI_DATA_PATH), 0)

    assert torch.all(torch.isclose(x["x"], inpt_ref["x"][0]))
    assert torch.all(torch.isfinite(x["x"]))
    assert torch.all(torch.isclose(y, inpt_ref["y"][0]))
    assert torch.all(torch.isclose(x["static"], inpt_ref["static"][0]))
    assert torch.all(torch.isclose(x["input_time"], inpt_ref["input_time"][0]))
    assert torch.all(torch.isclose(x["lead_time"], inpt_ref["lead_time"][0]))
    assert torch.all(torch.isclose(x["climate"], inpt_ref["climate"][0]))


@pytest.mark.skipif(not HAS_MERRA_DATA, reason="MERRA input data not available.")
def test_get_forecast_input_static():
    lead_times = [6]
    dataset = MERRAInputData(
        MERRA_DATA_PATH,
        input_time=6,
        lead_times=lead_times
    )

    static_data = dataset.get_forecast_input_static(np.datetime64("2020-01-01T06:00:00"), 4)
    assert static_data.shape == (4, 10, 360, 576)


@pytest.mark.skipif(not HAS_MERRA_DATA, reason="MERRA input data not available.")
def test_get_forecast_input_climate():
    input_times = [-6, 0]
    lead_times = [6]
    dataset = MERRAInputData(
        MERRA_DATA_PATH,
        input_time=3,
        lead_times=lead_times
    )
    climate_data = dataset.get_forecast_input_climate(np.datetime64("2020-01-01T06:00:00"), 2)
    assert climate_data.shape == (2, 160, 360, 576)


def create_file_dynamic(path: Path, year: int, month: int, day: int, hour: int):
    """
    Create a dummy MERRA2 training data file containing the day of the year in the surface variables
    and the hour of the day in the vertical variables.
    """
    data = xr.Dataset()
    for var in SURFACE_VARS:
        data[var] = (("latitude", "longitude"), day * np.ones((360, 576)))
    for var in VERTICAL_VARS:
        data[var] = (("levels", "latitude", "longitude"), hour * np.ones((len(LEVELS), 360, 576)))
    output_path = path / "dynamic" / f"{year}" / f"{month:02}" / f"{day:02}" / f"merra2_{year}{month:02}{day:02}{hour:02}0000.nc"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(output_path)


def create_file_static(path: Path):
    """
    Create a  MERRA2 static data file containing the day of the year in the surface variables
    and the hour of the day in the vertical variables.
    """
    data = xr.Dataset()
    for var in STATIC_SURFACE_VARS:
        data[var] = (("time", "latitude", "longitude"), np.arange(12)[:, None, None] * np.ones((12, 360, 576)))
    data["time"] = (
        ("time",),
        np.arange(
            np.datetime64("1980-01-01T00:00:00", "M"),
            np.datetime64("1981-01-01T00:00:00", "M"),
            np.timedelta64(1, "M")
        )
    )
    output_path = path / "static" / "static.nc"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(output_path)


def create_file_climatology(path: Path, year: int, month: int, day: int, hour: int):
    """
    Create PrithviWxC climatology files.
    """
    start_of_year = datetime(year=year, month=1, day=1)
    day_of_year = datetime(year=year, month=month, day=day)
    doy = (day_of_year - start_of_year).days + 1

    data_surf = xr.Dataset()
    for var in SURFACE_VARS:
        data_surf[var] = (("latitude", "longitude"), day * np.ones((360, 576)))
    output_path = path / "climatology" / f"climate_surface_doy{doy:03}_hour{hour:02}.nc"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_surf.to_netcdf(output_path)

    data_vert = xr.Dataset()
    for var in VERTICAL_VARS:
        data_vert[var] = (("levels", "latitude", "longitude"), hour * np.ones((len(LEVELS), 360, 576)))
    output_path = path / "climatology" / f"climate_vertical_doy{doy:03}_hour{hour:02}.nc"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_vert.to_netcdf(output_path)

def create_file_imerg(path: Path, accumulation_period: int, year: int, month: int, day: int, hour: int):
    """
    Create a dummy IMERG training data file containing the hour of the day as precipitation values so that
    the loaded data can be used to verify that the correct data is loaded.
    """
    data = xr.Dataset()
    data["surface_precip"] = (("latitude", "longitude"), hour * np.ones((360, 576)))
    output_path = path / f"imerg_{accumulation_period}" / f"{year}" / f"{month:02}" / f"{day:02}" / f"imerg_{year}{month:02}{day:02}{hour:02}0000.nc"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(output_path)


@pytest.fixture(scope="session")
def imerg_training_data_1(tmp_path_factory):
    """
    Create dummy training data for precipitation forecasts.
    """
    training_data_path = tmp_path_factory.mktemp("training_data")
    data_path = training_data_path.parent

    create_file_static(data_path)
    for hour in range(0, 24, 3):
        create_file_climatology(data_path, 2020, 1, 1, hour)
        create_file_dynamic(training_data_path, 2020, 1, 1, hour)
        create_file_imerg(training_data_path, 1, 2020, 1, 1, hour)

    return training_data_path

@pytest.fixture(scope="session")
def imerg_training_data_3(tmp_path_factory):
    """
    Create dummy training data for precipitation forecasts.
    """
    base_dir = tmp_path_factory.mktemp("training_data")

    create_file_static(base_dir)
    for hour in range(0, 24, 3):
        create_file_dynamic(base_dir, 2020, 1, 1, hour)
        create_file_climatology(base_dir, 2020, 1, 1, hour)
        create_file_imerg(base_dir, 3, 2020, 1, 1, hour)

    return base_dir


def test_direct_precip_forecast_dataset(imerg_training_data_1):
    """
    Test that direct precipitation forecast dataset loads the right time step data.
    """
    static_files = sorted(list(imerg_training_data_1.glob("static/2020/01/01/*.nc")))

    ds = DirectPrecipForecastDataset(
        imerg_training_data_1,
        accumulation_period=1,
        max_steps=3,
    )
    assert len(ds) == 6

    x, y = ds[0]
    assert torch.isclose(x["static"][6:], torch.tensor(0.0)).all()
    cos_doy = x["static"][2]
    assert torch.isclose(cos_doy, torch.cos(2 * np.pi * torch.tensor(1) / 366), atol=1e-3).all()
    sin_doy = x["static"][3]
    assert torch.isclose(sin_doy, torch.sin(2 * np.pi * torch.tensor(1) / 366), atol=1e-3).all()
    cos_hod = x["static"][4]
    assert torch.isclose(cos_hod, torch.cos(2 * np.pi * torch.tensor(3) / 24), atol=1e-3).all()
    sin_hod = x["static"][5]
    assert torch.isclose(sin_hod, torch.sin(2 * np.pi * torch.tensor(3) / 24), atol=1e-3).all()

    assert torch.isclose(x["x"][:, :20], torch.tensor(1.0)).all()
    assert torch.isclose(x["x"][0, 20:], torch.tensor(0.0)).all()
    assert torch.isclose(x["x"][1, 20:], torch.tensor(3.0)).all()
    assert (torch.tensor(6.0) <= y[0]).all()
    assert (y[0] <= torch.tensor(12.0)).all()
    assert torch.isclose(y[0], x["lead_time"] + 3).all()
    assert (3 <= x["lead_time"]).all()
    assert (x["lead_time"] <= 9).all()

    x, y = ds[1]
    assert torch.isclose(x["static"][6:], torch.tensor(0.0)).all()
    assert torch.isclose(x["x"][:, :20], torch.tensor(1.0)).all()
    assert torch.isclose(x["x"][0, 20:], torch.tensor(3.0)).all()
    assert torch.isclose(x["x"][1, 20:], torch.tensor(6.0)).all()
    assert (torch.tensor(9.0) < y[0]).all()
    assert (y[0] <= torch.tensor(15.0)).all()
    assert torch.isclose(y[0], x["lead_time"] + 6).all()
    assert (3 <= x["lead_time"]).all()
    assert (x["lead_time"] <= 9).all()
