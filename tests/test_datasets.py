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


def test_merra_input_data(merra_dataset_structure, sample_dataset_config):
    """
    Test that available input files for MERRA data are parsed correctly.
    """
    start_date = datetime(2023, 1, 1)
    structure = merra_dataset_structure(start_date, n_timesteps=10)
    dataset = MERRAInputData(
        training_data_path=structure['base_path'] / "training_data",
        input_time=sample_dataset_config['input_time'],
        lead_times=[3, 6],
        climatology=True,
        center_meridionally=sample_dataset_config['center_meridionally']
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

    dataset = MERRAInputData(
        training_data_path=structure['base_path'] / "training_data",
        input_time=sample_dataset_config['input_time'],
        lead_times=[-3],
        climatology=True,
        center_meridionally=sample_dataset_config['center_meridionally']
    )
    input_files = dataset.input_files[dataset.input_indices[:, 0]]
    output_files = dataset.input_files[dataset.output_indices[:, 0]]
    assert np.all(input_files == output_files)


def test_load_sample(merra_dataset_structure, sample_dataset_config):
    """
    Test that available input files for MERRA data are parsed correctly.
    """
    start_date = datetime(2023, 1, 1)
    structure = merra_dataset_structure(start_date, n_timesteps=10)
    dataset = MERRAInputData(
        training_data_path=structure['base_path'] / "training_data",
        input_time=sample_dataset_config['input_time'],
        lead_times=[3, 6],
        climatology=True,
        center_meridionally=sample_dataset_config['center_meridionally']
    )
    x, y = dataset[0]

    assert "x" in x
    assert x["x"].shape == (2, 160, 360, 576)
    assert "climate" in x
    assert x["climate"].shape == (160, 360, 576)
    assert "static" in x
    assert x["static"].shape == (10, 360, 576)
    assert y.shape == (160, 360, 576)


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
