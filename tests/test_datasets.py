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
    DirectPrecipForecastDataset,
    AutoregressivePrecipForecastDataset,
    ObservationLoader
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


def test_direct_precip_forecast_dataset(imerg_training_data_1):
    """
    Test that direct precipitation forecast dataset loads the right time step data.
    """
    ds = DirectPrecipForecastDataset(
        imerg_training_data_1,
        accumulation_period=1,
        max_steps=3,
    )
    assert len(ds) == 7

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
    assert (torch.tensor(3.0) <= y[0]).all()
    assert (y[0] <= torch.tensor(12.0)).all()
    assert torch.isclose(y[0], x["lead_time"] + 3).all()
    assert (3 <= x["lead_time"]).all()
    assert (x["lead_time"] <= 9).all()

    x, y = ds[1]
    assert torch.isclose(x["static"][6:], torch.tensor(0.0)).all()
    assert torch.isclose(x["x"][:, :20], torch.tensor(1.0)).all()
    assert torch.isclose(x["x"][0, 20:], torch.tensor(3.0)).all()
    assert torch.isclose(x["x"][1, 20:], torch.tensor(6.0)).all()
    assert (torch.tensor(6.0) <= y[0]).all()
    assert (y[0] <= torch.tensor(15.0)).all()
    assert torch.isclose(y[0], x["lead_time"] + 6).all()
    assert (3 <= x["lead_time"]).all()
    assert (x["lead_time"] <= 9).all()


def test_autoregressive_precip_forecast_dataset(imerg_training_data_1):
    """
    Test that direct precipitation forecast dataset loads the right time step data.
    """
    ds = AutoregressivePrecipForecastDataset(
        imerg_training_data_1,
        scaling_factors = imerg_training_data_1 / "scaling_factors",
        accumulation_period=1,
        max_steps=3,
    )
    assert len(ds) == 6

    x, y = ds[0]

    # Check first static tensor
    assert torch.isclose(x["static"][0, 6:], torch.tensor(0.0)).all()
    cos_doy = x["static"][0, 2]
    assert torch.isclose(cos_doy, torch.cos(2 * np.pi * torch.tensor(1) / 366), atol=1e-3).all()
    sin_doy = x["static"][0, 3]
    assert torch.isclose(sin_doy, torch.sin(2 * np.pi * torch.tensor(1) / 366), atol=1e-3).all()
    cos_hod = x["static"][0, 4]
    assert torch.isclose(cos_hod, torch.cos(2 * np.pi * torch.tensor(3) / 24), atol=1e-3).all()
    sin_hod = x["static"][0, 5]
    assert torch.isclose(sin_hod, torch.sin(2 * np.pi * torch.tensor(3) / 24), atol=1e-3).all()

    # Check first static tensor
    assert torch.isclose(x["static"][0, 6:], torch.tensor(0.0)).all()
    cos_doy = x["static"][1, 2]
    assert torch.isclose(cos_doy, torch.cos(2 * np.pi * torch.tensor(1.0) / 366), atol=1e-3).all()
    sin_doy = x["static"][1, 3]
    assert torch.isclose(sin_doy, torch.sin(2 * np.pi * torch.tensor(1.0) / 366), atol=1e-3).all()
    cos_hod = x["static"][1, 4]
    assert torch.isclose(cos_hod, torch.cos(2 * np.pi * torch.tensor(6.0) / 24), atol=1e-3).all()
    sin_hod = x["static"][1, 5]
    assert torch.isclose(sin_hod, torch.sin(2 * np.pi * torch.tensor(6.0) / 24), atol=1e-3).all()

    # Check second static tensor
    assert torch.isclose(x["static"][0, 6:], torch.tensor(0.0)).all()
    cos_doy = x["static"][1, 2]
    assert torch.isclose(cos_doy, torch.cos(2 * np.pi * torch.tensor(1.0) / 366), atol=1e-3).all()
    sin_doy = x["static"][1, 3]
    assert torch.isclose(sin_doy, torch.sin(2 * np.pi * torch.tensor(1.0) / 366), atol=1e-3).all()
    cos_hod = x["static"][1, 4]
    assert torch.isclose(cos_hod, torch.cos(2 * np.pi * torch.tensor(6.0) / 24), atol=1e-3).all()
    sin_hod = x["static"][1, 5]
    assert torch.isclose(sin_hod, torch.sin(2 * np.pi * torch.tensor(6.0) / 24), atol=1e-3).all()

    # Check climate tensors
    assert torch.isclose(x["climate"][0, :20], torch.tensor(1.0)).all() # Surface vars should contain the day
    assert torch.isclose(x["climate"][0, 20:], torch.tensor(6.0)).all() # Vertical vars should contain the hour
    assert torch.isclose(x["climate"][1, :20], torch.tensor(1.0)).all() # Surface vars should contain the day
    assert torch.isclose(x["climate"][1, 20:], torch.tensor(9.0)).all() # Vertical vars should contain the hour
    assert torch.isclose(x["climate"][2, :20], torch.tensor(1.0)).all() # Surface vars should contain the day
    assert torch.isclose(x["climate"][2, 20:], torch.tensor(12.0)).all() # Vertical vars should contain the hour

    assert torch.isclose(x["x"][:, :20], torch.tensor(1.0)).all()
    assert torch.isclose(x["x"][0, 20:], torch.tensor(0.0)).all()
    assert torch.isclose(x["x"][1, 20:], torch.tensor(3.0)).all()
    assert (torch.tensor(6.0) <= y["surface_precip"][0]).all()

    assert torch.isclose(y["surface_precip"][0], torch.tensor(6.0)).all()
    assert torch.isclose(y["surface_precip"][1], torch.tensor(9.0)).all()
    assert torch.isclose(y["surface_precip"][2], torch.tensor(12.0)).all()

    assert (x["lead_time"] == 3.0).all()
    assert (x["input_time"] == 3.0).all()


def test_observation_loader(imerg_training_data_3):
    """
    Test loading observations using the ObservationLoader
    """
    obs_loader = ObservationLoader(
        imerg_training_data_3 / "obs"
    )
    obs, meta = obs_loader.load_observations(np.datetime64("2020-01-01T12:00:00"))

    assert obs.shape == (12, 18, 32, 1, 30, 32)

    obs = np.unique(obs.numpy())
    assert (-1.0 + 2.0 / 300.0) in obs
