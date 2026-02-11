"""
Tests for the prithvi_precip.forecast module.
"""
from typing import Dict

import numpy as np
from pytorch_retrieve.tensors import MeanTensor
import torch
from torch import nn
from torch.utils.data import DataLoader
import xarray as xr

from prithvi_precip.forecast import (
    AutoregressiveForecastLoader,
    DirectForecastLoader,
    run_direct_forecast,
    run_autoregressive_forecast
)


def test_direct_forecast_loader(imerg_training_data_1):
    """
    Test that the direct forecast loader loads the expected data.
    """
    ds = DirectForecastLoader(
        imerg_training_data_1,
        init_times = np.arange(
            np.datetime64('2020-01-01T03:00:00'),
            np.datetime64("2020-01-01T07:00:00"),
            np.timedelta64(3, "h")
        ),
        n_steps=4,
        batch_size=2,
        observation_layers=16,
        full_climatology=True
    )
    assert len(ds) == 6

    init_time, valid_times, input_data = ds[0]
    assert init_time == np.datetime64("2020-01-01T03:00:00")
    val_times_ref = np.arange(
        np.datetime64("2020-01-01T06:00:00"),
        np.datetime64("2020-01-01T12:00:00"),
        np.timedelta64(3, "h"),
    )
    assert (valid_times == val_times_ref).all()

    # Static data should be duplicated across batch.
    assert torch.isclose(input_data["static"][6:], torch.tensor(0.0)).all()
    cos_doy = input_data["static"][0, 2]
    assert torch.isclose(cos_doy, torch.cos(2 * np.pi * torch.tensor(1) / 366), atol=1e-3).all()
    sin_doy = input_data["static"][0, 3]
    assert torch.isclose(sin_doy, torch.sin(2 * np.pi * torch.tensor(1) / 366), atol=1e-3).all()
    cos_hod = input_data["static"][0, 4]
    assert torch.isclose(cos_hod, torch.cos(2 * np.pi * torch.tensor(3) / 24), atol=1e-3).all()
    sin_hod = input_data["static"][0, 5]
    assert torch.isclose(sin_hod, torch.sin(2 * np.pi * torch.tensor(3) / 24), atol=1e-3).all()
    cos_doy = input_data["static"][1, 2]
    assert torch.isclose(cos_doy, torch.cos(2 * np.pi * torch.tensor(1) / 366), atol=1e-3).all()
    sin_doy = input_data["static"][1, 3]
    assert torch.isclose(sin_doy, torch.sin(2 * np.pi * torch.tensor(1) / 366), atol=1e-3).all()
    cos_hod = input_data["static"][1, 4]
    assert torch.isclose(cos_hod, torch.cos(2 * np.pi * torch.tensor(3) / 24), atol=1e-3).all()
    sin_hod = input_data["static"][1, 5]
    assert torch.isclose(sin_hod, torch.sin(2 * np.pi * torch.tensor(3) / 24), atol=1e-3).all()

    assert "obs" in input_data
    assert "obs_meta" in input_data
    vals = np.unique(input_data["obs"].numpy())
    assert np.all(np.isfinite(vals))
    assert -1.5 in vals
    assert (-1.0 + 2.0 * 1 / 300.0) in vals

    # Dynamic input should be init time and previous timestep.
    assert torch.isclose(input_data["x"][:, 0, :20], torch.tensor(1.0)).all()
    assert torch.isclose(input_data["x"][:, 0, 20:], torch.tensor(0.0)).all()
    assert torch.isclose(input_data["x"][:, 1, :20], torch.tensor(1.0)).all()
    assert torch.isclose(input_data["x"][:, 1, 20:], torch.tensor(3.0)).all()

    # Climate data should contain target hours
    assert torch.isclose(input_data["climate"][0, :20], torch.tensor(1.0)).all()
    assert torch.isclose(input_data["climate"][0, 20:], torch.tensor(6.0)).all()
    assert torch.isclose(input_data["climate"][1, :20], torch.tensor(1.0)).all()
    assert torch.isclose(input_data["climate"][1, 20:], torch.tensor(9.0)).all()

    assert (torch.isclose(input_data["lead_time"], torch.tensor([3.0, 6.0]))).all()
    assert (torch.isclose(input_data["input_time"], torch.tensor(3.0))).all()

    init_time, valid_times, input_data = ds[2]
    assert init_time is None
    assert valid_times is None
    assert input_data is None


def test_autoregressive_forecast_loader(imerg_training_data_1):
    """
    Test that the autoregressive forecast loader loads the expected data.
    """
    ds = AutoregressiveForecastLoader(
        imerg_training_data_1,
        init_times = np.arange(
            np.datetime64('2020-01-01T03:00:00'),
            np.datetime64("2020-01-01T07:00:00"),
            np.timedelta64(3, "h")
        ),
        n_steps=4,
        batch_size=2,
        full_climatology=True
    )
    assert len(ds) == 1

    init_times, valid_times, input_data = ds[0]

    init_times_ref = np.arange(
        np.datetime64("2020-01-01T03:00:00"),
        np.datetime64("2020-01-01T09:00:00"),
        np.timedelta64(3, "h"),
    )
    assert (init_times == init_times_ref).all()

    valid_times_ref = init_times_ref[:, None] + np.arange(1, 5) * np.timedelta64(3, "h")
    assert (valid_times == valid_times_ref).all()

    assert (torch.isclose(input_data["lead_time"], torch.tensor(3.0))).all()
    assert (torch.isclose(input_data["input_time"], torch.tensor(3.0))).all()

    # Static data should be duplicated across batch.
    for step in range(4):
        assert torch.isclose(input_data["static"][6:], torch.tensor(0.0)).all()
        cos_doy = input_data["static"][0, step, 2]
        assert torch.isclose(cos_doy, torch.cos(2 * np.pi * torch.tensor(1) / 366), atol=1e-3).all()
        sin_doy = input_data["static"][0, step, 3]
        assert torch.isclose(sin_doy, torch.sin(2 * np.pi * torch.tensor(1) / 366), atol=1e-3).all()
        cos_hod = input_data["static"][0, step, 4]
        assert torch.isclose(cos_hod, torch.cos(2 * np.pi * torch.tensor(3 + step * 3) / 24), atol=1e-3).all()
        sin_hod = input_data["static"][0, step, 5]
        assert torch.isclose(sin_hod, torch.sin(2 * np.pi * torch.tensor(3 + step * 3) / 24), atol=1e-3).all()

        cos_doy = input_data["static"][1, step, 2]
        assert torch.isclose(cos_doy, torch.cos(2 * np.pi * torch.tensor(1) / 366), atol=1e-3).all()
        sin_doy = input_data["static"][1, step, 3]
        assert torch.isclose(sin_doy, torch.sin(2 * np.pi * torch.tensor(1) / 366), atol=1e-3).all()
        cos_hod = input_data["static"][1, step, 4]
        assert torch.isclose(cos_hod, torch.cos(2 * np.pi * torch.tensor(6 + step * 3) / 24), atol=1e-3).all()
        sin_hod = input_data["static"][1, step, 5]
        assert torch.isclose(sin_hod, torch.sin(2 * np.pi * torch.tensor(6 + step * 3) / 24), atol=1e-3).all()

        # Dynamic input should be init time and previous timestep.
        assert torch.isclose(input_data["x"][0, 0, :20], torch.tensor(1.0)).all()
        assert torch.isclose(input_data["x"][0, 0, 20:], torch.tensor(0.0)).all()
        assert torch.isclose(input_data["x"][0, 1, :20], torch.tensor(1.0)).all()
        assert torch.isclose(input_data["x"][0, 1, 20:], torch.tensor(3.0)).all()

        assert torch.isclose(input_data["x"][1, 0, :20], torch.tensor(1.0)).all()
        assert torch.isclose(input_data["x"][1, 0, 20:], torch.tensor(3.0)).all()
        assert torch.isclose(input_data["x"][1, 1, :20], torch.tensor(1.0)).all()
        assert torch.isclose(input_data["x"][1, 1, 20:], torch.tensor(6.0)).all()

        # Climate data should contain target hours
        assert torch.isclose(input_data["climate"][0, step, :20], torch.tensor(1.0)).all()
        assert torch.isclose(input_data["climate"][0, step, 20:], torch.tensor(6.0 + step * 3)).all()
        assert torch.isclose(input_data["climate"][1, step, :20], torch.tensor(1.0)).all()
        assert torch.isclose(input_data["climate"][1, step, 20:], torch.tensor(9.0 + step * 3)).all()


class DirectModel(nn.Module):
    """
    Mock direct forecast model.
    """
    def forward(self, inpt: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """ Returns a single scalar field as the forecast. """
        return {"surface_precip": MeanTensor(inpt["x"][:, 0, :1])}


def test_run_direct_forecast(imerg_training_data_1, tmp_path):
    """
    Runs direct forecast with a dummy model ensuring that the expected result files are produced.
    """
    model = DirectModel()
    data_loader = DirectForecastLoader(
        imerg_training_data_1,
        init_times = np.arange(
            np.datetime64('2020-01-01T03:00:00'),
            np.datetime64("2020-01-01T07:00:00"),
            np.timedelta64(3, "h")
        ),
        n_steps=4,
        batch_size=2,
        full_climatology=True
    )
    data_loader = DataLoader(
        data_loader,
        batch_size=None,
        num_workers=2,
        collate_fn=lambda x: x,
        shuffle=False
    )

    run_direct_forecast(
        model,
        data_loader,
        tmp_path
    )

    result_files = sorted(list(tmp_path.glob("*.nc")))

    assert len(result_files) == 2
    assert result_files[0].name == "forecast_202001010300.nc"

    res = xr.load_dataset(result_files[0])
    valid_times_ref = np.arange(
        np.datetime64("2020-01-01T06:00:00"),
        np.datetime64("2020-01-01T18:00:00"),
        np.timedelta64(3, "h"),
    )
    assert "surface_precip" in res
    assert "valid_time" in res.dims
    assert "latitude" in res.dims
    assert "longitude" in res.dims
    assert (res.latitude[0].data < -89.0).all()
    assert (res.latitude[-1].data > 89.0).all()
    assert (res.longitude[0].data < -179).all()
    assert (res.longitude[-1].data > 179).all()

    assert (res.valid_time.data == valid_times_ref).all()
    assert (res.initialization_time.data == np.datetime64("2020-01-01T03:00:00")).all()


class AutoregressiveModel(nn.Module):
    """
    Mock direct forecast model.
    """
    def forward(self, inpt: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """ Returns a single scalar field as the forecast. """
        n_steps = inpt["static"].shape[1]
        return {"surface_precip": [MeanTensor(inpt["x"][:, 0, :1]) for _ in range(n_steps)]}


def test_run_autoregressive_forecast(imerg_training_data_1, tmp_path):
    """
    Runs autoregressive forecast with a dummy model ensuring that the expected result files are produced.
    """
    model = AutoregressiveModel()
    data_loader = AutoregressiveForecastLoader(
        imerg_training_data_1,
        init_times = np.arange(
            np.datetime64('2020-01-01T03:00:00'),
            np.datetime64("2020-01-01T07:00:00"),
            np.timedelta64(3, "h")
        ),
        n_steps=4,
        batch_size=2,
        full_climatology=True
    )
    data_loader = DataLoader(
        data_loader,
        batch_size=None,
        num_workers=2,
        collate_fn=lambda x: x
    )

    run_autoregressive_forecast(
        model,
        data_loader,
        tmp_path
    )

    result_files = sorted(list(tmp_path.glob("*.nc")))

    assert len(result_files) == 2
    assert result_files[0].name == "forecast_202001010300.nc"

    res = xr.load_dataset(result_files[0])
    valid_times_ref = np.arange(
        np.datetime64("2020-01-01T06:00:00"),
        np.datetime64("2020-01-01T18:00:00"),
        np.timedelta64(3, "h"),
    )
    assert "surface_precip" in res
    assert "valid_time" in res.dims
    assert "latitude" in res.dims
    assert "longitude" in res.dims
    assert (res.latitude[0].data < -89.0).all()
    assert (res.latitude[-1].data > 89.0).all()
    assert (res.longitude[0].data < -179).all()
    assert (res.longitude[-1].data > 179).all()

    assert (res.valid_time.data == valid_times_ref).all()
    assert (res.initialization_time.data == np.datetime64("2020-01-01T03:00:00")).all()
