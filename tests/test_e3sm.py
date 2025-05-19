"""
Tests for the prithvi_precip.e3sm module.
"""
import os
from pathlib import Path

import pytest

import numpy as np
import xarray as xr

from prithvi_precip.e3sm import (
    load_dynamic_data,
    E3SMS2SDataset
)


CLIMATOLOGY_PATH = os.environ.get("E3SM_DATA", None)
HAS_CLIMATOLOGY = CLIMATOLOGY_PATH is not None and Path(CLIMATOLOGY_PATH).exists()
NEEDS_CLIMATOLOGY = pytest.mark.skipif(not HAS_CLIMATOLOGY, reason="Needs climatology data.")


@pytest.fixture
def e3sm_data(tmp_path: Path) -> Path:
    """
    Creates mock-up E3SM training data with two time steps.
    """
    data_path = tmp_path / "e3sm"
    data_path.mkdir()

    lon = np.linspace(0, 360, 361)
    lon = 0.5 * (lon[1:] + lon[:-1])

    lat = np.linspace(-90, 90, 181)
    lat = 0.5 * (lat[1:] + lat[:-1])
    time = np.arange(
        np.datetime64("2020-01-01"),
        np.datetime64("2020-01-03"),
        np.timedelta64(1, "D")
    )


    data = xr.Dataset({
        "lat": (("lat",), lat),
        "lon": (("lon",), lon),
        "time": (("time",), time),
    })

    day_1 = data.time[0].dt.day
    day_2 = data.time[1].dt.day

    vars = [
        "LHFLX", "SHFLX", "FLDS", "FLDS", "FLNS", "FLUT", "PS", "QREFHT",
        "FSNS", "FSNT", "TS", "TGCLDIWP", "TGCLDLWP", "TMQ", "TS",
        "UBOT", "VBOT", "OMEGA500", "T250", "T050", "T100", "T150", "T200",
        "U850", "U050", "U100", "U150", "U200", "U250", "V", "V850", "V200"
    ]
    for var in vars:
        data[var] = (
                ("time", "lat", "lon"),
                np.ones((time.size, lat.size, lon.size), dtype=np.float32)
        )
        data[var].data[0, :, :] = day_1
        data[var].data[1, :, :] = day_2

    data.to_netcdf(data_path / "e3sm_training_data.nc")
    return data_path


@NEEDS_CLIMATOLOGY
def test_load_dynamic_data(e3sm_data):
    """
    Test loading of dynamic input data from E3SM training data file.
    """
    training_data = xr.load_dataset(e3sm_data / "e3sm_training_data.nc")

    input_data = load_dynamic_data(training_data[{"time": 0}], Path(CLIMATOLOGY_PATH) / "climatology")
    assert isinstance(input_data, np.ndarray)
    assert (input_data[0] == 1.0).all()

    input_data = load_dynamic_data(training_data[{"time": 1}], Path(CLIMATOLOGY_PATH) / "climatology")
    assert (input_data[0] == 2.0).all()


@NEEDS_CLIMATOLOGY
def test_e3sm_s2s_dataset():
    """
    Test the E3SMS2SDatasets.
    """
    dataset = E3SMS2SDataset(Path(CLIMATOLOGY_PATH) / "training_data")
    assert len(dataset) == 344

    inpt, target = dataset[0]

    assert "x" in inpt
    assert "static" in inpt
    assert "climate" in inpt
