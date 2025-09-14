"""
Pytest fixtures for prithvi_precip tests.
"""
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch
import xarray as xr

from prithvi_precip.data.merra2 import (
    SURFACE_VARS,
    VERTICAL_VARS,
    STATIC_SURFACE_VARS,
    LEVELS,
    LAT_BINS,
    LON_BINS
)


@pytest.fixture()
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def merra_dimensions():
    """Standard MERRA2 grid dimensions."""
    return {
        'latitude': 361,  # -90 to 90 degrees
        'longitude': 576,  # -180 to 179.375 degrees  
        'levels': len(LEVELS),
        'time': 1
    }


@pytest.fixture
def merra_coordinates(merra_dimensions):
    """Generate MERRA2 coordinate arrays."""
    return {
        'latitude': np.linspace(-90, 90, merra_dimensions['latitude']),
        'longitude': np.linspace(-180, 179.375, merra_dimensions['longitude']),
        'lev': np.array(LEVELS),
        'time': [np.datetime64('2023-01-01T00:00:00')]
    }


@pytest.fixture
def mock_merra_dynamic_data(merra_coordinates, merra_dimensions):
    """Create mock MERRA2 dynamic data file."""

    def create_file(timestamp: datetime, output_path: Path):
        data_vars = {}

        
        # Surface variables
        for var in SURFACE_VARS:
            data_vars[var] = xr.DataArray(
                timestamp.hour * np.ones((
                    merra_dimensions['latitude'], 
                    merra_dimensions['longitude']
                )).astype(np.float32),
                dims=['latitude', 'longitude'],
                coords={
                    'latitude': merra_coordinates['latitude'],
                    'longitude': merra_coordinates['longitude']
                }
            )
        
        # Vertical variables
        for var in VERTICAL_VARS:
            data_vars[var] = xr.DataArray(
                timestamp.hour * np.ones((
                    merra_dimensions['levels'],
                    merra_dimensions['latitude'],
                    merra_dimensions['longitude']
                )).astype(np.float32),
                dims=['lev', 'latitude', 'longitude'],
                coords={
                    'lev': merra_coordinates['lev'],
                    'latitude': merra_coordinates['latitude'],
                    'longitude': merra_coordinates['longitude']
                }
            )
        
        dataset = xr.Dataset(data_vars)
        dataset.to_netcdf(output_path)
        return output_path
    
    return create_file


@pytest.fixture
def mock_merra_static_data(merra_coordinates, merra_dimensions):
    """Create mock MERRA2 static data file."""
    def create_file(output_path: Path):
        data_vars = {}
        
        for var in STATIC_SURFACE_VARS:
            data_vars[var] = xr.DataArray(
                np.random.randn(
                    12,
                    merra_dimensions['latitude'], 
                    merra_dimensions['longitude']
                ).astype(np.float32),
                dims=['time', 'latitude', 'longitude'],
                coords={
                    "time": np.arange(
                        np.datetime64("1980-01-01", "M"),
                        np.datetime64("1981-01-01", "M"),
                        np.timedelta64(1, "M")
                    ),
                    'latitude': merra_coordinates['latitude'],
                    'longitude': merra_coordinates['longitude']
                }
            )
        
        dataset = xr.Dataset(data_vars)
        dataset.to_netcdf(output_path)
        return output_path
    
    return create_file


@pytest.fixture
def mock_precipitation_data(merra_coordinates, merra_dimensions):
    """Create mock precipitation output data."""
    def create_file(timestamp: datetime, output_path: Path):
        # Single precipitation variable
        precip_data = np.random.exponential(
            0.5, size=(
                merra_dimensions['latitude'] - 1,
                merra_dimensions['longitude']
            )
        ).astype(np.float32)

        lats = merra_coordinates['latitude']
        lats = 0.5 * (lats[1:] + lats[:-1])
        
        dataset = xr.Dataset({
            'surface_precip': xr.DataArray(
                precip_data[None],  # Add time dimension
                dims=['time', 'latitude', 'longitude'],
                coords={
                    'time': [timestamp],
                    'latitude': lats,
                    'longitude': merra_coordinates['longitude']
                }
            )
        })
        
        dataset.to_netcdf(output_path)
        return output_path
    
    return create_file


@pytest.fixture
def mock_observation_data(merra_coordinates, merra_dimensions):
    """Create mock satellite observation data."""
    def create_file(timestamp: datetime, output_path: Path, n_channels: int = 32):
        obs_data = np.random.randn(
            n_channels,
            merra_dimensions['latitude'], 
            merra_dimensions['longitude']
        ).astype(np.float32)
        
        dataset = xr.Dataset({
            'observations': xr.DataArray(
                obs_data,
                dims=['channel', 'latitude', 'longitude'],
                coords={
                    'channel': np.arange(n_channels),
                    'latitude': merra_coordinates['latitude'],
                    'longitude': merra_coordinates['longitude']
                }
            )
        })
        
        dataset.to_netcdf(output_path)
        return output_path
    
    return create_file


@pytest.fixture
def mock_climatology_data(merra_coordinates, merra_dimensions):
    """Create mock climatology data."""
    def create_file(timestamp: datetime, output_path: Path):

        data_vars = {}
        # Surface variables
        for var in SURFACE_VARS:
            data_vars[var] = xr.DataArray(
                np.random.randn(
                    merra_dimensions['latitude'],
                    merra_dimensions['longitude']
                ).astype(np.float32),
                dims=['latitude', 'longitude'],
                coords={
                    'latitude': merra_coordinates['latitude'],
                    'longitude': merra_coordinates['longitude']
                }
            )
        dataset = xr.Dataset(data_vars)
        fname = timestamp.strftime(f"climate_surface_doy%j_hour%H.nc")
        dataset.to_netcdf(output_path / fname)

        data_vars = {}
        # Vertical variables
        for var in VERTICAL_VARS:
            data_vars[var] = xr.DataArray(
                np.random.randn(
                    merra_dimensions['levels'],
                    merra_dimensions['latitude'],
                    merra_dimensions['longitude']
                ).astype(np.float32),
                dims=['lev', 'latitude', 'longitude'],
                coords={
                    'lev': merra_coordinates['lev'],
                    'latitude': merra_coordinates['latitude'],
                    'longitude': merra_coordinates['longitude']
                }
            )

        dataset = xr.Dataset(data_vars)
        fname = timestamp.strftime(f"climate_vertical_doy%j_hour%H.nc")
        dataset.to_netcdf(output_path / fname)
        return output_path
    
    return create_file


@pytest.fixture
def merra_dataset_structure(temp_data_dir, mock_merra_dynamic_data, 
                           mock_merra_static_data, mock_precipitation_data,
                           mock_climatology_data):
    """Create a complete mock MERRA dataset directory structure."""
    def create_structure(start_date: datetime, n_timesteps: int = 24, 
                        timestep_hours: int = 3):
        
        # Create directory structure
        dynamic_dir = temp_data_dir / "training_data" / "dynamic"
        static_dir = temp_data_dir / "static" 
        imerg_precip_dir = temp_data_dir / "training_data" / "imerg_3" / "surface_precip"
        era5_precip_dir = temp_data_dir / "training_data" / "era5_precip" / "surface_precip"
        clim_dir = temp_data_dir / "climatology"
        
        for dir_path in [dynamic_dir, static_dir, imerg_precip_dir, era5_precip_dir, clim_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        timestamps = []
        dynamic_files = []
        imerg_precip_files = []
        era5_precip_files = []
        
        # Generate time series of files
        for i in range(n_timesteps):
            timestamp = start_date + timedelta(hours=i * timestep_hours)
            timestamps.append(timestamp)
            
            # Create subdirectories by date
            date_str = timestamp.strftime("%Y/%m/%d")
            (dynamic_dir / date_str).mkdir(parents=True, exist_ok=True)
            (imerg_precip_dir / date_str).mkdir(parents=True, exist_ok=True)
            (era5_precip_dir / date_str).mkdir(parents=True, exist_ok=True)
            
            # Dynamic input files
            dynamic_file = (dynamic_dir / date_str / 
                          f"merra2_{timestamp.strftime('%Y%m%d%H%M%S')}.nc")
            mock_merra_dynamic_data(timestamp, dynamic_file)
            dynamic_files.append(str(dynamic_file.relative_to(temp_data_dir)))
            
            # Precipitation output files
            precip_file = (imerg_precip_dir / date_str /
                          f"imerg_{timestamp.strftime('%Y%m%d%H%M%S')}.nc")
            mock_precipitation_data(timestamp, precip_file)
            imerg_precip_files.append(str(precip_file.relative_to(temp_data_dir)))
            precip_file = (era5_precip_dir / date_str /
                          f"era5_precip_{timestamp.strftime('%Y%m%d%H%M%S')}.nc")
            mock_precipitation_data(timestamp, precip_file)
            era5_precip_files.append(str(precip_file.relative_to(temp_data_dir)))
            mock_climatology_data(timestamp, clim_dir)

        # Static files
        static_file = static_dir / "static.nc"
        mock_merra_static_data(static_file)
        
        return {
            'base_path': temp_data_dir,
            'timestamps': timestamps,
            'dynamic_files': dynamic_files,
            'imerg_precip_files': imerg_precip_files,
            'era5_precip_files': era5_precip_files,
            'static_file': str(static_file.relative_to(temp_data_dir)),
        }
    
    return create_structure


@pytest.fixture
def observation_dataset_structure(temp_data_dir, mock_observation_data):
    """Create mock observation dataset structure."""
    def create_structure(start_date: datetime, n_timesteps: int = 24,
                        timestep_hours: int = 3, n_channels: int = 32):
        
        obs_dir = temp_data_dir / "observations"
        obs_dir.mkdir(parents=True, exist_ok=True)
        
        timestamps = []
        obs_files = []
        
        for i in range(n_timesteps):
            timestamp = start_date + timedelta(hours=i * timestep_hours)
            timestamps.append(timestamp)
            
            date_str = timestamp.strftime("%Y/%m/%d")
            (obs_dir / date_str).mkdir(parents=True, exist_ok=True)
            
            obs_file = (obs_dir / date_str / 
                       f"obs_{timestamp.strftime('%Y%m%d%H%M%S')}.nc")
            mock_observation_data(timestamp, obs_file, n_channels)
            obs_files.append(str(obs_file.relative_to(temp_data_dir)))
        
        return {
            'base_path': temp_data_dir,
            'obs_path': obs_dir,
            'timestamps': timestamps,
            'obs_files': obs_files
        }
    
    return create_structure


@pytest.fixture
def sample_dataset_config():
    """Sample configuration for dataset testing."""
    return {
        'input_time': 3,
        'lead_times': [3, 6, 12, 24],
        'max_steps': 12,
        'center_meridionally': True,
        'tile_size': (30, 32),
        'n_tiles': (12, 18),
        'observation_layers': 32
    }


@pytest.fixture(scope="session")
def torch_device():
    """Get available torch device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_file_static(path: Path):
    """
    Create a  MERRA2 static data file containing the day of the year in the surface variables
    and the hour of the day in the vertical variables.
    """
    lats = 0.5 * (LAT_BINS[1:] + LAT_BINS[:-1])
    lons = 0.5 * (LON_BINS[1:] + LON_BINS[:-1])
    data = xr.Dataset()
    for var in STATIC_SURFACE_VARS:
        data[var] = (("time", "latitude", "longitude"), np.arange(12)[:, None, None] * np.ones((12, 360, 576)))
    data["latitude"] = (("latitude",), lats)
    data["longitude"] = (("longitude",), lons)
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


def create_file_scalers(path: Path):
    """
    Create a dummy IMERG training data file containing the hour of the day as precipitation values so that
    the loaded data can be used to verify that the correct data is loaded.
    """
    data = xr.Dataset()
    for var in SURFACE_VARS:
        data[var] = 1.0
    output_path = path / "scaling_factors" / "anomaly_variance_surface.nc"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(output_path)

    data = xr.Dataset()
    for var in VERTICAL_VARS:
        data[var] = (("lev",), np.ones(14))
    data["lev"] = (("lev",), LEVELS)
    output_path = path / "scaling_factors" / "anomaly_variance_vertical.nc"
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

    create_file_scalers(training_data_path)

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
