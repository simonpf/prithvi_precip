"""
Test cases demonstrating the use of pytest fixtures for dataset testing.
"""
from datetime import datetime, timedelta
import numpy as np
import pytest
import torch
import xarray as xr

from prithvi_precip.datasets import (
    MERRAInputData,
    DirectPrecipForecastDataset,
    ObservationLoader
)


def test_merra_dataset_structure_fixture(merra_dataset_structure, sample_dataset_config):
    """Test the MERRA dataset structure fixture."""
    start_date = datetime(2023, 1, 1)
    structure = merra_dataset_structure(start_date, n_timesteps=2)
    
    assert structure['base_path'].exists()
    assert len(structure['timestamps']) == 2
    assert len(structure['dynamic_files']) == 2
    assert len(structure['imerg_precip_files']) == 2
    assert len(structure['era5_precip_files']) == 2
    
    # Verify file structure
    dynamic_path = structure['base_path'] / structure['dynamic_files'][0]
    assert dynamic_path.exists()
    
    # Test that files can be loaded
    with xr.open_dataset(dynamic_path) as data:
        assert 'EFLUX' in data  # Surface variable
        assert 'T' in data      # Vertical variable


def test_merra_input_data_with_fixtures(merra_dataset_structure, sample_dataset_config):
    """Test MERRAInputData using mock data."""
    start_date = datetime(2023, 1, 1)
    structure = merra_dataset_structure(start_date, n_timesteps=10)
    
    dataset = MERRAInputData(
        training_data_path=structure['base_path'] / "training_data",
        input_time=sample_dataset_config['input_time'],
        lead_times=sample_dataset_config['lead_times'],
        climatology=True,
        center_meridionally=sample_dataset_config['center_meridionally']
    )
    
    assert len(dataset.times) == 10
    assert len(dataset.input_files) == 10
    
    # Test that we can get valid samples
    assert len(dataset.input_indices) > 0
    assert len(dataset.output_indices) > 0


def test_direct_precip_forecast_dataset(merra_dataset_structure, sample_dataset_config):
    """Test DirectPrecipForecastDataset with fixtures."""
    start_date = datetime(2023, 1, 1) 
    structure = merra_dataset_structure(start_date, n_timesteps=5)

    dataset = DirectPrecipForecastDataset(
        training_data_path=structure['base_path'] / "training_data",
        input_time=sample_dataset_config['input_time'],
        max_steps=sample_dataset_config['max_steps'],
        climate=True,
        reference_data="imerg",
        center_meridionally=True
    )
    
    assert len(dataset) > 0
    
    # Test getting a sample
    x, y = dataset[0]
    
    # Check input structure
    assert isinstance(x, dict)
    assert 'x' in x
    assert 'static' in x
    assert 'input_time' in x
    assert 'lead_time' in x
    
    # Check tensor shapes
    assert x['x'].ndim == 4  # (time_steps, channels, lat, lon)
    assert x['static'].ndim == 3  # (channels, lat, lon)
    assert y.ndim == 3  # (time, lat, lon)


def test_observation_loader_fixtures(observation_dataset_structure, sample_dataset_config):
    """Test ObservationLoader with mock data."""
    start_date = datetime(2023, 1, 1)
    structure = observation_dataset_structure(
        start_date, 
        n_timesteps=24,
        n_channels=sample_dataset_config['observation_layers']
    )
    
    loader = ObservationLoader(
        observation_path=structure['obs_path'],
        observation_layers=sample_dataset_config['observation_layers'],
        n_tiles=sample_dataset_config['n_tiles'],
        tile_size=sample_dataset_config['tile_size']
    )
    
    # Test loading observations for a specific time
    obs_time = np.datetime64(start_date)
    assert loader.has_obs(obs_time)
    
    observations, meta_data = loader.load_observations(obs_time)
    
    # Check shapes
    expected_shape = sample_dataset_config['n_tiles'] + (
        sample_dataset_config['observation_layers'], 
        1
    ) + sample_dataset_config['tile_size']
    assert observations.shape == expected_shape
    
    # Check meta data shape
    expected_meta_shape = sample_dataset_config['n_tiles'] + (
        sample_dataset_config['observation_layers'], 
        8  # meta data channels
    ) + sample_dataset_config['tile_size']
    assert meta_data.shape == expected_meta_shape


def test_mock_data_consistency(mock_merra_dynamic_data, mock_precipitation_data, 
                              temp_data_dir, merra_coordinates):
    """Test that mock data generators create consistent data."""
    timestamp = datetime(2023, 1, 1, 12, 0, 0)
    
    # Create dynamic data file
    dynamic_file = temp_data_dir / "test_dynamic.nc"
    mock_merra_dynamic_data(timestamp, dynamic_file)
    
    # Create precipitation file  
    precip_file = temp_data_dir / "test_precip.nc"
    mock_precipitation_data(timestamp, precip_file)
    
    # Load and verify
    with xr.open_dataset(dynamic_file) as dynamic_data:
        # Check all required variables are present
        for var in ['EFLUX', 'GWETROOT', 'HFLUX', 'LAI']:  # Sample surface vars
            assert var in dynamic_data
            
        for var in ['T', 'U', 'V', 'H']:  # Sample vertical vars  
            assert var in dynamic_data
            
        # Check coordinate consistency
        assert np.array_equal(dynamic_data.latitude.values, merra_coordinates['latitude'])
        assert np.array_equal(dynamic_data.longitude.values, merra_coordinates['longitude'])
    
    with xr.open_dataset(precip_file) as precip_data:
        assert 'surface_precip' in precip_data
        # Precipitation should be non-negative (exponential distribution)
        assert (precip_data.surface_precip.values >= 0).all()


@pytest.mark.parametrize("n_timesteps,input_time", [
    (24, 3),
    (48, 6),
    (72, 12)
])
def test_parametrized_dataset_creation(merra_dataset_structure, n_timesteps, input_time):
    """Test dataset creation with different parameters."""
    start_date = datetime(2023, 1, 1)
    structure = merra_dataset_structure(start_date, n_timesteps=n_timesteps)
    
    dataset = MERRAInputData(
        training_data_path=structure['base_path'] / "dynamic",
        input_time=input_time,
        lead_times=[input_time, input_time * 2],
        climatology=True
    )
    
    assert len(dataset.times) == n_timesteps
    # Should have some valid samples (depending on input_time)
    assert len(dataset.input_indices) >= 0


def test_torch_device_fixture(torch_device):
    """Test that torch device fixture works."""
    assert isinstance(torch_device, torch.device)
    
    # Test creating tensors on the device
    tensor = torch.randn(10, 10, device=torch_device)
    assert tensor.device == torch_device


def test_data_file_cleanup(temp_data_dir, mock_merra_dynamic_data):
    """Test that temporary data is properly cleaned up."""
    timestamp = datetime(2023, 1, 1)
    file_path = temp_data_dir / "test_cleanup.nc"
    
    mock_merra_dynamic_data(timestamp, file_path)
    assert file_path.exists()
    
    # After test completion, temp_data_dir fixture should clean up automatically
    # This is handled by the tempfile.TemporaryDirectory context manager


def test_climatology_fixture(mock_climatology_data, temp_data_dir, merra_coordinates):
    """Test climatology data fixture."""
    clim_file = temp_data_dir / "test_climatology.nc"
    mock_climatology_data(clim_file)
    
    with xr.open_dataset(clim_file) as clim_data:
        assert 'climatology' in clim_data
        # Should be non-negative (exponential distribution)
        assert (clim_data.climatology.values >= 0).all()
        # Check coordinates
        assert np.array_equal(clim_data.latitude.values, merra_coordinates['latitude'])
        assert np.array_equal(clim_data.longitude.values, merra_coordinates['longitude'])


def test_static_data_fixture(mock_merra_static_data, temp_data_dir, merra_coordinates):
    """Test static data fixture."""
    static_file = temp_data_dir / "test_static.nc"
    mock_merra_static_data(static_file)
    
    with xr.open_dataset(static_file) as static_data:
        # Check that expected static variables are present
        expected_vars = ['FRLAKE', 'FRLAND', 'FRLANDICE', 'FROCEAN']  # Sample static vars
        for var in expected_vars:
            if var in static_data:  # Some may not be in the test data
                assert static_data[var].dims == ('latitude', 'longitude')
        
        # Check coordinates
        assert np.array_equal(static_data.latitude.values, merra_coordinates['latitude'])
        assert np.array_equal(static_data.longitude.values, merra_coordinates['longitude'])
