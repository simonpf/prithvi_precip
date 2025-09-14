"""
prithvi_precip.datasets
=======================

Dataset for training the Prithvi Precip model.
"""
import logging

from .merra2 import MERRAInputData
from .precipitation import DirectPrecipForecastDataset, AutoregressivePrecipForecastDataset
from .observations import ObservationLoader
