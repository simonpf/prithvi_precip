"""
prithvi_precip.cli
==================

Command line interface for extracting training and verification data for the Prithvi Precip model.
"""
import click

from prithvi_precip.data import imerg, merra2, merra2_precip, era5_precip, mrms, stage4, spc
from prithvi_precip.obs import cpcir


@click.group()
def prithvi_precip():
    """The command line interface for the 'prithvi_precip' package."""
    pass

prithvi_precip.command(name="extract_merra_data")(merra2.extract_merra_data)
prithvi_precip.command(name="extract_static_merra_data")(merra2.extract_static_merra_data)
prithvi_precip.command(name="extract_imerg_precip")(imerg.extract_precip)
prithvi_precip.command(name="extract_merra2_precip")(merra2_precip.extract_precip)
prithvi_precip.command(name="extract_era5_precip")(era5_precip.extract_precip)
prithvi_precip.command(name="extract_mrms_precip")(mrms.extract_precip)
prithvi_precip.command(name="extract_stage4_precip")(stage4.extract_precip)
prithvi_precip.command(name="extract_severe_weather")(spc.extract_training_data)
prithvi_precip.command(name="extract_cpcir_obs")(cpcir.extract_cpcir_observations)
