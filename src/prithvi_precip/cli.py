"""
prithvi_precip.cli
==================

Command line interface for extracting training and verification data for the Prithvi Precip model.
"""
import click

from prithvi_precip.data import imerg, merra2_precip, era5_precip, mrms, spc


@click.group()
def prithvi_precip():
    """The command line interface for the 'prithvi_precip' package."""
    pass

prithvi_precip.command(name="extract_imerg_precip")(imerg.extract_precip)
prithvi_precip.command(name="extract_merra2_precip")(merra2_precip.extract_precip)
prithvi_precip.command(name="extract_era5_precip")(era5_precip.extract_precip)
prithvi_precip.command(name="extract_mrms_precip")(mrms.extract_precip)
prithvi_precip.command(name="extract_severe_weather")(spc.extract_training_data)
