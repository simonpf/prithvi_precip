"""
prithvi_precip.domains
======================

Defines domains over which data can be extracted.
"""
from typing import Tuple

import numpy as np

from pyresample import AreaDefinition


MERRA = AreaDefinition(
    area_id="MERRA2 grid",
    description="Regular lat/lon grid.",
    proj_id="merra",
    projection={
        "proj": "longlat",
        "datum": "WGS84",
    },
    width=576,
    height=360,
    area_extent=[-180.3125, 90, 179.6978, -90]
)


MEXICO = AreaDefinition(
    area_id="Mexico",
    description="Regular lat/lon grid.",
    proj_id="mx",
    projection={
        "proj": "longlat",
        "datum": "WGS84",
    },
    width=256,
    height=256,
    area_extent=[-120, 5, -94.4, 30.6]
)


BRAZIL = AreaDefinition(
    area_id="Brazil",
    description="Regular lat/lon grid.",
    proj_id="mx",
    projection={
        "proj": "longlat",
        "datum": "WGS84",
    },
    width=256,
    height=256,
    area_extent=[-120, -30.6, -94.4, 5]
)

def get_lonlats(domain: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get longitude and latitude coordinates for domain.

    Args:
        domain: The name of the domain.

    Return:
        A tuple containing the longitude and latitude coordinates for the domain.
    """
    domain = globals()[domain.upper()]
    lons, lats = domain.get_lonlats()
    return lons, lats


def get_lon_lat_bins(domain: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get longitude and latitude bins for domain.

    Args:
        domain: The name of the domain.

    Return:
        A tuple containing the longitude and latitude bins for the domain.
    """
    domain = globals()[domain.upper()]
    lons, lats = domain.get_lonlats()

    lon_bins = np.zeros(lons.size + 1)
    lon_bins[1:-1] = 0.5 * (lons[1:] + lons[:-1])
    lon_bins[0] = lon_bins[1] - (lons_bins[2] - lon_bins[1])
    lon_bins[-1] = lon_bins[-2] + (lons_bins[-2] - lon_bins[-3])

    lat_bins = np.zeros(lats.size + 1)
    lat_bins[1:-1] = 0.5 * (lats[1:] + lats[:-1])
    lat_bins[0] = lat_bins[1] - (lats_bins[2] - lat_bins[1])
    lat_bins[-1] = lat_bins[-2] + (lats_bins[-2] - lat_bins[-3])

    return lon_bins, lat_bins
