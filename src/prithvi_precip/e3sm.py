"""
prithvi_precip.e3sm
===================

Provides conversion function to convert E3SM model output to corresponding MERRA fields.
"""

from functools import cache, cached_property
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
import xarray as xr

from torch.utils.data import Dataset

from prithvi_precip.utils import load_static_input, load_climatology


def load_surface_climatology(
    time: np.datetime64, climatology_path: Path, variables: Optional[List[None]] = None
) -> xr.Dataset:
    """
    Load Prithvi-WxC surface-variable climatology data for a given time.

    Args:
        time: A numpy.datetime64 object specifying the day for which to load the climatology.
        climatology_path: A path object pointing to the folder containing the climatology data.
        variables: An optional list restricting the variables to load.

    Return:
        An xarray.Dataset containing the climatology data.
    """
    climatology_path = Path(climatology_path)
    time = time.astype("datetime64[s]").item()
    start_of_year = datetime(year=time.year, month=1, day=1)
    doy = min((time - start_of_year).days + 1, 365)
    fname = f"climate_surface_doy{doy:03}_hour00.nc"

    with xr.open_dataset(climatology_path / fname) as data:
        if variables is not None:
            loaded = data[variables].compute()
        else:
            loaded = data.compute()
    return loaded


def load_vertical_climatology(
    time: np.datetime64, climatology_path: Path, variables: Optional[List[None]] = None
) -> xr.Dataset:
    """
    Load Prithvi-WxC profile-variable climatology data for a given time.

    Args:
        time: A numpy.datetime64 object specifying the day for which to load the climatology.
        climatology_path: A path object pointing to the folder containing the climatology data.
        variables: An optional list restricting the variables to load.

    Return:
        An xarray.Dataset containing the climatology data.
    """
    climatology_path = Path(climatology_path)
    time = time.astype("datetime64[s]").item()
    start_of_year = datetime(year=time.year, month=1, day=1)
    doy = min((time - start_of_year).days + 1, 365)
    fname = f"climate_vertical_doy{doy:03}_hour00.nc"

    with xr.open_dataset(climatology_path / fname) as data:
        if variables is not None:
            loaded = data[variables].compute()
        else:
            loaded = data.compute()
    return loaded


def load_dynamic_data(e3sm_data: xr.Dataset, climatology_path: Path) -> xr.Dataset:
    """
    Loads dynamic data from E3SM data in the form expected from the Prithvi-WxC foundation model.

    Args:
        e3sm_data: An xarray.Dataset containing the E3SM data for a single time step.
        climatology_path: A path object pointing to the folder containing the Prithvi-WxC climatology
            data interpolated to the E3SM global grid.

    Return:
        A numpy array of shape [160, 180, 360] containing the dyanmic input data for the Prithvi-WxC model.
    """
    clim_surf = load_surface_climatology(e3sm_data.time.data, climatology_path)

    # EFLUX -> LHFLX
    eflux = e3sm_data["LHFLX"].data
    # GWETROOT from climatology
    gwetroot = clim_surf["GWETROOT"].data
    # HFLUX
    hflux = e3sm_data["SHFLX"].data  # lAI from climatology
    lai = clim_surf["LAI"].data
    # LWGAB -> FLDS: Downwelling long-wave flux at surface
    lwgab = e3sm_data["FLDS"].data
    # LWGEM -> FLNS - FLDS: Upwelling long-wave flux at surface
    lwgem = e3sm_data["FLDS"].data - e3sm_data["FLNS"].data
    # LWTUP -> FLUT
    lwtup = e3sm_data["FLUT"].data
    # PS -> PS
    ps = e3sm_data["PS"].data
    # QV2M -> QREFHT
    qv2m = e3sm_data["QREFHT"].data
    # SLP from climatology
    slp = clim_surf["SLP"].data
    # SWGNT
    swgnt = e3sm_data["FSNS"].data
    # SWTNT
    swtnt = e3sm_data["FSNT"].data
    # T2M
    t2m = e3sm_data["TS"].data
    # TQI
    tqi = e3sm_data["TGCLDIWP"].data
    # TQL
    tql = e3sm_data["TGCLDLWP"].data
    # TQV
    tqv = e3sm_data["TMQ"].data
    # TS
    ts = e3sm_data["TS"].data
    # U10M
    u10m = e3sm_data["UBOT"].data
    # V10M
    v10m = e3sm_data["VBOT"].data
    z0m = clim_surf["Z0M"].data

    clim_vert = load_vertical_climatology(e3sm_data.time.data, climatology_path)
    # Vertical vars
    height = np.flip(clim_vert["H"].data, 0)

    qi_clim = np.flip(clim_vert["QI"].data, 0)
    tqi_clim = np.trapz(qi_clim, x=-height, axis=0)
    qi = qi_clim * (tqi / tqi_clim)[None]

    ql_clim = np.flip(clim_vert["QL"].data, 0)
    tql_clim = np.trapz(ql_clim, x=-height, axis=0)
    ql = ql_clim * (tql / tql_clim)[None]

    cloud = np.minimum(qi / qi_clim.max(0)[None] + ql / ql_clim.max(0)[None], 1.0)

    qv_clim = np.flip(clim_vert["QV"].data, 0)
    tqv_clim = np.trapz(qv_clim, x=-height, axis=0)
    qv = qv_clim * (tqv / tqv_clim)[None]

    omega = np.flip(clim_vert["OMEGA"].data, 0)
    omega_scaling = e3sm_data["OMEGA500"].data / np.maximum(omega[7], 2e-1)
    omega = omega * omega_scaling[None]

    pl = np.flip(clim_vert["PL"], 0)

    t = np.flip(clim_vert["T"].data, 0)
    t = t * (e3sm_data["T250"].data / t[4])[None]
    t[0] = e3sm_data["T050"]
    t[1] = e3sm_data["T100"]
    t[2] = e3sm_data["T150"]
    t[3] = e3sm_data["T200"]

    u = np.flip(clim_vert["U"].data, 0)
    u = u * (e3sm_data["U850"].data / np.maximum(u[10], 1.0))[None]
    u[0] = e3sm_data["U050"]
    u[1] = e3sm_data["U100"]
    u[2] = e3sm_data["U150"]
    u[3] = e3sm_data["U200"]
    u[4] = e3sm_data["U250"]

    v = np.flip(clim_vert["V"].data, 0)
    v = v * (e3sm_data["V850"].data / np.maximum(v[10], 2.0))[None]
    v[3] = e3sm_data["V200"]

    return np.concatenate(
        [
            eflux[None],
            gwetroot[None],
            hflux[None],
            lai[None],
            lwgab[None],
            lwgem[None],
            lwtup[None],
            ps[None],
            qv2m[None],
            slp[None],
            swgnt[None],
            swtnt[None],
            t2m[None],
            tqi[None],
            tql[None],
            tqv[None],
            ts[None],
            u10m[None],
            v10m[None],
            z0m[None],
            cloud,
            height,
            omega,
            pl,
            qi,
            ql,
            qv,
            t,
            u,
            v,
        ],
        axis=0,
    )


class E3SMS2SDataset(Dataset):
    """
    A data loader class for regional S2S precipitation forecasts using E3SM climate-model data.

    The dataset expects a folder containing one or multiple NetCDF4 files containing the E3SM input
    data.
    """

    def __init__(
        self,
        data_dir: Path,
        static_data_dir: Optional[Path] = None,
        forecast_range: Tuple[int, int] = (14, 21),
        roi: Tuple[float, float] = (120, 45),
        precip_climatology: Optional[Path] = None,
    ):
        """
        Args:
            data_dir: The directory containign the training data.
            static_data_dir: The directory containing the static input data
                (climatology, static MERRA fields).
            forecast_range: A tuple [t_min, t_max) defining a half-closed interval for targeted
                forecast range in days. The lead time of each sample is sampled randomly from this
                interval.
            roi: A tuple (lon, lat) defining the center coordinate of the 20 x 20 px prediction region.
            precip_climatology: An optional path pointing to a NetCDF4 file containing a precipitation
                climatology to use to calculate anomalies. If not provided, anomalies are calculated
                on-the-fly at the beginning of the training.
        """
        super().__init__()
        self.data_dir = Path(data_dir)

        if static_data_dir is None:
            static_data_dir = self.data_dir.parent
        self.static_data_dir = Path(static_data_dir)
        self.forecast_range = forecast_range
        self.roi = roi
        self.rng = np.random.default_rng(42)

        self._precip_climatology = precip_climatology

    @cached_property
    def data_files(self):
        """
        List of all input data files.
        """
        return sorted(list(self.data_dir.glob("*.nc")))

    @cached_property
    def all_time_steps(self) -> Dict[int, Path]:
        """
        List all available time steps across different training input files.

        Return:
            A dictionary mapping the index of the first sample in each file to the corresponding
            file.
        """
        time_steps = {}
        tot_steps = 0

        for data_file in self.data_files:
            with xr.open_dataset(data_file) as data:
                # We need two input days, so we need to subtract one day.
                n_steps = data.time.size - self.forecast_range[1]
                time_steps[(tot_steps, tot_steps + n_steps)] = data_file
                tot_steps += n_steps

        return time_steps

    @cached_property
    def sample_counts(self) -> np.ndarray:
        """
        The index of the first sample in each input file.
        """
        return np.array([cts[0] for cts in self.all_time_steps.keys()])

    @cached_property
    def sample_keys(self) -> np.ndarray:
        """
        List of tuple describing the sample range for each data file.


        """
        return list(self.all_time_steps.keys())

    @cached_property
    def roi_bounds(self) -> Dict[str, slice]:
        """
        A dictionary defining the lons and lat slices defining the target area.
        """
        inpt_file = next(iter(self.all_time_steps.values()))
        with xr.open_dataset(inpt_file) as data:
            lons = data.lon.data
            lats = data.lat.data

        lon_c, lat_c = self.roi
        if lon_c < 0:
            lon_c += 360

        cntr_lon = np.searchsorted(lons, lon_c)
        cntr_lat = np.searchsorted(lats, lat_c)

        lon_start = cntr_lon - 10
        lon_end = cntr_lon + 10
        lat_start = cntr_lat - 10
        lat_end = cntr_lat + 10

        return {"lon": slice(lon_start, lon_end), "lat": slice(lat_start, lat_end)}

    @property
    def precip_climatology(self):
        """
        The precipitation climatology field.
        """
        if isinstance(self._precip_climatology, Path):
            with xr.open_dataset(precip_climatology) as data:
                self._precip_climatology = data.PRECT.data
        elif self._precip_climatology is None:
            precip_sum = None
            precip_cts = None
            files = list(self.all_time_steps.values())

            for path in tqdm(files, desc="Calculating precipitation climatology"):
                with xr.open_dataset(path) as data:
                    precip = data.PRECT.data * 1e3 * 3.6e3
                    valid = np.isfinite(precip)
                    if precip_sum is None:
                        precip_sum = precip.sum(0)
                        precip_cts = valid.sum(0)
                    else:
                        precip_sum += precip.sum(0)
                        precip_cts += valid.sum(0)
            self._precip_climatology = precip_sum / precip_cts
        return self._precip_climatology

    def worker_init_fn(self, w_id: int) -> None:
        """
        Seeds the dataset loader's random number generator.
        """
        seed = int.from_bytes(os.urandom(4), "big") + w_id
        self.rng = np.random.default_rng(seed)

    @cache
    def __len__(self) -> int:
        """
        The total number of time steps in the dataset.
        """
        return list(self.all_time_steps)[-1][-1]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Load input data.

        Args:
            index: The index of the training sample.

        Return:
            A dictionary containing the input data for regional forecasts with the Prithvi-WxC model.
        """
        file_ind = np.searchsorted(self.sample_counts, index, side="right")
        start_index = self.sample_counts[file_ind - 1]
        input_file = self.all_time_steps[self.sample_keys[start_index]]

        forecast_range = self.rng.integers(*self.forecast_range)

        with xr.open_dataset(input_file) as data:
            data = data[
                {
                    "time": [
                        index - start_index,
                        index - start_index + 1,
                        index + forecast_range,
                    ]
                }
            ].compute()

        dynamic = torch.stack(
            [
                torch.tensor(
                    load_dynamic_data(
                        data[{"time": 0}], self.static_data_dir / "climatology"
                    )
                ),
                torch.tensor(
                    load_dynamic_data(
                        data[{"time": 1}], self.static_data_dir / "climatology"
                    )
                ),
            ]
        )

        init_time = np.datetime64(str(data.time.data[1]))
        target_time = np.datetime64(str(data.time.data[2]))
        static = load_static_input(init_time, self.static_data_dir)
        climate = load_climatology(target_time, self.static_data_dir)
        lead_time = (target_time - init_time).astype("timedelta64[h]").astype("float32")

        inpt = {
            "x": dynamic.to(dtype=torch.float32),
            "static": torch.tensor(static).to(dtype=torch.float32),
            "climate": torch.tensor(climate).to(dtype=torch.float32),
            "lead_time": torch.tensor(lead_time).to(dtype=torch.float32),
            "input_time": torch.tensor(24.0),
        }

        lon_bounds = self.roi_bounds["lon"]
        lat_bounds = self.roi_bounds["lat"]
        target = data[{"time": 2}][self.roi_bounds].PRECT.data * 1e3 * 3.6e3
        target = target - self.precip_climatology[lat_bounds, lon_bounds]

        inpt.update(
            {
                "x_regional": inpt["x"][..., lat_bounds, lon_bounds],
                "climate_regional": inpt["climate"][..., lat_bounds, lon_bounds],
                "static_regional": inpt["static"][..., lat_bounds, lon_bounds],
            }
        )

        return inpt, target
