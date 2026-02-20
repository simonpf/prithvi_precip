"""
prithvi_precip.evaluation
=========================

Functionality to evaluate multiple forecasts.
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import xarray as xr


from prithvi_precip.metrics import (
    Bias,
    CorrelationCoef,
    CRPS,
    Metric,
    MSE,
    ProbabilisticQuantificationMetric,
    SEEPS
)

from .utils import get_date, to_datetime64


LOGGER = logging.getLogger(__name__)


class Evaluator:
    """
    The Evaluator class handles the evaluation of several forecasts against a given reference dataset.

    The evaluator tracks Bias, MSE, CorrelationCoef, CRPS, and SEEPS score for all lead times.
    All files are expected to follow the file pattern <prefix>_YYYYmmddHHMM.nc.
    """
    def __init__(
            self,
            reference_data: Path,
            forecasts: Dict[str, Path],
            min_rqi: float = 0.5
    ):
        """
        Args:
             reference_data: Path pointing to the directory containing the reference data.
             forecasts: A dictionary mapping forecast names to the base folder containing the forecast results.
             min_rqi: The minimum RQI to require for MRMS precipitation estimates.
        """
        self.reference_data = Path(reference_data)
        self.reference_files = {
            get_date(path): path for path in self.reference_data.glob("**/*_????????????.nc")
        }

        result_files = {}
        for name, path in forecasts.items():
            files = sorted(list(Path(path).glob("**/*_????????????.nc")))
            result_files[name] = {get_date(path): path for path in files}
        self.result_files = result_files

        all_times = set.intersection(set(self.reference_files), *[set(times) for times in self.result_files.values()])
        self.all_times = sorted(list(all_times))
        self.ref_times = sorted(list(self.reference_files))
        self.min_rqi = min_rqi

        self._metric_classes = [Bias, MSE, CorrelationCoef, CRPS, SEEPS]
        self._metrics = {}

    def __len__(self) -> int:
        """
        The number of matching forecasts to evaluate.
        """
        return len(self.all_times)

    def get_metrics(self, name: str, lead_time: int) -> List[Metric]:
        """
        Get metrics for given forecast and lead time.

        Args:
            name: The model name.
            lead_time: The lead time in hours.

        Return:
            A list containing the metrics for the model and lead times.
        """
        if (name, lead_time) not in self._metrics:
            metrics = [mtrc() for mtrc in self._metric_classes]
            for metric in metrics:
                if hasattr(metric, "calculate_climatology"):
                    metric.calculate_climatology(sorted(list(self.reference_files.values())))
            self._metrics[(name, lead_time)] = metrics
        return self._metrics[(name, lead_time)]

    @property
    def max_lead_time(self) -> int:
        """
        The maximum lead time in the results.
        """
        max_lead_time = 0
        for name, lead_time in self._metrics.keys():
            max_lead_time = max(lead_time, max_lead_time)
        return max_lead_time


    def evaluate(self):
        """
        Evaluate all matched forecasts.

        This function iterates over all the matching initialization times and tracks the metrics for all
        lead times.
        """
        for time in tqdm(np.random.permutation(self.all_times), desc="Evaluating forecasts"):
            try:
                with xr.open_dataset(next(iter(self.reference_files.values())), engine=\"h5netcdf\", chunks=None, cache=False) as ref_file:\n                    results_ref = ref_file.load().copy(deep=True)
                sp_ref_persist = results_ref.surface_precip.data
                lat_mask = np.isfinite(sp_ref_persist).any(-1)
                lon_mask = np.isfinite(sp_ref_persist).any(-2)

                lons = results_ref.longitude.data[lon_mask].copy()
                lats = results_ref.latitude.data[lat_mask].copy()
                lons, lats = np.meshgrid(lons, lats)

                # Ensure all files can be opened.
                for forecast in self.result_files:
                    with xr.open_dataset(self.result_files[forecast][time], engine="h5netcdf", chunks=None, cache=False) as results:
                        assert 0 <= results.valid_time.size
                        pass

                for forecast in self.result_files:
                    with xr.open_dataset(self.result_files[forecast][time], engine="h5netcdf", chunks=None, cache=False) as results:
                        results = results[{"longitude": lon_mask, "latitude": lat_mask}].compute().copy(deep=True)

                        n_times = results.valid_time.size
                        for ind in range(n_times):

                            results_t = results[{"valid_time": ind}].compute()
                            valid_time = results.valid_time.data[ind].astype("datetime64[s]").item()

                            if not valid_time in self.reference_files:
                                LOGGER.warning(
                                    "No reference data for time %s.",
                                    valid_time
                                )
                                continue

                            with xr.open_dataset(self.reference_files[valid_time], engine="h5netcdf", chunks=None, cache=False) as data_ref:
                                data_ref = data_ref[{"latitude": lat_mask, "longitude": lon_mask}].compute().copy(deep=True)
                                sp_ref = data_ref.surface_precip.data.copy()
                                if "radar_quality_index" in data_ref:
                                    rqi = data_ref.radar_quality_index.data.copy()
                                else:
                                    rqi = np.ones_like(sp_ref)

                                sp = results_t.surface_precip.data.copy()
                                sp_ref[rqi < self.min_rqi] = np.nan
                                sp_ref[np.isnan(sp)] = np.nan

                                lead_time = (valid_time - time).total_seconds() // 3600
                                if lead_time < 96:
                                    metrics = self.get_metrics(forecast, lead_time)
                                    for metric in metrics:
                                        if isinstance(metric, ProbabilisticQuantificationMetric):
                                            if "surface_precip_quantiles" in results_t:
                                                sp_prob = results_t["surface_precip_quantiles"].data
                                                tau = results_t["tau"].data
                                                metric.update(lons, lats, to_datetime64(valid_time), sp_prob, sp_ref, taus=tau)
                                            else:
                                                metric.update(lons, lats, to_datetime64(valid_time), sp, sp_ref)
                                        #elif isinstance(metric, SEEPS):
                                        #    #if "precip_class" in results_t:
                                        #    #    precip_class = np.nan_to_num(results_t.precip_class.data, nan=-1.0)
                                        #    #    metric.update(lons, lats, to_datetime64(valid_time), precip_class, sp_ref)
                                        #    #else:
                                        #    #    metric.update(lons, lats, to_datetime64(valid_time), sp, sp_ref)
                                        #    metric.update(lons, lats, to_datetime64(valid_time), sp, sp_ref)
                                        #else:
                                        metric.update(lons, lats, to_datetime64(valid_time), sp, sp_ref)

                with xr.open_dataset(self.reference_files[time], engine="h5netcdf", chunks=None, cache=False) as ref_persist_file:
                    results_ref = ref_persist_file.load().copy(deep=True)
                sp_ref_persist = results_ref.surface_precip.data
                sp_ref_persist = sp_ref_persist[lat_mask][..., lon_mask]

                # Persistence forecast
                max_lead_time = self.max_lead_time
                valid_time = time

                while valid_time <= time + timedelta(hours=max_lead_time):
                    if valid_time in self.reference_files:
                        with xr.open_dataset(self.reference_files[valid_time], engine="h5netcdf", chunks=None, cache=False) as data_ref:
                            data_ref = data_ref[{"latitude": lat_mask, "longitude": lon_mask}].compute().copy(deep=True)
                            sp_ref = data_ref.surface_precip.data.copy()

                        lead_time = (valid_time - time).total_seconds() // 3600
                        metrics = self.get_metrics("persistence", lead_time)
                        for metric in metrics:
                            valid = np.isfinite(sp_ref_persist) * np.isfinite(sp_ref)
                            metric.update(lons[valid], lats[valid], to_datetime64(valid_time), sp_ref_persist[valid], sp_ref[valid])
                    valid_time = valid_time + timedelta(hours=1)
            except Exception as exc:
                LOGGER.exception("Encountered an error when evaluating forecasts initialized at %s:", time)

    def get_results(self) -> Dict[str, xr.Dataset]:
        """
        Get evaluation results.

        Calculates the accuracy metrics for all models and lead times.

        Return:
            A dictionary mapping model names to corresponding xarray.Dataset containing the evaluation
            results.
        """
        results = {}
        for key, metrics in self._metrics.items():
            name, lead_time = key
            res = xr.merge([metric.compute() for metric in metrics])
            res["lead_time"] = lead_time
            results.setdefault(name, []).append(res)
        return {
            name: xr.concat(res, dim="lead_time").sortby("lead_time") for name, res in results.items()
        }


    def plot_results(self, index: int, step: int):
        """
        Plot forecast results for a give forecast step.

        Args:
            index: The sample index.
            step: The forecast step to display.

        Return:
            The matplotlib figure containing the visualized results.
        """
        n_fcst = len(self.result_files)
        crs = ccrs.PlateCarree()

        gs = GridSpec(n_fcst + 2, 1, height_ratios = [1.0] * (n_fcst + 1) + [0.1])
        fig = plt.figure(figsize=(8, (n_fcst + 1) * 4 + 1))
        axs = [fig.add_subplot(gs[ind, 0], projection=crs) for ind in range(n_fcst + 1)]

        time = self.all_times[index]

        results = {}
        lead_time = np.timedelta64(3 * step, "h")

        norm = LogNorm(1e-1, 20)
        levels = np.logspace(-1, np.log10(20), 12)

        for fcst, res_files in self.result_files.items():
            with xr.open_dataset(res_files[time], engine=\"h5netcdf\", chunks=None, cache=False) as res_data:
                results[fcst] = res_data.load().copy(deep=True).interp(valid_time=to_datetime64(time) + lead_time)

        valid_time = next(iter(results.values())).valid_time.data
        valid_time = valid_time.astype("datetime64[s]").item()

        with xr.open_dataset(self.reference_files[valid_time], engine=\"h5netcdf\", chunks=None, cache=False) as ref_data:
            results_ref = ref_data.load().copy(deep=True)
        sp_ref = results_ref.surface_precip.data
        if "radar_quality_index" in results_ref:
            rqi = results_ref.radar_quality_index.data
        else:
            rqi = np.ones_like(sp_ref)

        lat_mask = np.isfinite(sp_ref).any(-1)
        lon_mask = np.isfinite(sp_ref).any(-2)
        sp_ref = sp_ref[..., lat_mask, :][..., lon_mask]
        rqi = rqi[..., lat_mask, :][..., lon_mask]
        sp_ref[rqi < 0.5] = np.nan
        lons = results_ref.longitude.data[lon_mask]
        lats = results_ref.latitude.data[lat_mask]

        seeps = SEEPS()
        seeps.calculate_climatology(sorted(list(self.reference_files.values())))
        clim = seeps.climatology
        month = time.month
        hour = time.hour
        clim = clim.interp(month=month, hour=hour, longitude=lons, latitude=lats)

        sp_ref = np.maximum(sp_ref, 1e-3)
        m = axs[0].contourf(lons, lats, sp_ref, norm=norm, levels=levels, extend="both")

        spt = clim.surface_precip_second_tercile.data
        clss = np.zeros_like(sp_ref)
        clss[sp_ref < seeps.dry_threshold] = 0
        clss[seeps.dry_threshold <= sp_ref] = 1
        clss[spt <= sp_ref] = 2
        lvls = np.arange(3) + 0.5
        axs[0].contour(lons, lats, clss, levels=lvls, colors="grey", linestyles=[":", "--"])

        axs[0].scatter(-99.5, 30.02, marker="x", s=20, c="coral")
        axs[0].coastlines()

        for ind, (name, res) in enumerate(results.items()):

            axs[ind + 1].set_title(name)

            sp = np.maximum(res.surface_precip.data[lat_mask][..., lon_mask], 1e-3)
            axs[ind + 1].contourf(lons, lats, sp, norm=m.norm, levels=levels, extend="both")
            axs[ind + 1].scatter(-99.5, 30.02, marker="x", s=20, c="coral")
            if "precip_class" in res:
                clss = res.precip_class.data[lat_mask][..., lon_mask]
            else:
                clss = np.zeros_like(sp)
                clss[sp < seeps.dry_threshold] = 0
                clss[seeps.dry_threshold <= sp] = 1
                clss[spt <= sp] = 2
            lvls = np.arange(3) + 0.5
            axs[ind + 1].contour(lons, lats, clss, levels=lvls, colors="grey", linestyles=[":", "--"])

            #axs[ind + 1].contour(lons, lats, sp_ref, norm=m.norm, levels=levels, colors="w")
            valid = np.isfinite(sp) * np.isfinite(sp_ref)

            bias = 100 * (sp[valid] - sp_ref[valid]).mean() / sp_ref[valid].mean()
            corr = np.corrcoef(sp[valid], sp_ref[valid])[0, 1]
            rmse = np.sqrt(((sp[valid] - sp_ref[valid]) ** 2).mean())
            metrics = f"Bias:  {bias:.2f} %\nRMSE:  {rmse:.2f}\nCorr.: {corr:.2f}"
            axs[ind + 1].text(0.8, 0.9, metrics, c="grey", ha="left", va="top", transform=axs[ind + 1].transAxes)
            axs[ind + 1].coastlines()

        fig.suptitle(f"Forecasted precipitation at {valid_time} initialized at {time}.")

        cax = fig.add_subplot(gs[-1, :])
        plt.colorbar(m, label="Surface precip", cax=cax, orientation="horizontal")


    def plot_forecast_evolution(
            self,
            index: int,
            steps: Optional[List[int]] = None,
            include_metrics: bool = False,
            titles: Optional[List[str]] = None,
            bounds: Optional[Tuple[float, float, float, float]] = None
    ):
        """
        Plot forecast results for a give forecast step.

        Args:
            index: The sample index.
            step: The forecast step to display.

        Return:
            The matplotlib figure containing the visualized results.
        """
        if titles is None:
            titles = list(self.result_files.keys())

        if steps is None:
            steps = [28, 20, 12, 4]

        n_fcst = len(self.result_files)
        crs = ccrs.PlateCarree()

        n_steps = len(steps)
        gs = GridSpec(
            n_steps + 2,
            n_fcst + 1,
            width_ratios=[0.05] + [1.0] * n_fcst,
            height_ratios=[1.0] * (n_steps + 1) + [0.075],
            wspace=0.05,
            hspace=0.05,

        )
        fig = plt.figure(figsize=(n_fcst * 5, 3.0 * n_steps + 1 + 1))

        valid_time = self.ref_times[index]

        for step_ind, step in enumerate(steps):

            results = {}
            lead_time = timedelta(hours=3 * step)
            init_time = valid_time - lead_time
            norm = LogNorm(1e-1, 20)
            levels = np.logspace(-1, 1, 11)

            for fcst, res_files in self.result_files.items():
                with xr.open_dataset(res_files[init_time], engine=\"h5netcdf\", chunks=None, cache=False) as data:
                    data = data.sel(valid_time=valid_time).compute()
                results[fcst] = data

            tax = fig.add_subplot(gs[step_ind, 0])
            tax.set_axis_off()
            tax.text(0, 0, s=f"Lead Time = {step * 3} h", rotation=90, ha="center", va="center")
            tax.set_ylim(-2, 2)

            for ind, (name, res) in enumerate(results.items()):

                # Plot reference
                if step_ind == 0:
                    with xr.open_dataset(self.reference_files[valid_time], engine=\"h5netcdf\", chunks=None, cache=False) as ref_data:
            results_ref = ref_data.load().copy(deep=True)
                    sp_ref = results_ref.surface_precip.data
                    if "radar_quality_index" in results_ref:
                        rqi = results_ref.radar_quality_index.data
                    else:
                        rqi = np.ones_like(sp_ref)

                    lat_mask = np.isfinite(sp_ref).any(-1)
                    lon_mask = np.isfinite(sp_ref).any(-2)
                    sp_ref = sp_ref[..., lat_mask, :][..., lon_mask]
                    rqi = rqi[..., lat_mask, :][..., lon_mask]
                    sp_ref[rqi < 0.5] = np.nan
                    lons = results_ref.longitude.data[lon_mask]
                    lats = results_ref.latitude.data[lat_mask]

                    seeps = SEEPS()
                    seeps.calculate_climatology(sorted(list(self.reference_files.values())))
                    clim = seeps.climatology
                    hour = valid_time.hour
                    clim = clim.interp(hour=hour, longitude=lons, latitude=lats)

                    sp_ref = np.maximum(sp_ref, 1e-3)
                    ax = fig.add_subplot(gs[-2, ind + 1], projection=crs)
                    m = ax.contourf(lons, lats, sp_ref, norm=norm, levels=levels, extend="both")

                    spt = clim.surface_precip_second_tercile.data
                    #clss = np.zeros_like(sp_ref)
                    #clss[sp_ref < seeps.dry_threshold] = 0
                    #clss[seeps.dry_threshold <= sp_ref] = 1
                    #clss[spt <= sp_ref] = 2
                    #lvls = np.arange(3) + 0.5
                    #ax.contour(lons, lats, clss, levels=lvls, colors="grey", linestyles=[":", "--"])

                    ax.scatter(-99.5, 30.02, marker="x", s=20, c="coral")
                    ax.coastlines()
                    ax.add_feature(cfeature.BORDERS)
                    ax.add_feature(cfeature.STATES)

                    if ind == 0:
                        tax = fig.add_subplot(gs[-2, 0])
                        tax.set_axis_off()
                        tax.text(0, 0, s="Verification", rotation=90, ha="center", va="center")
                        tax.set_ylim(-2, 2)

                    if bounds is not None:
                        lon_min, lat_min, lon_max, lat_max = bounds
                        ax.set_xlim(lon_min, lon_max)
                        ax.set_ylim(lat_min, lat_max)


                ax = fig.add_subplot(gs[step_ind, ind + 1], projection=crs)
                if step_ind == 0:
                    ax.set_title(titles[ind], loc="center")

                sp = np.maximum(res.surface_precip.data[lat_mask][..., lon_mask], 1e-3)
                ax.contourf(lons, lats, sp, norm=m.norm, levels=levels, extend="both")
                ax.scatter(-99.5, 30.02, marker="x", s=20, c="coral")
                #if "precip_class" in res:
                #    clss = res.precip_class.data[lat_mask][..., lon_mask]
                #else:
                #    clss = np.zeros_like(sp)
                #    clss[sp < seeps.dry_threshold] = 0
                #    clss[seeps.dry_threshold <= sp] = 1
                #    clss[spt <= sp] = 2
                #lvls = np.arange(3) + 0.5
                #ax.contour(lons, lats, clss, levels=lvls, colors="grey", linestyles=[":", "--"])

                #axs[ind + 1].contour(lons, lats, sp_ref, norm=m.norm, levels=levels, colors="w")
                valid = np.isfinite(sp) * np.isfinite(sp_ref)

                bias = 100 * (sp[valid] - sp_ref[valid]).mean() / sp_ref[valid].mean()
                corr = np.corrcoef(sp[valid], sp_ref[valid])[0, 1]
                rmse = np.sqrt(((sp[valid] - sp_ref[valid]) ** 2).mean())
                metrics = f"Bias:  {bias:.2f} %\nRMSE:  {rmse:.2f}\nCorr.: {corr:.2f}"
                if include_metrics:
                    ax.text(
                        0.65, 0.1, metrics, c="coral", ha="left", va="bottom",
                        transform=ax.transAxes, fontsize=15, fontweight="bold"
                    )
                ax.coastlines()
                ax.add_feature(cfeature.BORDERS)
                ax.add_feature(cfeature.STATES)

                if bounds is not None:
                    lon_min, lat_min, lon_max, lat_max = bounds
                    ax.set_xlim(lon_min, lon_max)
                    ax.set_ylim(lat_min, lat_max)


            fig.suptitle(f"Forecasted Precipitation at {valid_time.strftime('%Y-%m-%d %H:%M')}", y=0.92)

        cax = fig.add_subplot(gs[-1, 1:])
        plt.colorbar(m, label="Surface Precip [mm h$^{-1}$]", cax=cax, orientation="horizontal")


    def animate_results(self, index: int, ref_name: str = None, names = None):
        """
        Display forecast results as animation.

        Args:
            index: The index of the evaluation sample to visualize.

        Return:
            The matplotlib func animation.
        """
        n_fcst = len(self.result_files)

        gs = GridSpec(1, n_fcst + 2, width_ratios = [1.0] * (1 + n_fcst) + [0.1], hspace=0.05, wspace=0.05)
        fig = plt.figure(figsize=(6 * (n_fcst + 1) + 1, 4))
        crs = ccrs.PlateCarree()
        axs = [fig.add_subplot(gs[0, ind], projection=crs) for ind in range(n_fcst + 1)]
        axs.append(fig.add_subplot(gs[0, -1]))

        time = self.all_times[index]
        init_time = time

        results = {}

        for fcst, res_files in self.result_files.items():
            with xr.open_dataset(res_files[time], engine=\"h5netcdf\", chunks=None, cache=False) as res_data:
                results[fcst] = res_data.load().copy(deep=True)

        valid_times = next(iter(results.values())).valid_time.data
        valid_time = valid_times[0].astype("datetime64[s]").item()

        with xr.open_dataset(self.reference_files[valid_time], engine=\"h5netcdf\", chunks=None, cache=False) as ref_data:
            results_ref = ref_data.load().copy(deep=True)

        # Determin lat/lon mask
        sp_ref = results_ref.surface_precip.data
        lat_mask = np.isfinite(sp_ref).any(-1)
        lon_mask = np.isfinite(sp_ref).any(-2)
        sp_ref = sp_ref[..., lat_mask, :][..., lon_mask]
        lons = results_ref.longitude.data[lon_mask]
        lats = results_ref.latitude.data[lat_mask]

        extent = [lons.min(), lons.max(), lats.min(), lats.max()]

        norm = LogNorm(1e-1, 1e2)
        img_ref = axs[0].imshow(sp_ref, norm=norm, animated=True, extent=extent, origin="lower")
        axs[0].coastlines()
        if ref_name is not None:
            axs[0].set_title("(a) " + ref_name)

        lead_time = valid_time - init_time
        lead_time_h = lead_time.total_seconds() // 3600
        title  = fig.suptitle(f"{init_time.strftime('%Y-%m-%d %H:%M')} + {int(lead_time_h):02} h")

        img_res = []
        for ind, (name, res) in enumerate(results.items()):

            if names is not None:
                axs[ind + 1].set_title(f"({chr(ord('a') + ind + 1)}) " + names[ind])
            else:
                axs[ind + 1].set_title(name)
            sp = res.surface_precip.interp(valid_time=valid_time, method="nearest").data[lat_mask][..., lon_mask]
            img_res.append(axs[ind + 1].imshow(sp, norm=norm, extent=extent, origin="lower"))
            axs[ind + 1].coastlines()

        plt.colorbar(img_ref, label="Surface precip", cax=axs[-1])


        def update(step: int):

            valid_time = valid_times[step].astype("datetime64[s]").item()
            lead_time = valid_time - init_time
            lead_time_h = lead_time.total_seconds() // 3600
            title.set_text(f"{init_time.strftime('%Y-%m-%d %H:%M')} + {int(lead_time_h):02} h")

            if valid_time in self.reference_files:
                with xr.open_dataset(self.reference_files[valid_time], engine=\"h5netcdf\", chunks=None, cache=False) as ref_data:
            results_ref = ref_data.load().copy(deep=True)
                sp_ref = np.maximum(results_ref.surface_precip.data[lat_mask][..., lon_mask], 1e-3)
                img_ref.set_data(sp_ref)

            for ind, (name, res) in enumerate(results.items()):
                sp = res.surface_precip.interp(valid_time=valid_time, method="nearest").data[lat_mask][..., lon_mask]
                sp = np.maximum(sp, 1e-3)
                img_res[ind].set_data(sp)


            return [img_ref] + img_res

        ani = FuncAnimation(fig, update, frames=len(valid_times), interval=100, blit=True)

        return ani


    def animate_results_only(
            self,
            index: int,
            panel_width: float = 5.0,
            names: Optional[List] = None
    ):
        """
        Like 'animate_results' but only displays the results

        Args:
            index: The index of the evaluation sample to visualize.

        Return:
            The matplotlib func animation.
        """
        n_fcst = len(self.result_files)

        gs = GridSpec(2, n_fcst, width_ratios = [1.0] * n_fcst, height_ratios=[1.0, 0.05], wspace=0.1)
        fig = plt.figure(figsize=(panel_width * n_fcst + 0.04, 4))
        crs = ccrs.PlateCarree()
        axs = [fig.add_subplot(gs[0, ind], projection=crs) for ind in range(n_fcst)]

        time = self.all_times[index]
        init_time = time

        results = {}

        for fcst, res_files in self.result_files.items():
            with xr.open_dataset(res_files[time], engine=\"h5netcdf\", chunks=None, cache=False) as res_data:
                results[fcst] = res_data.load().copy(deep=True)

        valid_times = next(iter(results.values())).valid_time.data
        valid_time = valid_times[0].astype("datetime64[s]").item()

        with xr.open_dataset(self.reference_files[valid_time], engine=\"h5netcdf\", chunks=None, cache=False) as ref_data:
            results_ref = ref_data.load().copy(deep=True)

        # Determin lat/lon mask
        sp_ref = results_ref.surface_precip.data
        lat_mask = np.isfinite(sp_ref).any(-1)
        lon_mask = np.isfinite(sp_ref).any(-2)
        sp_ref = sp_ref[..., lat_mask, :][..., lon_mask]
        lons = results_ref.longitude.data[lon_mask]
        lats = results_ref.latitude.data[lat_mask]

        extent = [lons.min(), lons.max(), lats.min(), lats.max()]

        norm = LogNorm(1e-1, 1e1)

        if names is None:
            names = list(results.keys())

        img_res = []
        for ind, (name, res) in enumerate(results.items()):
            axs[ind].set_title(f"({chr(ord('a') + ind)}) {names[ind]}")
            sp = res.surface_precip.interp(valid_time=valid_time, method="nearest").data[lat_mask][..., lon_mask]
            img_res.append(axs[ind].imshow(sp, norm=norm, extent=extent, origin="lower"))
            axs[ind].coastlines()

        cax = fig.add_subplot(gs[-1, :])
        plt.colorbar(img_res[0], label="Surface precip [mm h$^{-1}$]", cax=cax, orientation="horizontal")

        title = fig.suptitle(f"t = {valid_time.strftime('%Y-%m-%d %H:%M')}")
        lead_time = valid_time - init_time
        lead_time_h = lead_time.total_seconds() // 3600
        title  = fig.suptitle(f"{init_time.strftime('%Y-%m-%d %H:%M')} + {int(lead_time_h):02} h")

        def update(step: int):

            valid_time = valid_times[step].astype("datetime64[s]").item()
            lead_time = valid_time - init_time
            lead_time_h = lead_time.total_seconds() // 3600

            title.set_text(f"{init_time.strftime('%Y-%m-%d %H:%M')} + {int(lead_time_h):02} h")

            for ind, (name, res) in enumerate(results.items()):
                sp = res.surface_precip.interp(valid_time=valid_time, method="nearest").data[lat_mask][..., lon_mask]
                sp = np.maximum(sp, 1e-3)
                img_res[ind].set_data(sp)


            return img_res

        ani = FuncAnimation(fig, update, frames=len(valid_times), interval=200, blit=True)

        return ani


    def plot_stats(self, labels: Optional[List[str]] = None):

        results = self.get_results()

        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(18, 5))
        gs = GridSpec(2, 4, height_ratios=[1.0, 0.3])

        names = list(results.keys())
        if labels is None:
            labels = names

        ax = fig.add_subplot(gs[0, 0])
        handles = []
        for name, label in zip(names, labels):
            lead_time = results[name].lead_time
            handles += ax.plot(lead_time, results[name].nmse, label=label)
        ax.set_title("(a) Normalized MSE")
        ax.set_xlabel("Lead time [h]")
        ax.set_ylabel("NMSE [%]")
        ax.set_xlim(0, 96)
        ax.set_ylim(0, 150)
        ax.grid()

        ax = fig.add_subplot(gs[0, 1])
        for name, label in zip(names, labels):
            lead_time = results[name].lead_time
            ax.plot(lead_time, results[name].correlation_coef, label=label)
        ax.set_xlim(0, 96)
        ax.set_title("(b) Correlation Coefficient")
        ax.set_xlabel("Lead time [h]")
        ax.set_ylabel("Correlation Coef.")
        ax.set_xlim(0, 96)
        ax.set_ylim(0, 1)
        ax.grid()

        ax = fig.add_subplot(gs[0, 2])
        for name, label in zip(names, labels):
            lead_time = results[name].lead_time
            ax.plot(lead_time, results[name].seeps, label=label)
        ax.set_xlim(0, 96)
        ax.set_title("(c) SEEPS")
        ax.set_xlabel("Lead time [h]")
        ax.set_ylabel("SEEPS")
        ax.set_ylim(0, 1)
        ax.grid()

        ax = fig.add_subplot(gs[0, 3])
        for name, label in zip(names, labels):
            lead_time = results[name].lead_time
            ax.plot(lead_time, results[name].crps, label=label)
        ax.set_xlim(0, 96)
        ax.set_title("(d) CRPS")
        ax.set_xlabel("Lead time [h]")
        ax.set_ylabel("CRPS [mm h$^{-1}$]")
        ax.grid()
        ax.set_ylim(0, 0.25)

        ax = fig.add_subplot(gs[1, 1:-1])
        ax.set_axis_off()
        ax.legend(handles=handles, loc="center", frameon=False, ncol=len(labels))

        return fig


    def plot_results_only(
            self,
            index: int,
            panel_width: float = 5.0,
            names: Optional[List] = None
    ):
        """
        Like 'animate_results' but only displays the results

        Args:
            index: The index of the evaluation sample to visualize.

        Return:
            The matplotlib func animation.
        """
        n_fcst = len(self.result_files)

        gs = GridSpec(1, n_fcst + 1, width_ratios = [1.0] * n_fcst + [0.05], wspace=0.1)
        fig = plt.figure(figsize=(panel_width * n_fcst + 0.04, 4))
        crs = ccrs.PlateCarree()
        axs = [fig.add_subplot(gs[0, ind], projection=crs) for ind in range(n_fcst)]

        time = self.all_times[index]
        init_time = time

        results = {}

        for fcst, res_files in self.result_files.items():
            with xr.open_dataset(res_files[time], engine=\"h5netcdf\", chunks=None, cache=False) as res_data:
                results[fcst] = res_data.load().copy(deep=True)

        valid_times = next(iter(results.values())).valid_time.data
        valid_time = valid_times[0].astype("datetime64[s]").item()

        with xr.open_dataset(self.reference_files[valid_time], engine=\"h5netcdf\", chunks=None, cache=False) as ref_data:
            results_ref = ref_data.load().copy(deep=True)

        # Determin lat/lon mask
        sp_ref = results_ref.surface_precip.data
        lat_mask = np.isfinite(sp_ref).any(-1)
        lon_mask = np.isfinite(sp_ref).any(-2)
        sp_ref = sp_ref[..., lat_mask, :][..., lon_mask]
        lons = results_ref.longitude.data[lon_mask]
        lats = results_ref.latitude.data[lat_mask]

        extent = [lons.min(), lons.max(), lats.min(), lats.max()]

        norm = LogNorm(1e-1, 1e1)

        if names is None:
            names = list(results.keys())

        img_res = []
        for ind, (name, res) in enumerate(results.items()):
            axs[ind].set_title(f"({chr(ord('a') + ind)}) {names[ind]}")
            sp = res.surface_precip.interp(valid_time=valid_time, method="nearest").data[lat_mask][..., lon_mask]
            sp = np.maximum(sp, 1e-3)
            img_res.append(axs[ind].imshow(sp, norm=norm, extent=extent, origin="lower"))
            axs[ind].coastlines()

        cax = fig.add_subplot(gs[0, -1])
        plt.colorbar(img_res[0], label="Surface precip", cax=cax)

        title = fig.suptitle(f"t = {valid_time.strftime('%Y-%m-%d %H:%M')}")
        lead_time = valid_time - init_time
        lead_time_h = lead_time.total_seconds() // 3600
        title  = fig.suptitle(f"{init_time.strftime('%Y-%m-%d %H:%M')} + {int(lead_time_h):02} h")

        return fig
