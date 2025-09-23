"""
prithvi_precip.metrics
======================

Metrics for evaluating Prithvi Precip forecasts.
"""
from functools import cache
import hashlib
from math import ceil
from multiprocessing import shared_memory, Lock, Manager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
from scipy.stats import binned_statistic_dd
from tqdm import tqdm
import xarray as xr


_MANAGER = None


def get_manager() -> Manager:
    """
    Cached access to multi-processing manager.
    """
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = Manager()
    return _MANAGER


class Metric:
    """
    Base class for metrics that manages shared data arrays and can be used to manage the
    access to those arrays.
    """
    def __init__(self, buffers: Dict[str, Tuple[Tuple[int], str]]):
        super().__init__()
        self.lock = get_manager().Lock()
        self._buffers = {}
        for name, (shape, dtype) in buffers.items():
            array = np.zeros(shape, dtype=dtype)
            shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
            array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            array[:] = 0.0
            self._buffers[name] = (shm.name, shape, dtype)
        self.owner = True

    def __getattr__(self, name: str) -> Any:
        buffers = self.__dict__.get("_buffers", None)
        if buffers is not None:
            if name in buffers:
                shm, shape, dtype = buffers[name]
                if isinstance(shm, str):
                    shm = shared_memory.SharedMemory(shm)
                buffers[name] = (shm, shape, dtype)
                return np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def reset(self) -> None:
        """
        Reset metric state.

        Sets all buffers associated with the metric to zero, assuming that this is
        a valid initial state. If this is not the case, the child class should overwrite
        the function.
        """
        for name in self._buffers:
            array = getattr(self, name)
            array[:] = 0.0

    def cleanup(self) -> None:
        """
        Remove shared memory
        """
        if hasattr(self, "_buffers"):
            for name, shm in self._buffers.items():
                shm = shm[0]
                if isinstance(shm, str):
                    shm = shared_memory.SharedMemory(shm)
                shm.unlink()

    def __del__(self):
        """
        Close connections to shared memory.
        """
        if hasattr(self, "_buffers"):
            for name, shm in self._buffers.items():
                shm = shm[0]
                if isinstance(shm, str):
                    shm = shared_memory.SharedMemory(shm)
                shm.close()
                if self.owner:
                    pass
                    # shm.unlink()


class QuantificationMetric(Metric):
    """
    Helper class to identify metrics to assess precipitation quantification.
    """

class ProbabilisticQuantificationMetric(Metric):
    """
    Helper class to identify metrics to assess precipitation quantification.
    """

class Bias(QuantificationMetric):
    """
    The bias, or mean error, calculated as the mean value of the difference between
    prediction and target values:

    .. math::

      \\text{Bias} = \\mathbf{E}\{y_\\text{pred} - y_\\text{target}\}

    where the mean is calculated over all results passed to the 'compute' method for
    which the target values are finite.
    """

    def __init__(self, relative: bool = True):
        """
        Args:
            relative: If True, the bias is calculated as percent of the mean reference
                 precipitation. Else the bias is calculated as absolute value.
        """
        super().__init__(
            buffers={
                "x_sum": ((1,), np.float64),
                "y_sum": ((1,), np.float64),
                "counts": ((1,), np.float64),
            }
        )
        self.relative = relative

    def update(
            self,
            lons: np.ndarray,
            lats: np.ndarray,
            prediction: np.ndarray,
            target: np.ndarray,
    ) -> None:
        """
        Update metric values with given prediction.

        Args:
             prediction: An np.ndarray containing the predicted values.
             target: An np.ndarray containing the reference values.
        """
        pred = prediction
        valid = np.isfinite(target)
        pred = pred[valid]
        target = target[valid]
        lats = lats[valid]
        weights = np.cos(np.deg2rad(lats))

        with self.lock:
            self.x_sum += (weights * pred).sum()
            self.y_sum += (weights * target).sum()
            self.counts += weights.sum()

    def compute(self, name: Optional[str] = None) -> xr.Dataset:
        """
        Calculate the MSE for all results passed to this metric object.

        Return:
            An xarray.Dataset containing a single, scalar variable 'mse' containing the
            MSE for the assessed results.
        """
        with np.errstate(invalid="ignore"):
            if self.relative:
                bias = 100.0 * (self.x_sum - self.y_sum) / self.y_sum
            else:
                bias = (self.x_sum - self.y_sum) / self.counts

        bias = xr.Dataset({"bias": bias[0]})
        bias.bias.attrs["full_name"] = "Bias"
        bias.bias.attrs["unit"] = "\%" if self.relative else "mm h^{-1}"
        return bias


class MSE(QuantificationMetric):
    """
    The mean-squared error calculated as the mean value of the squared difference between
    prediction and target values:

    .. math::

      \\text{MSE} = (\\mathbf{E}\{y_\\text{pred} - y_\\text{target}\})^2

    where mean is calculated over all results passed to the 'compute' method for
    which the target values are finite.
    """

    def __init__(self):
        super().__init__(
            buffers={
                "tot_sq_error": ((1,), np.float64),
                "counts": ((1,), np.float64),
            }
        )

    def update(
            self,
            lons: np.ndarray,
            lats: np.ndarray,
            prediction: np.ndarray,
            target: np.ndarray,
    ) -> None:
        """
        Update metric values with given prediction.

        Args:
            lons: The longitude coordinates of the results.
            lats: The latitude coordinates of the results.
            prediction: An np.ndarray containing the predicted values.
            target: An np.ndarray containing the reference values.
        """
        pred = prediction
        valid = np.isfinite(target)
        pred = pred[valid]
        target = target[valid]
        lats = lats[valid]
        weights = np.cos(np.deg2rad(lats))

        with self.lock:
            self.tot_sq_error += (((pred - target) ** 2) * weights).sum()
            self.counts += weights.sum()

    def compute(self) -> xr.Dataset:
        """
        Calculate the MSE for all results passed to this metric object.

        Return:
            An xarray.Dataset containing a single, scalar variable 'mse' representing
            the MSE calculated over all results passed to this metric object.
        """
        with np.errstate(invalid='ignore'):
            mse = xr.Dataset({"mse": (self.tot_sq_error / self.counts)[0]})
        mse.mse.attrs["full_name"] = "MSE"
        mse.mse.attrs["unit"] = "(mm h^{-1})^2"
        return mse


class CorrelationCoef(QuantificationMetric):
    """
    The linear correlation coefficient between predictions and target values.

    .. math::

      \\text{Correlation coeff.} = \\mathbf{E}\\frac{
      (y_\\text{pred} - \\mu_{y_\\text{pred}})(y_\\text{target} - \\mu{y_\\text{target})}
      }{
       \\sigma_{y_\\text{pred}} \sigma_{y_\\text{target}}
      }


    where the mean is calculated over all results passed to the 'compute' method for
    which the target values are finite and :math:`\\mu` and :math:`\\sigma` are used to denote
    the mean and standard deviations of the distributions of :math:`y_\text{pred}` and
    :math:`y_\\text{target}`.
    """

    def __init__(self):
        super().__init__(
            buffers={
                "x_sum": ((1,), np.float64),
                "x2_sum": ((1,), np.float64),
                "y_sum": ((1,), np.float64),
                "y2_sum": ((1,), np.float64),
                "xy_sum": ((1,), np.float64),
                "counts": ((1,), np.float64),
            }
        )

    def update(
            self,
            lons: np.ndarray,
            lats: np.ndarray,
            prediction: np.ndarray,
            target: np.ndarray,
    ) -> None:
        """
        Update metric values with given prediction.

        Args:
            lons: The longitude coordinates of the results.
            lats: The latitude coordinates of the results.
            prediction: An np.ndarray containing the predicted values.
            target: An np.ndarray containing the reference values.
        """

        pred = prediction
        valid = np.isfinite(target)

        pred = pred[valid]
        target = target[valid]
        lats = lats[valid]
        weights = np.cos(np.deg2rad(lats))

        with self.lock:
            self.x_sum += (pred * weights).sum()
            self.x2_sum += ((pred**2) * weights).sum()
            self.y_sum += (target * weights).sum()
            self.y2_sum += ((target**2) * weights).sum()
            self.xy_sum += ((pred * target) * weights).sum()
            self.counts += weights.sum()


    def compute(self) -> xr.Dataset:
        """
        Calculate the bias for all results passed to this metric object.

        Return:
            An xarray.Dataset containing a single, scalar variable 'bias' or 'bias_{name}'.

        """
        with np.errstate(invalid='ignore'):
            x_mean = self.x_sum / self.counts
            x2_mean = self.x2_sum / self.counts
            x_sigma = np.sqrt(x2_mean - x_mean**2)
            y_mean = self.y_sum / self.counts
            y2_mean = self.y2_sum / self.counts
            y_sigma = np.sqrt(y2_mean - y_mean**2)
            xy_mean = self.xy_sum / self.counts
            corr = (xy_mean - x_mean * y_mean) / (x_sigma * y_sigma)

        corr = xr.Dataset({"correlation_coef": corr[0]})
        corr.correlation_coef.attrs["full_name"] = "Correlation coeff."
        corr.correlation_coef.attrs["unit"] = ""
        return corr


class CRPS(ProbabilisticQuantificationMetric):
    """
    The continuous ranked probability score supporting both deterministic and quantile predictions.

    This metric calculates the CRPS using
    .. math::

      \\text{CRPS} = 2.0 \\int_0^1 (\\tau - I_{y_\\text{target} < y_\\text{pred}}) (y_\\text{target} - y_\\text{pred}) d\\tau

    If the prediction is deterministic, this metric simply calculates the mean-absolute error. If the prediction consists
    of several quantiles the integral above is approximated using the trapezoidal rule.
    """
    def __init__(self):
        super().__init__(
            buffers={
                "crps": ((1,), np.float64),
                "counts": ((1,), np.float64),
            }
        )

    def update(
            self,
            lons: np.ndarray,
            lats: np.ndarray,
            prediction: np.ndarray,
            target: np.ndarray,
            taus: Optional[np.ndarray] = None
    ) -> None:
        """
        Update metric values with given prediction.

        Args:
            lons: The longitude coordinates of the results.
            lats: The latitude coordinates of the results.
            prediction: An np.ndarray containing the predicted values.
            target: An np.ndarray containing the reference values.
        """
        pred = prediction
        if pred.ndim == target.ndim:
            pred = pred[None]
            taus = 0.5 * np.ones(1)
        else:
            if prediction.ndim != target.ndim + 1:
                raise ValueError(
                    "The prediction for the CRPS score should have the same or one more dimension "
                    "than the target"
                )
            if taus is None:
                raise ValueError(
                    "If the prediction for the CRPS score has one more dimension than the target, the "
                    "corresponding quantiles must be provided as 'tau'."
                )

        valid = np.isfinite(target)
        if valid.sum() < 1:
            return None

        pred = pred[..., valid]
        target = target[valid][None]
        lons = lons[valid]
        lats = lats[valid]
        weights = np.cos(np.deg2rad(lats))

        target = np.broadcast_to(target, pred.shape)
        taus = np.broadcast_to(taus[..., None], target.shape)
        diff = target - pred
        crps = np.where(0 < diff, taus, taus - 1.0) * diff

        if crps.shape[0] == 1:
            crps = 2.0 * np.abs(crps)[0]
        else:
            crps = 2.0 * np.trapz(crps, x=taus, axis=0)

        with self.lock:
            self.crps += (crps * weights).sum()
            self.counts += weights.sum()


    def compute(self) -> xr.Dataset:
        """
        Calculate the bias for all results passed to this metric object.

        Return:
            An xarray.Dataset containing a single, scalar variable 'bias' or 'bias_{name}'.

        """
        with np.errstate(invalid='ignore'):
            crps = self.crps / self.counts

        crps = xr.Dataset({"crps": crps[0]})
        crps.crps.attrs["full_name"] = "CRPS"
        crps.crps.attrs["unit"] = ""
        return crps


class SEEPS(QuantificationMetric):
    """

    """

    def __init__(self, resolution: float = 2.0):
        super().__init__(
            buffers={
                "seeps": ((1,), np.float64),
                "counts": ((1,), np.float64),
            }
        )
        self.dry_threshold = 0.01
        self.resolution = resolution
        self.climatology = None


    def calculate_climatology(
            self,
            reference_files: List[Path]
    ) -> None:
        """
        Calculate climatology to distinguish dry, light, and heavy precip.
        """

        reference_files = sorted(reference_files)
        ref_file = reference_files[0]
        normalized = str(ref_file.resolve())
        hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        fname = f"seeps_climatology_{hash}.nc"
        if Path(fname).exists():
            result = xr.load_dataset(fname)
            self.climatology = result
            return result


        sp_bins = np.logspace(np.log10(self.dry_threshold), np.log10(2e2), 101)
        n_lons = ceil(360 / self.resolution)
        lon_bins = np.linspace(-180, 180, n_lons + 1)
        n_lats = ceil(180 / self.resolution)
        lat_bins = np.linspace(-90, 90, n_lats + 1)

        dists = np.zeros((n_lats, n_lons, sp_bins.size - 1))
        dry = np.zeros((n_lats, n_lons))
        cts = np.zeros((n_lats, n_lons))

        for path in tqdm(reference_files):
            try:
                with xr.open_dataset(path) as data:
                    lons = data.longitude.load().data
                    lats = data.latitude.load().data
                    lons, lats = np.meshgrid(lons, lats)
                    lons = lons.ravel()
                    lats = lats.ravel()
                    sp = data.surface_precip.data.ravel()
                    dists += binned_statistic_dd([lats, lons, sp], sp, bins=(lat_bins, lon_bins, sp_bins), statistic="count")[0]
                    dry += binned_statistic_dd([lats, lons], sp < self.dry_threshold, bins=(lat_bins, lon_bins), statistic="sum")[0]
                    cts += binned_statistic_dd([lats, lons], 0 <= sp, bins=(lat_bins, lon_bins), statistic="sum")[0]
            except ValueError:
                pass

        d_sp = np.diff(sp_bins)
        pdf = dists / dists.sum(axis=-1, keepdims=True) / d_sp[None, None]
        cdf = (pdf * d_sp[None, None]).cumsum(axis=-1)

        result = xr.Dataset({
            "latitude": (("latitude",), 0.5 * (lat_bins[1:] + lat_bins[:-1])),
            "longitude": (("longitude",), 0.5 * (lon_bins[1:] + lon_bins[:-1])),
            "surface_precip": (("surface_precip",), 0.5 * (sp_bins[1:] + sp_bins[:-1])),
            "cdf": (("latitude", "longitude", "surface_precip"), cdf)
        })

        p_dry = dry / cts
        p_dry[cts < 1] = np.nan
        result["p_dry"] = (("latitude", "longitude"), p_dry)


        tau_67 = np.zeros((n_lats, n_lons))
        for lat_ind in range(n_lats):
            for lon_ind in range(n_lons):
                tau_67[lat_ind, lon_ind] = np.interp(
                    2.0 / 3.0,
                    result.cdf.data[lat_ind, lon_ind],
                    result.surface_precip.data
                )

        result["surface_precip_second_tercile"] = (("latitude", "longitude"), tau_67)
        result.to_netcdf(fname)
        self.climatology = result
        return result


    def update(
            self,
            lons: np.ndarray,
            lats: np.ndarray,
            prediction: np.ndarray,
            target: np.ndarray,
    ) -> None:
        """
        Update metric values with given prediction.

        Args:
             prediction: An np.ndarray containing the predicted values.
             target: An np.ndarray containing the reference values.
        """
        if self.climatology is None:
            raise ValueError(
                "Need to calculate climatology before the SEEPS score can be calculated."
            )

        pred = prediction
        valid = np.isfinite(target)
        pred = pred[valid]
        target = target[valid]
        lons = lons[valid]
        lats = lats[valid]
        weights = np.cos(np.deg2rad(lats))

        coords = xr.Dataset({
            "latitude": (("samples",), lats),
            "longitude": (("samples",), lons),
        })
        try:
            clim = self.climatology.interp(latitude=coords.latitude, longitude=coords.longitude)

            spt = clim.surface_precip_second_tercile.data
            p_1 = clim.p_dry.data
            p_2 = (1.0 - p_1) * 2.0 / 3.0
            p_3 = (1.0 - p_1) * 1.0 / 3.0

            classes_target = np.zeros_like(target, dtype=np.uint64)
            classes_target[target < self.dry_threshold] = 0
            classes_target[self.dry_threshold <= target] = 1
            classes_target[spt <= target] = 2

            classes_pred = np.zeros_like(pred, dtype=np.uint64)
            classes_pred[pred < self.dry_threshold] = 0
            classes_pred[self.dry_threshold <= pred] = 1
            classes_pred[spt <= pred] = 2
            mask = spt <= pred

            zeros = np.zeros_like(p_1)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                costs = np.stack(np.array([
                    np.stack([zeros, 1.0 - (1.0 - p_1), 1.0 / p_3 + 1.0 / (1.0 - p_1)]),
                    np.stack([1.0 / p_1, zeros, 1.0 / p_3]),
                    np.stack([1 / p_1 + 1.0 / (1.0 - p_3), 1.0 / (1.0 - p_3), zeros])
                ]))

            scores = 0.5 * costs[classes_pred, classes_target, np.arange(p_1.size)]
            valid = np.isfinite(scores)
            scores = scores[valid]
            weights = np.ones_like(weights[valid])

            with self.lock:
                self.seeps += (weights * scores).sum()
                self.counts += weights.sum()
        except Exception:
            pass


    def compute(self, name: Optional[str] = None) -> xr.Dataset:
        """
        Calculate the MSE for all results passed to this metric object.

        Return:
            An xarray.Dataset containing a single, scalar variable 'mse' containing the
            MSE for the assessed results.
        """
        with np.errstate(invalid="ignore"):
            seeps = self.seeps / self.counts

        seeps = xr.Dataset({"seeps": seeps[0]})
        seeps.seeps.attrs["full_name"] = "SEEPS"
        seeps.seeps.attrs["unit"] = ""
        return seeps



class ACC(CorrelationCoef):
    """
    Anomaly Correlation Coefficient (ACC)
    """

    def __init__(self, resolution: float = 1.0):
        super().__init__()
        self.resolution = resolution
        self.climatology = None


    def calculate_climatology(
            self,
            reference_files: List[Path]
    ) -> None:
        """
        Calculate climatology to distinguish dry, light, and heavy precip.
        """

        reference_files = sorted(reference_files)
        ref_file = reference_files[0]
        normalized = str(ref_file.resolve())
        hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        fname = f"acc_climatology_{hash}.nc"
        if Path(fname).exists():
            result = xr.load_dataset(fname)
            self.climatology = result
            return result


        n_lons = ceil(360 / self.resolution)
        lon_bins = np.linspace(-180, 180, n_lons + 1)
        n_lats = ceil(180 / self.resolution)
        lat_bins = np.linspace(-90, 90, n_lats + 1)

        sp_tot = np.zeros((n_lats, n_lons))
        sp_cts = np.zeros((n_lats, n_lons))

        for path in tqdm(reference_files):
            with xr.open_dataset(path) as data:
                lons = data.longitude.load().data
                lats = data.latitude.load().data
                lons, lats = np.meshgrid(lons, lats)
                lons = lons.ravel()
                lats = lats.ravel()
                sp = data.surface_precip.data.ravel()
                sp_tot += binned_statistic_dd([lats, lons], sp, bins=(lat_bins, lon_bins), statistic="sum")[0]
                sp_cts += binned_statistic_dd([lats, lons], 0 <= sp, bins=(lat_bins, lon_bins), statistic="sum")[0]

        result = xr.Dataset({
            "latitude": (("latitude",), 0.5 * (lat_bins[1:] + lat_bins[:-1])),
            "longitude": (("longitude",), 0.5 * (lon_bins[1:] + lon_bins[:-1])),
            "surface_precip": (("latitude", "longitude",), sp_tot / sp_cts)
        })
        result.to_netcdf(fname)
        self.climatology = result
        return result


    def compute(self) -> xr.Dataset:
        """
        Calculate the bias for all results passed to this metric object.

        Return:
            An xarray.Dataset containing a single, scalar variable 'bias' or 'bias_{name}'.

        """
        res = super().compute().rename(correlation_coef="acc")
        res.acc.attrs["full_name"] = "Anomaly correlation coef."
        res.acc.attrs["unit"] = ""
        return res

    def update(
            self,
            lons: np.ndarray,
            lats: np.ndarray,
            prediction: np.ndarray,
            target: np.ndarray,
    ) -> None:
        """
        Update metric values with given prediction.

        Args:
             prediction: An np.ndarray containing the predicted values.
             target: An np.ndarray containing the reference values.
        """
        if self.climatology is None:
            raise ValueError(
                "Need to calculate climatology before the SEEPS score can be calculated."
            )

        pred = prediction
        valid = np.isfinite(target)
        if valid.sum() < 1:
            return None

        pred = pred[valid]
        target = target[valid]
        lons = lons[valid]
        lats = lats[valid]

        coords = xr.Dataset({
            "latitude": (("samples",), lats),
            "longitude": (("samples",), lons),
        })

        clim = self.climatology.interp(latitude=coords.latitude, longitude=coords.longitude)

        CorrelationCoef.update(self, lons, lats, pred - clim.surface_precip.data, target - clim.surface_precip.data)
