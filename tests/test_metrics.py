"""
Tests for the prithvi_precip.metrics module.
"""

from concurrent.futures import ProcessPoolExecutor
from typing import List

import numpy as np
import pytest
from scipy.fftpack import idctn
from scipy import stats
import xarray as xr

from prithvi_precip.metrics import (
    Metric,
    Bias,
    MSE,
    CorrelationCoef,
    CRPS,
    ACC,
    SEEPS
)


def evaluate_normal_preds(metric: Metric) -> None:
    """
    Helper function that  evaluates the given metric with
    random values from two Normal distributions centered on 0 for
    the predictions and 10 for the target values.
    """
    x = np.random.normal(size=(100, 100))
    y = np.random.normal(size=(100, 100)) + 10
    lons = np.zeros_like(x)
    lats = np.zeros_like(x)
    metric.update(lons, lats, x, y)


def evaluate_normal_preds_with_invalid(metric: Metric) -> None:
    """
    Same as evaluate_normal_preds but predictions are set to NAN with a probability
    of 50%.
    """
    x = np.random.normal(size=(100, 100))
    x[np.random.rand(*x.shape) > 0.5] = np.nan
    y = np.random.normal(size=(100, 100)) + 10
    lons = np.zeros_like(x)
    lats = np.zeros_like(x)
    metric.update(lons, lats, x, y)


def test_bias():
    """
    Test calculation of the bias.
    """
    n_jobs = 1024
    pool = ProcessPoolExecutor(max_workers=8)

    bias = Bias()
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_normal_preds, bias))

    for task in tasks:
        task.result()

    result = bias.compute()
    assert np.isclose(result.bias.data, -100.0, rtol=1e-2)

    bias = Bias(relative=False)
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_normal_preds, bias))

    for task in tasks:
        task.result()

    result = bias.compute()
    assert np.isclose(result.bias.data, -10.0, atol=1e-2)


def evaluate_fixed(metric: Metric) -> None:
    """
    Helper function the evaluated the given metric with fixed predictions
    with the value 0 and fixed targets with the value 1.
    """
    x = np.zeros((100, 100))
    y = np.ones_like(x)
    lons = np.zeros_like(x)
    lats = np.zeros_like(x)
    metric.update(lons, lats, x, y)


def test_mse():
    """
    Ensure that the calculated MSE is close to 102.
    """
    n_jobs = 1024
    pool = ProcessPoolExecutor(max_workers=8)

    mse = MSE()
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_normal_preds, mse))

    for task in tasks:
        task.result()

    result = mse.compute()
    assert np.isclose(result.mse.data, 102, atol=1e-1)


def test_correlation_coef_indep():
    """
    Ensure that the calculated correlation coefficient is close to 0 for
    completely independent random predictions and targets.
    """
    n_jobs = 1024
    pool = ProcessPoolExecutor(max_workers=8)

    corr_coef = CorrelationCoef()
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_normal_preds, corr_coef))

    for task in tasks:
        task.result()

    result = corr_coef.compute()
    assert np.isclose(result.correlation_coef.data, 0.0, atol=1e-2)


def evaluate_dependent_preds(metric: Metric) -> None:
    """
    Helper function that evaluates evaluates the given metric with
    random values from a Normal distributions where the target
    y is simply y = 2 * x.
    """
    x = np.random.normal(size=(100, 100))
    y = 2.0 * x
    lons = np.zeros_like(x)
    lats = np.zeros_like(x)
    metric.update(lons, lats, x, y)


def evaluate_anticorrelated_preds(metric: Metric) -> None:
    """
    Helper function that evaluates evaluates the given metric with
    random values from a Normal distributions where the target
    y is simply y = - 2 * x.
    """
    x = np.random.normal(size=(100, 100))
    y = -2.0 * x
    lons = np.zeros_like(x)
    lats = np.zeros_like(x)
    metric.update(lons, lats, x, y)


def test_correlation_coef_dep():
    """
    Ensure that the calculated correlation coefficient is close to -1 for
    for perfectly anti-correlated predictions and targets.
    """
    n_jobs = 1024
    pool = ProcessPoolExecutor(max_workers=8)

    corr_coef = CorrelationCoef()
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_dependent_preds, corr_coef))
    for task in tasks:
        task.result()
    result = corr_coef.compute()
    assert np.isclose(result.correlation_coef.data, 1.0, atol=1e-2)

    corr_coef.reset()
    corr_coef = CorrelationCoef()
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_anticorrelated_preds, corr_coef))
    for task in tasks:
        task.result()
    result = corr_coef.compute()
    assert np.isclose(result.correlation_coef.data, -1.0, atol=1e-2)


def crps_normal(z):
    """
    Closed form of CRPS score for a Normal reference distribution and a point value z.

    Args:
        z: The point value.

    Returns:
        The CRPS value of a predicted normal distribution for the realization z.

    """
    return z * (2.0 * stats.norm.cdf(z) - 1.0) + 2.0 * stats.norm.pdf(z) - 1.0 / np.sqrt(np.pi)


def test_crps():
    """
    Tests CRPS metric using the closed form available for Gaussian distributions.
    """

    # Prediction are 32 quantiles of a Gaussian distribution
    truths = np.random.uniform(size=10_000)
    lons = np.zeros_like(truths)
    lats = np.zeros_like(truths)
    quantiles = np.linspace(0, 1, 34)[1:-1]
    pred = stats.norm.ppf(quantiles)[..., None]
    pred = np.broadcast_to(pred, (32,) + truths.shape)

    crps_ref = crps_normal(truths).mean()

    crps = CRPS()
    crps.update(lons, lats, pred, truths, taus=quantiles)
    res = crps.compute()
    assert np.isclose(res.crps.data, crps_ref, rtol=0.02)

    # For scalar predictions, the CRPS should just be the MAE
    truths = np.random.uniform(size=10_000)
    pred = np.random.uniform(size=10_000)
    lons = np.zeros_like(truths)
    lats = np.zeros_like(truths)

    crps_ref = np.abs(pred - truths).mean()
    crps = CRPS()
    crps.update(lons, lats, pred, truths)
    res = crps.compute()
    assert np.isclose(res.crps.data, crps_ref, rtol=0.02)


@pytest.fixture
def acc_test_data_without_background(tmp_path):
    """
    ACC test data without background signal.
    """
    output_path = tmp_path / "no_background"
    output_path.mkdir()

    lons = np.linspace(-180, 180, 256)
    lats = np.linspace(-90, 90, 256)

    for time in np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-06-01"),
            np.timedelta64(1, "D")
    ):
        date = time.astype("datetime64[s]").item()
        fname = date.strftime("precip_%Y%m%d%H%M%S.nc")
        precip = np.random.normal(size=(256, 256))
        data = xr.Dataset({
            "longitude": (("longitude",), lons),
            "latitude": (("latitude",), lats),
            "surface_precip": (("latitude", "latitude"), precip)
        })
        data.to_netcdf(output_path / fname)
    return output_path


@pytest.fixture
def acc_test_data_with_background(tmp_path):
    """
    ACC test data no background.
    """
    output_path = tmp_path / "background"
    output_path.mkdir()

    lons = np.linspace(-180, 180, 256)
    lats = np.linspace(-90, 90, 256)

    for time in np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-06-01"),
            np.timedelta64(1, "D")
    ):
        date = time.astype("datetime64[s]").item()
        fname = date.strftime("precip_%Y%m%d%H%M%S.nc")
        precip = np.random.normal(size=(256, 256))
        r = np.sqrt(lons[None] ** 2 + lats[:, None] ** 2)
        data = xr.Dataset({
            "longitude": (("longitude",), lons),
            "latitude": (("latitude",), lats),
            "surface_precip": (("latitude", "latitude"), r + precip)
        })
        data.to_netcdf(output_path / fname)
    return output_path


def test_acc(acc_test_data_without_background, acc_test_data_with_background):
    """
    Tests ACC score using two scenarios:

        1. No backgroun variability, ACC should be same as correlation coeff
        2. Independent signals with shared background variability, ACC should be 0.

    """
    # No background signal, ACC should be the same as CorCoef.
    acc = ACC()
    corr = CorrelationCoef()
    acc.calculate_climatology(list(acc_test_data_without_background.glob("*.nc")))

    lons = np.random.uniform(-180, 180, size=(256, 256))
    lats = np.random.uniform(-90, 90, size=(256, 256))

    truth = np.random.normal(size=(256, 256))
    pred = truth + 0.1 * np.random.normal(size=(256, 256))

    acc.update(lons, lats, pred, truth)
    acc = acc.compute().acc.data

    corr.update(lons, lats, pred, truth)
    corr = corr.compute().correlation_coef.data

    assert np.isclose(acc, corr, rtol=0.01)


    # With background signal, ACC smaller be the same as CorCoef.
    acc = ACC()
    corr = CorrelationCoef()
    acc.calculate_climatology(list(acc_test_data_with_background.glob("*.nc")))

    lons = np.random.uniform(-180, 180, size=(512, 512))
    lats = np.random.uniform(-90, 90, size=(512, 512))
    r = np.sqrt(lons ** 2 + lats ** 2)

    truth = r + np.random.normal(size=(512, 512))
    pred = r + np.random.normal(size=(512, 512))

    acc.update(lons, lats, pred, truth)
    clim = acc.climatology
    acc = acc.compute().acc.data

    corr.update(lons, lats, pred, truth)
    corr = corr.compute().correlation_coef.data

    assert 0 < corr
    assert np.isclose(acc, 0.0, atol=0.05)


@pytest.fixture
def seeps_test_data_with_background(tmp_path):
    """
    SEEPS test data with background.
    """
    output_path = tmp_path / "background"
    output_path.mkdir()

    lons = np.linspace(-180, 180, 256)
    lats = np.linspace(-90, 90, 256)

    for time in np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-12-01"),
            np.timedelta64(1, "D")
    ):
        date = time.astype("datetime64[s]").item()
        fname = date.strftime("precip_%Y%m%d%H%M%S.nc")
        r = np.sqrt(lons[None] ** 2 + lats[:, None] ** 2)
        precip = np.random.exponential(scale=r / 10.0, size=r.shape)
        data = xr.Dataset({
            "longitude": (("longitude",), lons),
            "latitude": (("latitude",), lats),
            "surface_precip": (("latitude", "latitude"), precip)
        })
        data.to_netcdf(output_path / fname)
    return output_path


def test_seeps(seeps_test_data_with_background):
    """
    Tests SEEPS score on an example where precipitation is assumed to follow
    a exponential distribution with spatially dependent scale parameter.
    """
    # With background signal, SEEPS smaller be the same as CorCoef.
    seeps_0 = SEEPS()
    seeps_1 = SEEPS()
    corr = CorrelationCoef()
    files = sorted(list(seeps_test_data_with_background.glob("*.nc")))
    seeps_0.calculate_climatology(files)
    seeps_1.calculate_climatology(files)

    clim = seeps_0.climatology

    medians = np.zeros_like(clim.surface_precip_second_tercile.data)
    for lat_ind in range(clim.latitude.size):
        for lon_ind in range(clim.longitude.size):
            medians[lat_ind, lon_ind] = np.interp(0.5, clim.cdf.data[lat_ind, lon_ind], clim.surface_precip.data)
    clim["median"] = (("latitude", "longitude"), medians)

    for path in files:
        with xr.open_dataset(path) as data:
            lons = data.longitude.load().data
            lats = data.latitude.load().data
            medians_i = clim["median"].interp(latitude=lats, longitude=lons)
            sp = data.surface_precip.load().data
            lons, lats = np.meshgrid(lons, lats)
            seeps_0.update(lons, lats, sp, sp)
            seeps_1.update(lons, lats, medians_i.data, sp)

    seeps = seeps_0.compute().seeps.data
    assert np.isclose(seeps, 0.0, atol=0.1)

    seeps = seeps_1.compute().seeps.data
    assert np.isclose(seeps, 1.0, atol=0.1)
