"""
Tests for the prithvi_precip.metrics module.
"""

from concurrent.futures import ProcessPoolExecutor
from typing import List

import numpy as np
from scipy.fftpack import idctn
from scipy import stats
import xarray as xr

from prithvi_precip.metrics import (
    Metric,
    Bias,
    MSE,
    CorrelationCoef,
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
