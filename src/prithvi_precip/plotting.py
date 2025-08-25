"""
prithvi_precip.plotting
=======================
"""
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cartopy.crs as ccrs
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import numpy as np
import torch
import xarray as xr


def plot_tiles(
        tnsr: torch.Tensor,
        global_y,
        global_x,
        local_y,
        local_x,
        channel: Optional[int] = None,
        colorbar: bool = True,
        cmap: Optional[str] = None,
        ax: Optional["Axes"] = None
):
    """
    Plot tiled tensor.
    """
    tnsr = tnsr.detach().cpu()
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(10, 5))
    gap = 3

    n_tiles_y = tnsr.shape[global_y]
    n_tiles_x = tnsr.shape[global_x]
    height = tnsr.shape[local_y]
    width = tnsr.shape[local_x]

    all_tiles = tnsr.squeeze()
    if channel is not None:
        all_tiles = all_tiles[..., channel, :, :]
    norm = Normalize(np.nanmin(all_tiles), np.nanmax(all_tiles))

    for tile_ind_y in range(n_tiles_y):
        for tile_ind_x in range(n_tiles_x):
            tile = tnsr.select(global_y, tile_ind_y).select(global_x - 1, tile_ind_x).squeeze()
            if channel is not None:
                tile = tile[channel]

            start = (width + gap) * tile_ind_x
            end = start + width
            x = np.arange(start, end)
            start = (height + gap) * tile_ind_y
            end = start + height
            y = np.arange(start, end)

            m = ax.pcolormesh(x, y, tile, cmap=cmap)

    if colorbar:
        plt.colorbar(m)


def set_style():
    """
    Set the IPWGML matplotlib style.
    """
    plt.style.use(Path(__file__).parent / "files" / "prithvi_precip.mplstyle")


def animate_results(
        results: Union[Dict[str, xr.Dataset], xr.Dataset],
        panel_width: float = 5.0,
        names: Optional[List] = None,
        variable: str = "surface_precip",
        norm: Optional[Any] = None,
        clabel: str = "Surface precip [mm h$^{-1}$]",
        include_metrics: bool = False,
        n_cols: int = -1
):
    """
    Like 'animate_results' but only displays the results

    Args:
        index: The index of the evaluation sample to visualize.

    Return:
        The matplotlib func animation.
    """
    if isinstance(results, xr.Dataset):
        results = {"Prithvi Precip": results}

    n_fcst = len(results)
    if n_cols < 0:
        n_rows = 1
        n_cols = n_fcst
    else:
        n_rows = ceil(n_fcst / n_cols)

    gs = GridSpec(n_rows + 1, n_cols, width_ratios = [1.0] * n_cols, height_ratios=[1.0] * n_rows + [0.05], wspace=0.1)
    fig = plt.figure(figsize=(panel_width * n_cols + 0.04, 4 * n_rows))
    crs = ccrs.PlateCarree()
    axs = [fig.add_subplot(gs[ind // n_cols, ind % n_cols], projection=crs) for ind in range(n_fcst)]

    results_ref = next(iter(results.values()))
    time = results_ref.initialization_time.data
    init_time = time.astype("datetime64[s]").item()

    valid_times = next(iter(results.values())).valid_time.data
    valid_time = valid_times[0].astype("datetime64[s]").item()

    sp_ref = results_ref.surface_precip.data
    lons = results_ref.longitude.data
    lats = results_ref.latitude.data
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    if norm is None:
        norm = LogNorm(1e-1, 1e1)

    if names is None:
        names = list(results.keys())

    if clabel is None:
        clabel = "Surface precip [mm h$^{-1}$]"

    img_res = []
    texts = []

    for ind, (name, res) in enumerate(results.items()):
        axs[ind].set_title(f"({chr(ord('a') + ind)}) {names[ind]}")
        vals = res[variable].data[0]
        img_res.append(axs[ind].imshow(vals, norm=norm, extent=extent, origin="lower"))
        axs[ind].coastlines()

        if 0 < ind and include_metrics:
            sp_ret = vals
            valid = np.isfinite(sp_ret) * np.isfinite(sp_ref[0])
            corr = np.corrcoef(sp_ret[valid], sp_ref[0][valid])[0, 1]
            mse = ((sp_ret[valid] - sp_ref[0][valid]) ** 2).mean()
            mae = np.abs(sp_ret[valid] - sp_ref[0][valid]).mean()
            bias = 100.0 * (sp_ret[valid] - sp_ref[0][valid]).mean() / sp_ref[0][valid].mean()
            metrics = f"Bias: {bias:.2f} %\nCorr.: {corr:.2f}\nMSE: {mse:.2f}"
            texts.append(axs[ind].text(
                0.05, 0.15, metrics,
                transform=axs[ind].transAxes, ha='left', va='center', fontsize=12, color='deeppink'
            ))

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
            vals = res[variable][step].data
            if variable == "surface_precip":
                vals = np.maximum(vals, 1e-3)
            img_res[ind].set_data(vals)

            if 0 < ind and include_metrics:
                sp_ret = vals
                valid = np.isfinite(sp_ret) * np.isfinite(sp_ref[step])
                corr = np.corrcoef(sp_ret[valid], sp_ref[step][valid])[0, 1]
                mse = ((sp_ret[valid] - sp_ref[step][valid]) ** 2).mean()
                mae = np.abs(sp_ret[valid] - sp_ref[step][valid]).mean()
                bias = 100.0 * (sp_ret[valid] - sp_ref[step][valid]).mean() / sp_ref[ind][valid].mean()
                metrics = f"Bias: {bias:.2f} %\nCorr.: {corr:.2f}\nMSE: {mse:.2f}"
                texts[ind - 1].set_text(metrics)

        return texts + img_res

    ani = FuncAnimation(fig, update, frames=len(valid_times), interval=200, blit=True)
    plt.close()

    return ani
