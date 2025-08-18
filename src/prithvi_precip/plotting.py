"""
prithvi_precip.plotting
=======================
"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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

            m = ax.pcolormesh(x, y, tile, norm=norm, cmap=cmap)

    if colorbar:
        plt.colorbar(m)


def set_style():
    """
    Set the IPWGML matplotlib style.
    """
    plt.style.use(Path(__file__).parent / "files" / "prithvi_precip.mplstyle")
