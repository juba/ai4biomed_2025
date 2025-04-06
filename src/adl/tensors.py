"""
Utility functions for the tensors notebook.
"""

import numpy as np
import plotnine as pn
import plotnine.options as pn_options
import polars as pl
import torch


def plot_points1d(
    x: torch.Tensor, w: torch.Tensor | np.ndarray | int | float | None = None
) -> None:
    """
    Plot 1D input points and a parameter point on a horizontal line.

    Parameters
    ----------
    x : torch.Tensor
        A 1D tensor representing the inut points to be plotted along the x-axis.
    w : torch.Tensor
        A single value tensor representing the parameter point to be plotted.

    """
    df_x = pl.DataFrame({"x": x.numpy(), "y": 0})
    if isinstance(w, torch.Tensor):
        w = w.detach().numpy()
    pn_options.figure_size = (10, 0.8)
    g = (
        pn.ggplot(df_x)
        + pn.geom_hline(yintercept=0, linetype="dotted")
        + pn.geom_point(pn.aes(x="x"), y=0, color="white", fill="yellowgreen", size=4)
        + pn.theme_minimal()
        + pn.theme(
            axis_title_x=pn.element_blank(),
            axis_ticks_y=pn.element_blank(),
            axis_text_y=pn.element_blank(),
            panel_grid_major_y=pn.element_blank(),
            panel_grid_minor_y=pn.element_blank(),
        )
    )
    if w is not None:
        g = g + pn.geom_point(x=w, y=0, color="white", fill="orchid", size=4)
    g.show()
