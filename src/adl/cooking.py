"""
Utility functions for the overfitting and batches notebooks.
"""

import numpy as np
import plotnine as pn
import polars as pl
import torch

X_AXIS_LABEL = "Cooking time (min)"
Y_AXIS_LABEL = "Score out of 10"
MAX_SCORE = 10
MIN_SCORE = 0


def generate_data(size: int, noise_scale=0.2) -> tuple:
    """
    Generate sample data for overfitting example.

    Args:
        size (int): length of generated data.

    Returns:
        tuple: tuple of time, score data.
    """
    time = np.linspace(start=5, stop=120, num=size)
    score = (-(0.095 * time**2) + 12 * time + 11) / 50 + np.random.normal(
        loc=0, scale=noise_scale, size=time.shape
    )
    score[score > MAX_SCORE] = MAX_SCORE
    score[score < MIN_SCORE] = MIN_SCORE

    time = torch.tensor(time).float().view(-1, 1)
    score = torch.tensor(score).float().view(-1, 1)
    return time, score


def bin_plot(time: torch.Tensor, score: torch.Tensor) -> pn.ggplot:
    """
    Generate a binned plot of generated data

    Args:
        time (np.array): time values array.
        score (np.array): score values array.

    Returns:
        pn.ggplot: scatter plot.
    """
    x = time.detach().squeeze().numpy()
    y = score.detach().squeeze().numpy()

    return (
        pn.ggplot(mapping=pn.aes(x=x, y=y))
        + pn.geom_bin_2d(bins=100)
        + pn.labs(x=X_AXIS_LABEL, y=Y_AXIS_LABEL)
        + pn.theme_minimal()
        + pn.theme(plot_background=pn.element_rect(fill="white"))
    )


def scatter_plot(time: torch.Tensor, score: torch.Tensor, size: float = 2.5, alpha: float = 0.5) -> pn.ggplot:
    """
    Generate a scatter plot of generated data

    Args:
        time (np.array): time values array.
        score (np.array): score values array.
        size (float): point size.
        alpha (float): point opacity.

    Returns:
        pn.ggplot: scatter plot.
    """
    x = time.detach().squeeze().numpy()
    y = score.detach().squeeze().numpy()

    return (
        pn.ggplot(mapping=pn.aes(x=x, y=y))
        + pn.geom_point(color="#e41a1c", size=size, alpha=alpha)
        + pn.labs(x=X_AXIS_LABEL, y=Y_AXIS_LABEL)
        + pn.theme_minimal()
        + pn.theme(plot_background=pn.element_rect(fill="white"))
        + pn.scale_y_continuous(limits=(MIN_SCORE, MAX_SCORE))
    )


def scatter_plot_pred(time: torch.Tensor, score: torch.Tensor, score_pred: torch.Tensor) -> pn.ggplot:
    """
    Generate a scatter plot of data and predictions.

    Args:
        time (np.array): time values array.
        score (np.array): score values array.
        score_pred (np.array): predicted score values array.

    Returns:
        pn.ggplot: scatter plot.
    """
    x = time.detach().squeeze().numpy()
    y = score.detach().squeeze().numpy()
    y_pred = score_pred.detach().squeeze().numpy()

    d = pl.DataFrame({"time": x, "score": y, "score_pred": y_pred}).rename(
        {"score": "true_score", "score_pred": "predicted_score"}
    )
    d_long = d.unpivot(index="time", on=["true_score", "predicted_score"])
    return (
        pn.ggplot(d_long, pn.aes(x="time"))
        + pn.geom_segment(pn.aes(xend="time", y="true_score", yend="predicted_score"), color="#333", data=d)
        + pn.geom_point(pn.aes(y="value", color="variable"), size=2.5, alpha=0.5)
        + pn.labs(x=X_AXIS_LABEL, y=Y_AXIS_LABEL, color="")
        + pn.scale_color_brewer(type="qualitative", palette="Set1", limits=("true_score", "predicted_score"))
        + pn.theme_minimal()
        + pn.theme(plot_background=pn.element_rect(fill="white"))
        + pn.scale_y_continuous(limits=(MIN_SCORE, MAX_SCORE))
    )


def line_plot(model: torch.nn.Module, step: float = 0.5) -> pn.ggplot:
    """
    Generate a line plot of predicted values.

    Args:
        model: trained torch model.
        step (float): step size. Defaults to 1.

    Returns:
        pn.ggplot: line plot.
    """

    time_cont = np.linspace(5, 120, num=round(115 / step))
    x = torch.tensor(time_cont).float().view(-1, 1)
    score_pred_cont = model(x).squeeze().detach().numpy()
    return (
        pn.ggplot(mapping=pn.aes(x=time_cont, y=score_pred_cont))
        + pn.geom_line(color="#377eb8")
        + pn.labs(x=X_AXIS_LABEL, y=Y_AXIS_LABEL)
        + pn.theme_minimal()
        + pn.theme(plot_background=pn.element_rect(fill="white"))
        + pn.scale_y_continuous(limits=(MIN_SCORE, MAX_SCORE))
    )
