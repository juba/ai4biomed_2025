"""
Utility functions for simple model with one parameter (regression without intercept).
"""

from typing import Literal

import plotnine as pn
import polars as pl
import torch
from numpy.random import default_rng


def train(
    x: torch.Tensor,
    y: torch.Tensor,
    step_size: float,
    epochs: int,
    w_init: float = 0.0,
    forward_type: Literal["simple", "sin"] = "simple",
    batch_size: int | None = None,
) -> pl.DataFrame:
    """
    Run training of a simple regression model without intercept.

    Args:
        x (torch.tensor): input tensor.
        y (torch.tensor): target tensor.
        step_size (float): step size (learning rate).
        epochs (int): number of epoch to run.
        w_init (float, optional): starting w value. Defaults to 0.0.

    Returns:
        pl.DataFrame: data frame of loss, gradient and weight values at each epoch.
    """
    w = torch.tensor(w_init, requires_grad=True)

    def forward(x):
        if forward_type == "simple":
            return w * x
        elif forward_type == "sin":
            return sin_fn(x, w=w)

    def loss_fn(y_pred, y):
        return torch.mean((y_pred - y) ** 2)

    if batch_size is not None:
        loader = torch.utils.data.DataLoader(SinDataset(x, y), batch_size=batch_size, shuffle=True)

    def train_step(epoch):
        if batch_size is not None:
            steps_values = {"epoch": epoch, "step": [], "w": [], "loss": [], "gradient": [], "new_w": []}
            loss = torch.tensor(0.0)
            for step, (x_batch, y_batch) in enumerate(loader):
                weight = w.item()
                yt_pred = forward(x_batch)
                batch_loss = loss_fn(yt_pred, y_batch)
                loss += batch_loss
                batch_loss.backward()
                gradient = w.grad.item()
                w.data = w.data - step_size * w.grad
                _ = w.grad.zero_()
                steps_values["step"].append(step)
                steps_values["w"].append(weight)
                steps_values["loss"].append(batch_loss.item())
                steps_values["gradient"].append(gradient)
                steps_values["new_w"].append(w.item())
            return pl.DataFrame(steps_values)
        else:
            weight = w.item()
            yt_pred = forward(x)
            loss = loss_fn(yt_pred, y)
            loss.backward()
            if w.grad is not None:
                gradient = w.grad.item()

                w.data = w.data - step_size * w.grad
                new_weight = w.item()
                _ = w.grad.zero_()
            return pl.DataFrame(
                {
                    "epoch": epoch,
                    "w": round(weight, 2),
                    "loss": round(loss.item(), 2),
                    "gradient": round(gradient, 2),
                    "new_w": round(new_weight, 2),
                }
            )

    res = [train_step(epoch + 1) for epoch in range(epochs)]
    return pl.concat(res)


def plot_train(
    x: torch.Tensor,
    y: torch.Tensor,
    step_size: float,
    epochs: int,
    wmin: float,
    wmax: float,
    w_init: float = 0.0,
    forward_type: Literal["simple", "sin"] = "simple",
    batch_size: int | None = None,
) -> pn.ggplot:
    """
    Plot graphical representation of the training process of a simple
    regression model without intercept.

    Args:
        x (torch.tensor): input tensor.
        y (torch.tensor): target tensor.
        step_size (float): step size (learning rate).
        epochs (int): number of epoch to run.
        wmin (float): minimum parameter value.
        wmax (float): maximum parameter value.
        w_init (float, optional): starting w value. Defaults to 0.0.

    Returns:
        pn.ggplot: plot of training process.
    """
    train_values = train(x, y, step_size, epochs, w_init, forward_type=forward_type, batch_size=batch_size)
    loss_values = compute_loss(x, y, wmin, wmax, forward_type=forward_type)

    title = f"step_size={step_size}" if batch_size is None else f"batch_size={batch_size}"

    return (
        pn.ggplot(loss_values, pn.aes(x="w", y="loss"))
        + pn.geom_line(linetype="dashed", color="grey")
        + pn.geom_point(pn.aes(y="loss", x="w", color="epoch"), data=train_values)
        + pn.geom_path(pn.aes(y="loss", x="w", color="epoch"), data=train_values)
        + pn.scale_color_cmap("viridis")
        + pn.theme_minimal()
        + pn.ggtitle(title)
        + pn.coord_cartesian(xlim=(wmin, wmax))
        + pn.theme(plot_background=pn.element_rect(fill="white"))
    )


def compute_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    wmin: float,
    wmax: float,
    nsteps: int = 300,
    forward_type: Literal["simple", "sin"] = "simple",
) -> pl.DataFrame:
    """
    Compute loss and gradient for different parameter values for a simple regression
    model without intercept.

    Args:
        x (torch.tensor): input tensor.
        y (torch.tensor): target tensor.
        wmin (float): minimum parameter value.
        wmax (float): maximum parameter value.
        nsteps (int, optional): number of values to compute. Defaults to 30.

    Returns:
        pl.DataFrame: data frame of loss and gradient values.
    """
    step = (wmax - wmin) / nsteps
    values = torch.arange(wmin, wmax + step, step)

    def compute_loss_value(value):
        w = value.clone().detach().requires_grad_(True)  # noqa: FBT003

        def forward(x):
            if forward_type == "simple":
                return w * x
            elif forward_type == "sin":
                return sin_fn(x, w=w)

        def loss_fn(y_pred, y):
            return torch.mean((y_pred - y) ** 2)

        y_pred = forward(x)
        loss = loss_fn(y_pred, y)
        loss.backward()

        return {"w": value.item(), "loss": loss.item(), "gradient": w.grad.item()}

    losses = pl.DataFrame([compute_loss_value(val) for val in values]).with_columns(
        (pl.col("w") + pl.col("gradient") / 300).alias("end")
    )
    return losses


def plot_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    wmin: float,
    wmax: float,
    nsteps: int = 30,
    gradient: bool = True,
    ylim: tuple = (None, None),
    forward_type: Literal["simple", "sin"] = "simple",
) -> pn.ggplot:
    """
    Plot the loss and gradient for different parameter values for a simple regression
    model without intercept.


    Args:
        x (torch.tensor): input tensor.
        y (torch.tensor): target tensor.
        wmin (float): minimum parameter value.
        wmax (float): maximum parameter value.
        nsteps (int, optional): number of values to compute. Defaults to 30.
        gradient (bool, optional): if True, plot gradient values at each step.
        ylim (tuple): y-axis limits. Defaults to automatic limits.

    Returns:
        pn.ggplot: plotnine plot.
    """
    loss_values = compute_loss(x, y, wmin, wmax, nsteps, forward_type=forward_type)
    g = (
        pn.ggplot(loss_values, pn.aes(x="w", y="loss"))
        + pn.geom_line()
        + pn.theme_minimal()
        + pn.theme(plot_background=pn.element_rect(fill="white"))
        + pn.coord_cartesian(ylim=ylim)
    )
    if gradient:
        g = (
            g
            + pn.geom_point(pn.aes(x="w", y="loss"))
            + pn.geom_segment(
                pn.aes(x="w", xend="end", y="loss", yend="loss"),
                color="#F55",
                arrow=pn.arrow(angle=20, length=0.07, type="closed"),
            )
        )
    return g


def plot_batch_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    n_batches: int,
    wmin: float = -2,
    wmax: float = 6,
    ylim: tuple | None = (0, 1000),
    nsteps: int = 300,
    forward_type: Literal["simple", "sin"] = "simple",
) -> pn.ggplot:
    """
    Plot the losses of different batches to be compared to the loss of the full dataset.

    Args:
        x (torch.tensor): input tensor.
        y (torch.tensor): target tensor.
        wmin (float): minimum parameter value.
        wmax (float): maximum parameter value.
        batch_size (int): batch size.
        n_batches (int): number of batches to plot.
        ylim (tuple): y-axis limits.

    Returns:
        pn.ggplot: plotnine plot.
    """
    g = (
        pn.ggplot(mapping=pn.aes(x="w", y="loss"))
        + pn.theme_minimal()
        + pn.theme(plot_background=pn.element_rect(fill="white"))
        + pn.coord_cartesian(ylim=ylim)
        + pn.labs(title=f"batch size: {batch_size}")
    )
    for _ in range(n_batches):
        batch_indices = torch.randperm(len(x))[:batch_size]
        x_batch = x[batch_indices]
        y_batch = y[batch_indices]
        batch_loss_values = compute_loss(
            x_batch, y_batch, wmin, wmax, nsteps=nsteps, forward_type=forward_type
        )
        g = g + pn.geom_line(data=batch_loss_values, color="#333", alpha=0.3)
    full_loss_values = compute_loss(x, y, wmin, wmax, nsteps=nsteps, forward_type=forward_type)
    g = g + pn.geom_line(data=full_loss_values, color="#F40", linetype="dashed", size=1.2)
    return g


# Complex loss example


def sin_fn(x: torch.Tensor, w: torch.Tensor | float):
    """
    Function to simulate a complex one parameter loss function.
    """
    if not (isinstance(w, torch.Tensor)):
        w = torch.tensor(w, dtype=torch.float)
    return (
        torch.sin(w) * torch.exp(1 - torch.cos(x)) ** 2
        + torch.cos(x) * torch.exp(1 - torch.sin(w)) ** 2
        + ((x - w) ** 2)
    )


class SinDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])


# Pre compute x and y values for complex loss
rng = default_rng(42)
X_SIN = torch.tensor(rng.uniform(-2, 2, 1000))
Y_SIN = sin_fn(X_SIN, w=-8.0)


def plot_sin_loss():
    """
    Complex loss plotting.
    """
    return plot_loss(X_SIN, Y_SIN, wmin=-13, wmax=13, forward_type="sin", gradient=False, nsteps=300)


def plot_sin_batch_loss(batch_size: int, n_batches: int):
    """
    Complex batch loss plotting.
    """
    return plot_batch_loss(
        X_SIN,
        Y_SIN,
        wmin=-13,
        wmax=13,
        ylim=(0, 10_000),
        forward_type="sin",
        batch_size=batch_size,
        n_batches=n_batches,
    )


def plot_sin_train(epochs: int, w_init: float, step_size: float, batch_size: int | None):
    """
    Complex loss training process plotting.
    """
    return plot_train(
        X_SIN,
        Y_SIN,
        wmin=-13,
        wmax=13,
        forward_type="sin",
        epochs=epochs,
        w_init=w_init,
        step_size=step_size,
        batch_size=batch_size,
    )
