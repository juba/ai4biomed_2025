"""
Utility functions for simple regression with intercept.
"""

from typing import Any

import plotnine as pn
import polars as pl
import torch


def train(
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    step_size: float | None = None,
    w_init: float = 0.0,
    b_init: float = 0.0,
    optimizer=None,
    optimizer_params: dict[str, Any] | None = None,
    scheduler=None,
    scheduler_params: dict | None = None,
) -> pl.DataFrame:
    """
    Run training of a simple regression model with intercept.

    Args:
        x (torch.Tensor): input data.
        y (torch.Tensor): target values.
        step_size (float): step size (learning rate).
        epochs (int): number of epoch to run.
        w_init (float, optional): starting w value. Defaults to 0.0.
        b_init (float, optional): starting b value. Defaults to 0.0.
        optimizer: torch optimizer function.
        optimizer_params: torch optimizer parameters.
        scheduler: torch scheduler function.
        scheduler_params: torch scheduler parameters.


    Returns:
        pl.DataFrame: data frame of loss, gradient and weight values at each epoch.
    """

    w = torch.tensor(w_init, requires_grad=True)
    b = torch.tensor(b_init, requires_grad=True)

    if optimizer is None:
        optimizer = torch.optim.SGD  # type:ignore
        optimizer_params = {"lr": step_size}
    if optimizer_params is None:
        optimizer_params = {}
    optimizer = optimizer(params=[w, b], **optimizer_params)

    if scheduler_params is None:
        scheduler_params = {}
    if scheduler is not None:
        scheduler = scheduler(optimizer, **scheduler_params)

    def forward(x) -> torch.Tensor:
        return w * x + b

    loss_fn = torch.nn.MSELoss()

    def train_step(epoch) -> dict:
        optimizer.zero_grad()  # type: ignore
        weight = w.item()
        bias = b.item()
        yt_pred = forward(x)
        loss = loss_fn(yt_pred, y)
        loss.backward()
        if w.grad is not None:
            weight_gradient = w.grad.item()
        if b.grad is not None:
            bias_gradient = b.grad.item()
        optimizer.step()  # type: ignore
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss)
            else:
                scheduler.step()

        new_weight = w.item()
        new_bias = b.item()

        return {
            "epoch": epoch,
            "weight": weight,
            "bias": bias,
            "loss": loss,
            "weight_gradient": weight_gradient,
            "new_weight": new_weight,
            "bias_gradient": bias_gradient,
            "new_bias": new_bias,
        }

    res = [train_step(epoch + 1) for epoch in range(epochs)]
    return pl.DataFrame(res)


def compute_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    wmin: float,
    wmax: float,
    bmin: float,
    bmax: float,
    nsteps: int = 10,
) -> pl.DataFrame:
    """
    Compute loss and gradient for different parameter values for a simple regression
    model with intercept.

    Args:
        x (torch.Tensor): input tensor.
        y (torch.Tensor): target tensor.
        wmin (float): minimum weight value.
        wmax (float): maximum weight value.
        bmin (float): minimum bias value.
        bmax (float): maximum bias value.
        nsteps (int, optional): number of values to compute. Defaults to 30.

    Returns:
        pl.DataFrame: data frame of loss and gradient values.
    """
    wstep = (wmax - wmin) / nsteps
    wvalues = torch.arange(wmin, wmax + wstep, wstep)
    bstep = (bmax - bmin) / nsteps
    bvalues = torch.arange(bmin, bmax + bstep, bstep)

    def compute_loss_value(x, y, wval, bval):
        w = wval.clone().detach().requires_grad_(True)  # noqa: FBT003
        b = bval.clone().detach().requires_grad_(True)  # noqa: FBT003

        def forward(x):
            return w * x + b

        def loss_fn(y_pred, y):
            return torch.mean((y_pred - y) ** 2)

        y_pred = forward(x)
        loss = loss_fn(y_pred, y)
        loss.backward()

        return {
            "w": wval.item(),
            "b": bval.item(),
            "loss": loss.item(),
            "wgrad": w.grad.item(),
            "bgrad": b.grad.item(),
        }

    return pl.DataFrame(
        [compute_loss_value(x, y, wval, bval) for wval in wvalues for bval in bvalues]
    )


def plot_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    true_weight: float,
    true_bias: float,
    grad_scale: float,
    w_factor: float = 0.8,
    b_factor: float = 0.8,
    nsteps: int = 8,
) -> pn.ggplot:
    """
    Plot the loss and gradient for different parameter values for a simple regression
    model with intercept.

    Args:
        x (torch.Tensor): input tensor.
        y (torch.Tensor): target tensor.
        true_weight (float): real weight value.
        true_bias (float): real bias value.
        grad_scale (float): scale by which divinding gradient length when plotting.
        w_factor (float): factor to apply to w axis.
        b_factor (float): factor to apply to b axis.
        nsteps (int, optional): number of values to compute. Defaults to 8.

    Returns:
        pn.ggplot: plotnine plot.
    """
    wmin, wmax = (
        true_weight - abs(true_weight) * w_factor,
        true_weight + abs(true_weight) * w_factor,
    )
    bmin, bmax = (
        true_bias - abs(true_bias) * b_factor,
        true_bias + abs(true_bias) * b_factor,
    )

    loss_values = compute_loss(x, y, wmin, wmax, bmin, bmax, nsteps)

    max_arrow_prop = 0.025
    loss_values = loss_values.with_columns(
        wend=pl.col("w") + pl.col("wgrad") / grad_scale,
        bend=pl.col("b") + pl.col("bgrad") / grad_scale,
    ).with_columns(
        arrow=(
            ((pl.col("wend") - pl.col("w")).abs() / pl.col("w").max()) > max_arrow_prop
        )
        | (((pl.col("bend") - pl.col("b")).abs() / pl.col("b").max()) > max_arrow_prop)
    )
    loss_arrows = loss_values.filter(pl.col("arrow"))
    loss_without_arrows = loss_values.filter(pl.col("arrow").not_())
    return (
        pn.ggplot(loss_values, pn.aes(x="w", y="b"))
        + pn.geom_point(pn.aes(x="w", y="b", size="loss"), color="#CCC")
        + pn.geom_point(
            data=pl.DataFrame({"w": true_weight, "b": true_bias}),
            fill="dodgerblue",
            color="white",
            size=5,
        )
        + pn.geom_segment(
            pn.aes(x="w", xend="wend", y="b", yend="bend"),
            data=loss_arrows,
            color="#F55",
            arrow=pn.arrow(angle=20, length=0.05, type="closed"),
        )
        + pn.geom_segment(
            pn.aes(x="w", xend="wend", y="b", yend="bend"),
            data=loss_without_arrows,
            color="#F55",
        )
        + pn.coord_cartesian(
            xlim=(wmin - (wmax - wmin) * 0.05, wmax + (wmax - wmin) * 0.05),
            ylim=(bmin - (bmax - bmin) * 0.05, bmax + (bmax - bmin) * 0.05),
        )
        + pn.scale_size_continuous(range=(0.2, 8))
        + pn.theme(panel_grid=pn.element_blank())
    )


def plot_train(
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    true_weight: float,
    true_bias: float,
    grad_scale: float,
    w_init: float = 0.0,
    b_init: float = 0.0,
    w_factor: float = 0.8,
    b_factor: float = 0.8,
    nsteps: int = 8,
    step_size: float | None = None,
    optimizer=None,
    optimizer_params: dict | None = None,
    scheduler=None,
    scheduler_params: dict | None = None,
) -> pn.ggplot:
    """
    Plot graphical representation of the training process of a simple
    regression model without intercept.

    Args:
        x (torch.Tensor): input tensor.
        y (torch.Tensor): target tensor.
        epochs (int): number of epoch to run.
        true_weight (float): real weight value.
        true_bias (float): real bias value.
        grad_scale (float): scale by which divinding gradient length when plotting.
        w_init (float, optional): starting w value. Defaults to 0.0.
        b_init (float, optional): starting b value. Defaults to 0.0.
        w_factor (float): factor to apply to w axis.
        b_factor (float): factor to apply to b axis.
        nsteps (int, optional): number of values to compute. Defaults to 8.
        step_size (float): step size (learning rate).
        optimizer: torch optimizer function.
        optimizer_params: torch optimizer parameters.
        scheduler: torch scheduler function.
        scheduler_params: torch scheduler parameters.


    Returns:
        pn.ggplot: plot of training process.
    """

    train_values = train(
        x=x,
        y=y,
        epochs=epochs,
        step_size=step_size,
        w_init=w_init,
        b_init=b_init,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        scheduler=scheduler,
        scheduler_params=scheduler_params,
    )
    loss_plot = plot_loss(
        x=x,
        y=y,
        true_weight=true_weight,
        true_bias=true_bias,
        grad_scale=grad_scale,
        nsteps=nsteps,
        w_factor=w_factor,
        b_factor=b_factor,
    )

    if step_size is not None:
        title = f"step_size={step_size}"
    elif optimizer is not None:
        title = f"{optimizer.__name__} - {optimizer_params}"

    return (
        loss_plot
        + pn.geom_point(pn.aes(x="weight", y="bias", color="epoch"), data=train_values)
        + pn.geom_path(pn.aes(x="weight", y="bias", color="epoch"), data=train_values)
        + pn.scale_color_cmap("viridis")
        + pn.ggtitle(title)
    )
