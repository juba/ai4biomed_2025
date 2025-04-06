"""
Utility functions for optimizers notebook.
"""

from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
import torch


def sin_loss_fn(w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    return (
        torch.sin(w2) * torch.exp(1 - torch.cos(w1)) ** 2
        + torch.cos(w1) * torch.exp(1 - torch.sin(w2)) ** 2
        + (w1 - w2) ** 2
    )


def reg_loss_fn(w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    return w1**2 + w2**2 + 1.8 * w1 * w2


def train(
    epochs: int,
    loss_fn: Callable,
    step_size: float | None = None,
    w1_init: float = 0.0,
    w2_init: float = 0.0,
    optimizer=None,
    optimizer_params: dict[str, Any] | None = None,
    scheduler=None,
    scheduler_params: dict | None = None,
) -> pl.DataFrame:
    w1 = torch.tensor(w1_init, requires_grad=True)
    w2 = torch.tensor(w2_init, requires_grad=True)

    if optimizer is None:
        optimizer = torch.optim.SGD  # type:ignore
        optimizer_params = {"lr": step_size}
    if optimizer_params is None:
        optimizer_params = {}
    optimizer = optimizer(params=[w1, w2], **optimizer_params)

    if scheduler_params is None:
        scheduler_params = {}
    if scheduler is not None:
        scheduler = scheduler(optimizer, **scheduler_params)

    def train_step(epoch) -> dict:
        optimizer.zero_grad()  # type: ignore
        w1_value = w1.item()
        w2_value = w2.item()
        loss = loss_fn(w1, w2)
        loss.backward()
        optimizer.step()  # type: ignore
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss)
            else:
                scheduler.step()
        return {
            "epoch": epoch,
            "w1": w1_value,
            "w2": w2_value,
            "loss": loss.item(),
        }

    res = [train_step(epoch + 1) for epoch in range(epochs)]

    return pl.DataFrame(res)


def plot_loss(
    w1min: float,
    w1max: float,
    w2min: float,
    w2max: float,
    loss_fn: Callable,
    nsteps: int = 8,
    ax: plt.Axes = None,  # type:ignore
) -> None:
    x = torch.linspace(w1min, w1max, nsteps)
    y = torch.linspace(w2min, w2max, nsteps)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    Z = loss_fn(X, Y)

    show_plot = False
    if ax is None:
        show_plot = True
        fig, ax = plt.subplots()
    ax.set_xlim((w1min, w1max))
    ax.set_ylim((w2min, w2max))  # type: ignore
    cs = ax.contour(X, Y, Z, levels=30, cmap="turbo", linewidths=1)
    ax.clabel(cs, fontsize=6, levels=cs.levels[::2])  # type: ignore
    # im = ax.imshow(
    #     Z,
    #     interpolation="bicubic",
    #     cmap="gist_earth",
    #     aspect="auto",
    #     origin="lower",
    # )

    if show_plot:
        plt.show()


def plot_train(
    w1min: float,
    w1max: float,
    w2min: float,
    w2max: float,
    epochs: int,
    loss_fn: Callable,
    w1_init: float = 0.0,
    w2_init: float = 0.0,
    optimum: tuple[float, float] | None = None,
    nsteps: int = 8,
    step_size: float | None = None,
    optimizer=None,
    optimizer_params: dict | None = None,
    scheduler=None,
    scheduler_params: dict | None = None,
    ax: plt.Axes = None,  # type: ignore
) -> None:
    train_values = train(
        epochs=epochs,
        step_size=step_size,
        loss_fn=loss_fn,
        w1_init=w1_init,
        w2_init=w2_init,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        scheduler=scheduler,
        scheduler_params=scheduler_params,
    )

    show_plot = False
    if ax is None:
        show_plot = True
        fig, ax = plt.subplots()

    plot_loss(
        loss_fn=loss_fn,
        w1min=w1min,
        w1max=w1max,
        w2min=w2min,
        w2max=w2max,
        nsteps=nsteps,
        ax=ax,
    )  # type: ignore

    title = ""
    if step_size is not None:
        title += f"step_size={step_size}"
    if optimizer is not None:
        title += f"{optimizer.__name__} - {optimizer_params}"
    if scheduler is not None:
        title += f" - {scheduler.__name__}"

    x = train_values.get_column("w1")
    y = train_values.get_column("w2")
    epoch = train_values.get_column("epoch")

    if ax is None:
        return

    ax.set_title(title)
    ax.scatter(
        x,
        y,
        c=epoch,
        cmap="viridis",
        zorder=5.0,
    )
    ax.plot(x, y, "-", c="#999", linewidth=1, zorder=4.5)
    if optimum is not None:
        ax.scatter(
            *optimum,
            s=100,
            edgecolor="white",
            color="dodgerblue",
            linewidth=1,
            zorder=4.0,
        )

    if show_plot:
        plt.show()
