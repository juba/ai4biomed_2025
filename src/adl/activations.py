"""
Helpers for activation functions notebook.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def train(model, x, y, epochs=1000):
    """
    Basic model training process.
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)  # type: ignore
    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
    return model


@torch.no_grad
def plot(x, y, model):
    """
    Target and predicted values plot.
    """
    plt.plot(x, y, ".", label="Target")
    plt.plot(x, model(x), "-", label="Predicted")
    plt.legend()
    plt.show()


def plot_activation_fns(activation_fns: dict):
    x_act = torch.tensor(np.linspace(-5, 5, 100)).float()
    fig, axs = plt.subplots(nrows=1, ncols=len(activation_fns), figsize=(12, 4))
    for ax, (label, fn) in zip(axs, activation_fns.items(), strict=True):
        ax.set_title(label)
        ax.set_xlim((-4, 4))
        ax.set_ylim((-4, 4))
        ax.axhline(0, color="#AAA", linestyle="--", linewidth=1)
        ax.axvline(0, color="#AAA", linestyle="--", linewidth=1)
        ax.grid(visible=True, color="#EEE")
        ax.plot(x_act, fn(x_act))
