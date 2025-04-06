"""
Metrics notebook helpers.
"""

import polars as pl
import torch
from sklearn.model_selection import train_test_split


def stratified_split(d: pl.DataFrame, valid_proportion: float):
    d_pandas = d.to_pandas()
    train, valid = train_test_split(
        d_pandas, test_size=valid_proportion, stratify=d_pandas["Class"], random_state=42
    )
    X_train = torch.tensor(train.drop(columns="Class").values).float()
    X_valid = torch.tensor(valid.drop(columns="Class").values).float()
    y_train = torch.tensor(train[["Class"]].values).long().squeeze()
    y_valid = torch.tensor(valid[["Class"]].values).long().squeeze()
    return X_train, X_valid, y_train, y_valid
