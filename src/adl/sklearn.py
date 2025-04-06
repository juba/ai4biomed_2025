"""
scikit-learn utility functions.
"""

import numpy as np
from scipy.sparse import spmatrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torch import Tensor


def skl_regression(
    x: list | Tensor | np.ndarray | spmatrix,
    y: list | Tensor | np.ndarray,
    *,
    fit_intercept: bool = True,
) -> dict:
    """
    Perform linear regression using scikit-learn and return fit metrics.

    Parameters
    ----------
    x : list or torch.Tensor or numpy.ndarray
        Input features (independent variable).
    y : list or torch.Tensor or numpy.ndarray
        Target values (dependent variable).
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.

    Returns
    -------
    tuple
        A tuple containing:
        - mse :
            Mean squared error of the predictions
        - slope :
            Slope coefficient of the linear regression
        - intercept :
            Intercept term of the linear regression
    """
    # Reshape input data
    x_sk = np.array(x).reshape(-1, 1)
    # Fit regression
    reg = LinearRegression(fit_intercept=fit_intercept).fit(x_sk, y)
    # Get parameters
    slope = float(reg.coef_[0])
    intercept = float(reg.intercept_) if fit_intercept else None
    # Compute MSE
    y_pred = reg.predict(x_sk)
    mse = float(mean_squared_error(y_pred, y))

    return {"mse": mse, "slope": slope, "intercept": intercept}
