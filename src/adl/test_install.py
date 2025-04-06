print("Checking installation...")

import numpy as np
import plotnine as pn
import polars as pl
import jupyterlab
import torch
import torchvision

import adl

if __name__ == "__main__":
    print(
        f"""
Torch version: {torch.__version__}
Torchvision version: {torchvision.__version__}
Polars version: {pl.__version__}
Numpy version: {np.__version__}
Plotnine version: {pn.__version__}
Jupyter lab version: {jupyterlab.__version__}
adl version: {adl.__version__}

-----------------------------------
Everything seems good !
-----------------------------------
        """
    )
