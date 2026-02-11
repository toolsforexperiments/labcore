from typing import Union, Optional

import numpy as np
import pandas as pd
import xarray as xr


Data = Union[xr.Dataset, pd.DataFrame]
"""Type alias for valid data. Can be either a pandas DataFrame or an xarray Dataset."""


def data_dims(data: Optional[Data]) -> tuple[list[str], list[str]]:
    if data is None:
        return [], []

    if isinstance(data, pd.DataFrame):
        return list(data.index.names), data.columns.to_list()
    elif isinstance(data, xr.Dataset):
        return [str(c) for c in list(data.coords)], list(data.data_vars)
    else:
        raise NotImplementedError


def split_complex(data: Data) -> Data:
    """Split complex dependents into real and imaginary parts.

    TODO: should update units as well

    Parameters
    ----------
    data
        input data.

    Returns
    -------
    data with complex dependents split into real and imaginary parts.

    Raises
    ------
    NotImplementedError
        if data is not a pandas DataFrame or an xarray Dataset.
    """
    indep, dep = data_dims(data)

    if not isinstance(data, pd.DataFrame) and not isinstance(data, xr.Dataset):
        raise NotImplementedError

    dropped = []
    for d in dep:
        if np.iscomplexobj(data[d]):
            data[f"{d}_Re"] = np.real(data[d])
            data[f"{d}_Im"] = np.imag(data[d])
            if isinstance(data, xr.Dataset):
                data[f"{d}_Re"].attrs = data[d].attrs
                data[f"{d}_Im"].attrs = data[d].attrs
            dropped.append(d)
    if isinstance(data, pd.DataFrame):
        return data.drop(columns=dropped)
    else:
        return data.drop_vars(dropped)
