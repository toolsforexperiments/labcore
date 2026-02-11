from typing import Tuple, Any, Optional, Union, Dict, List, Type
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from matplotlib import pyplot as plt
import lmfit
import xarray as xr


class Parameter:
    def __init__(self, name: str, value: Any = None, **kw: Any):
        self.name = name
        self.value = value
        self._attrs = {}
        for k, v in kw.items():
            self._attrs[k] = v

    def __getattr__(self, key: str) -> Any:
        return self._attrs[key]


class Parameters(OrderedDict):
    """A collection of parameters"""

    def add(self, name: str, **kw: Any) -> None:
        """Add/overwrite a parameter in the collection."""
        self[name] = Parameter(name, **kw)


class AnalysisResult(object):
    def __init__(self, parameters: Dict[str, Union[Dict[str, Any], Any]]):
        self.params = Parameters()
        for k, v in parameters.items():
            if isinstance(v, dict):
                self.params.add(k, **v)
            else:
                self.params.add(k, value=v)

    def eval(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Analysis types that produce data (like filters or fits) should implement this."""
        raise NotImplementedError

    def params_to_dict(self) -> Dict[str, Any]:
        """Get all analysis parameters.
        Returns a dictionary that contains one key per parameter (its name).
        Each value contains all attributes of the parameter object, except
        those whose names start with `_` and those that are callable.
        """
        ret: Dict[str, Any] = {}
        for name, param in self.params.items():
            ret[name] = {}
            for n in dir(param):
                attr = getattr(param, n)
                if n[0] != "_" and not callable(attr):
                    ret[name][n] = attr
        return ret


class Analysis(object):
    """Basic analysis object.

    Parameters
    ----------
    coordinates
        may be a single 1d numpy array (for a single coordinate) or a tuple
        of 1d arrays (for multiple coordinates).
    data
        a 1d array of data
    """

    def __init__(
        self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray], data: np.ndarray
    ):
        """Constructor of `Analysis`."""
        self.coordinates = coordinates
        self.data = data

    def analyze(
        self,
        coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
        data: np.ndarray,
        *args: Any,
        **kwargs: Any
    ) -> AnalysisResult:
        """Needs to be implemented by each inheriting class."""
        raise NotImplementedError

    def run(self, *args: Any, **kwargs: Any) -> AnalysisResult:
        return self.analyze(self.coordinates, self.data, **kwargs)


class FitResult(AnalysisResult):
    def __init__(self, lmfit_result: lmfit.model.ModelResult):
        self.lmfit_result = lmfit_result
        self.params = lmfit_result.params

    def eval(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self.lmfit_result.eval(*args, **kwargs)


class Fit(Analysis):
    @staticmethod
    def model(*arg: Any, **kwarg: Any) -> np.ndarray:
        raise NotImplementedError

    def analyze(
        self,
        coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
        data: np.ndarray,
        dry: bool = False,
        params: Dict[str, Any] = {},
        *args: Any,
        **fit_kwargs: Any
    ) -> FitResult:
        model = lmfit.model.Model(self.model)

        _params = lmfit.Parameters()
        for pn, pv in self.guess(coordinates, data).items():
            _params.add(pn, value=pv)
        for pn, pv in params.items():
            if isinstance(pv, lmfit.Parameter):
                _params[pn] = pv
            else:
                _params[pn].set(value=pv)

        if dry:
            for pn, pv in _params.items():
                pv.set(vary=False)
        lmfit_result = model.fit(
            data, params=_params, coordinates=coordinates, **fit_kwargs
        )

        return FitResult(lmfit_result)

    @staticmethod
    def guess(
        coordinates: Union[Tuple[np.ndarray, ...], np.ndarray], data: np.ndarray
    ) -> Dict[str, Any]:
        raise NotImplementedError


# -- Tools for fitting xarray/pandas data -- #


def xr2fitinput(arr: xr.DataArray) -> Tuple[List[np.ndarray], np.ndarray]:
    shp = arr.shape
    coords1d = (arr[k].values for k in arr.dims)
    coords = [a.flatten() for a in np.meshgrid(*coords1d, indexing="ij")]
    if len(coords) == 1:
        coords = coords[0]
    vals = arr.values.flatten()
    return coords, vals


def fit_and_add_to_ds(
    ds: xr.Dataset,
    dim_name: str,
    fit_class: Type[Fit],
    dim_order: Optional[List[int]] = None,
    **run_kwargs: Any
) -> Tuple[xr.Dataset, FitResult]:
    arr = ds[dim_name]
    coords_, vals = xr2fitinput(arr)
    if dim_order is not None:
        coords = [coords_[i] for i in dim_order]
    else:
        coords = coords_

    fit = fit_class(coords, vals)
    result = fit.run(**run_kwargs)
    fit_data = result.eval()

    fit_darr = xr.DataArray(
        name=arr.name + "_fit",
        data=fit_data.reshape(arr.shape),
        dims=arr.dims,
        coords=arr.coords,
    )
    ds[fit_darr.name] = fit_darr
    return ds, result


def plot_ds_2d_with_fit(
    ds: xr.Dataset,
    dim_name: str,
    x: Optional[str] = None,
    y: Optional[str] = None,
    plot_kwargs: Dict[str, Any] = {},
):
    data = ds[dim_name]
    fit = ds[dim_name + '_fit']

    if x is None:
        x = data.dims[0]
    if y is None:
        y = data.dims[1]

    dataopts = dict(
        clim=(data.min(), data.max()),
        cmap='magma',
    )
    dataopts.update(**plot_kwargs)
    
    title = 'data'
    folder = ds.attrs.get('raw_data_folder', None)
    if folder is not None:
        title += f": {Path(folder).stem}"

    plot = data.hvplot.quadmesh(
        x=x,
        y=y,
        title=title,
        **dataopts
    ) \
    + fit.hvplot.quadmesh(
        x=x,
        y=y,
        title='fit',
        **dataopts
    ) \
    + (data - fit).hvplot.quadmesh(
        x=x,
        y=y,
        title='residuals',
        cmap='bwr',
        **plot_kwargs,
    )
    
    return plot
