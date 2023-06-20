from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from param import Parameter, Parameterized
import panel as pn
from panel.widgets import RadioButtonGroup as RBG
import holoviews as hv
import hvplot.xarray


class PlottingApp:
    
    def __init__(self):
        self.data: Optional[xr.Dataset] = None

    def make(self):
        raise NotImplementedError
    

class SimplePlot(PlottingApp):

    def __init__(self):
        super().__init__()

    def process(self, **cbs):
        ret = self.data
        for k, v in cbs.items():
            if v == 'Mean':
                ret = ret.mean(k)

        return plot_grid_as_2d(ds=ret)

    def make(self):
        assert self.data is not None
        dims = list(self.data.dims.keys())

        redux_cbs = {
            d: RBG(name=d, options=['None', 'Mean'], value='None') for d in dims
        }
        elts = []
        for k, v in redux_cbs.items():
            elts += [k, v]

        return pn.Column(
            pn.GridBox(*elts, ncols=2), 
            pn.bind(self.process, **redux_cbs)
            )


def plot_grid_as_2d(ds):
    dims = list(ds.dims.keys())
    return ds.hvplot.image()
