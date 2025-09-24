import logging

import hvplot.xarray  # assumes xarray is being used
import numpy as np
from .fit import fit_and_add_to_ds
logger = logging.getLogger(__name__)

# Note: This file provides hvplot-based versions of plotting functions from mpl.py.
# Functions assume xarray DataArrays or Datasets as input where appropriate.



def fit_and_plot_1d(ds, name, fit_class, dim_order=None, run_kwargs={}):

    ds2, result = fit_and_add_to_ds(
        ds=ds,
        dim_name=name,
        fit_class=fit_class,
        dim_order=dim_order,
        **run_kwargs,
    )
    return ds2, result, plot_fit_1d_hv(ds, name)


def plot_fit_1d_hv(ds, name):
    # print("igot here")
    datada = ds[name]
    fitda = ds[name + "_fit"]
    if len(datada.dims) > 1:
        raise RuntimeError("This function only supports data with one independent.")
    dim_name = datada.dims[0]
    # Data and fit overlay
    data_plot = datada.hvplot.scatter(x=dim_name, y=name, label="data", color="blue")
    fit_plot = fitda.hvplot.line(x=dim_name, y=name + "_fit", label="fit", color="red")
    overlay = data_plot * fit_plot
    # Residuals
    # residuals = (datada - fitda)
    # print(residuals)
    # res_plot = residuals.hvplot.scatter(x=dim_name, y="Residuals", label="residuals", color="green")
    # layout = (overlay + res_plot).cols(1)
    # return layout
    return overlay

# Example conversion for a histogram plot (readout_hist):
def readout_hist_hv(signal, nbins=41, log=True):
    import pandas as pd
    I = signal.real
    Q = signal.imag
    lim = np.max((I**2. + Q**2.)**.5)
    df = pd.DataFrame({'I': I, 'Q': Q})
    hist = df.hvplot.hexbin(x='I', y='Q', gridsize=nbins, xlim=(-lim, lim), ylim=(-lim, lim), logz=log, cmap='viridis')
    return hist

# Note: For other functions like pplot, ppcolormesh, waterfall, plot_wigner, etc.,
# similar conversions can be made using hvplot.line, hvplot.scatter, hvplot.image, etc.
# If you need a specific function converted, let me know!
