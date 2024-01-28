from typing import Tuple, List, Optional, Union, Any, Dict
import logging

import numpy as np
from numpy import ndarray
from numpy import complexfloating

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import gridspec, cm, colors, ticker
from matplotlib.colors import rgb2hex
import seaborn as sns

from .fit import FitResult, Fit, fit_and_add_to_ds


logger = logging.getLogger(__name__)


def fit_and_plot_1d(
    ds,
    name,
    fit_class,
    dim_order: Optional[List[int]] = None,
    run_kwargs: Dict[str, Any] = {},
):
    ds2, result = fit_and_add_to_ds(
        ds=ds,
        dim_name=name,
        fit_class=fit_class,
        dim_order=dim_order,
        **run_kwargs,
    )
    return ds2, result, plot_fit_1d(ds, name)


def plot_fit_1d(ds, name):
    datada = ds[name]
    fitda = ds[name + "_fit"]

    if len(datada.dims) > 1:
        raise RuntimeError("This function only supports data with one independent.")
    dimda = datada.coords[datada.dims[0]]

    fig, ax = plt.subplot_mosaic(
        [["data"], ["res"]],
        sharex=True,
    )
    ax["data"].plot(dimda, datada, ".")
    ax["data"].plot(dimda, fitda, "-")
    ax["res"].plot(dimda, datada - fitda, ".")

    format_ax(
        ax["data"],
        ylabel=f"{datada.name} ({datada.attrs.get('units', '')})",
    )
    format_ax(
        ax["res"],
        xlabel=f"{dimda.name} ({dimda.attrs.get('units', '')})",
        ylabel=f"residuals",
    )
    return fig


# def readout_hist(signal: ndarray, fig_ax: Optional[Tuple[Figure, Axes]] = None,
#                  nbins: int = 41, log: bool = True) -> Tuple[Figure, Axes]:
#     """
#     Plots an IQ histogram.

#     Parameters
#     ----------
#     signal:
#         array-like, data in complex form.

#     Returns
#     -------
#     Figure
#         The matplotlib Figure with the plot.
#     """
#     I = signal.real
#     Q = signal.imag
#     lim = np.max((I**2. + Q**2.)**.5)

#     if fig_ax is None:
#         fig, ax = plt.subplots(1, 1)
#     else:
#         fig, ax = fig_ax

#     ax.set_xlabel('I')
#     ax.set_ylabel('Q')
#     h, xe, ye = np.histogram2d(
#         I, Q, bins=nbins,
#         range=[[-lim, lim], [-lim, lim]],
#     )
#     h[h==0] = np.nan
#     im = ax.pcolormesh(
#         xe, ye, h,
#         cmap='viridis',
#         norm='log' if log else 'linear',
#     )

#     ax.set_aspect('equal')
#     format_ax(ax, xlabel='I', ylabel='Q')

#     cb = fig.colorbar(im, ax=ax, shrink=0.5)
#     format_right_cb(cb)
#     cb.ax.set_xlabel('cts', ha='left')

#     return fig, ax


# # tools for prettier plotting
# def pplot(ax, x, y, yerr=None, linex=None, liney=None, color=None, fmt='o',
#           alpha=0.5, mew=0.5, **kw):

#     zorder = kw.pop('zorder', 2)
#     line_dashes = kw.pop('line_dashes', [])
#     line_lw = kw.pop('line_lw', 2)
#     line_alpha = kw.pop('line_alpha', 0.5)
#     line_color = kw.pop('line_color', color)
#     line_zorder = kw.pop('line_zorder', 1)
#     line_from_ypts = kw.pop('line_from_ypts', False)
#     elinewidth = kw.pop('elinewidth', 0.5)
#     label = kw.pop('label', None)
#     label_x = kw.pop('label_x', x[-1])
#     label_y_ofs = kw.pop('label_y_ofs', 0)
#     label_kw = kw.pop('label_kw', {})
#     fill_color = kw.pop('fill_color', None)

#     syms = []

#     if linex is None:
#         linex = x

#     if type(liney) == str:
#         if liney == 'data':
#             liney = y

#     edge_plot_kws = dict(mfc='None', mew=mew, zorder=zorder)
#     edge_plot_kws.update(kw)
#     if colors is not None:
#         edge_plot_kws['mec'] = color
#     edge, = ax.plot(x, y, fmt, **edge_plot_kws)
#     color = edge.get_color()

#     # TODO: the z-ordering with this method isn't too great (error bars may
#     #   be hidden behind the line...)
#     if yerr is not None:
#         err = ax.errorbar(x, y, yerr=yerr, fmt='none', ecolor=color, capsize=0,
#                           elinewidth=elinewidth, zorder=zorder-1)
#         empty_symbol_kws = edge_plot_kws.copy()
#         empty_symbol_kws.update({'mfc': 'w', 'mew': 0, 'zorder': zorder-1}, )
#         _ = ax.plot(x, y, fmt, **empty_symbol_kws)
#         # syms.append(err)

#     if liney is None and line_from_ypts:
#         liney = y.copy()

#     if liney is not None:
#         if line_color is None:
#             line_color = color
#         line, = ax.plot(linex, liney, dashes=line_dashes, lw=line_lw,
#                         color=line_color, zorder=line_zorder, alpha=line_alpha)
#         syms.append(line)

#     if fill_color is None:
#         fill_color = color

#     fill, = ax.plot(x, y, fmt, mec='none', mfc=fill_color, alpha=alpha,
#                     zorder=zorder-1, **kw)

#     syms.append(fill)
#     syms.append(edge)

#     if label is not None:
#         label_idx = np.argmin(np.abs(x - label_x))
#         ax.annotate(label, (label_x, y[label_idx] + label_y_ofs),
#                     color=color, **label_kw)

#     return tuple(syms)


# def ppcolormesh(ax, x, y, z, make_grid=True, **kw):
#     if make_grid:
#         _x, _y = pcolorgrid(x, y)
#     else:
#         _x, _y = x, y

#     im = ax.pcolormesh(_x, _y, z, **kw)
#     ax.set_xlim(_x.min(), _x.max())
#     ax.set_ylim(_y.min(), _y.max())

#     return im


# def waterfall(ax, xs, ys, offset=None, style='pplot', **kw):
#     cmap = kw.pop('cmap', mpl.rcParams['image.cmap'])
#     linex = kw.pop('linex', xs)
#     liney = kw.pop('liney', None)
#     draw_baselines = kw.pop('draw_baselines', False)
#     baseline_kwargs = kw.pop('baseline_kwargs', {})

#     ntraces = ys.shape[0]
#     if offset is None:
#         offset = ys.max() - ys.min()

#     if 'color' not in kw:
#         colorseq = get_color_cycle(ntraces, colormap=cmap)
#     else:
#         c = kw.pop('color', None)
#         colorseq = [c for n in range(ntraces)]

#     for iy, yvals in enumerate(ys):
#         x = xs if len(xs.shape) == 1 else xs[iy]
#         y = yvals + iy * offset
#         lx = linex if len(linex.shape) == 1 else linex[iy]
#         ly = None if liney is None else liney[iy] + iy * offset
#         color = colorseq[iy]

#         if draw_baselines:
#             baseline_opts = dict(color=color, lw=1, dashes=[1, 1])
#             for k, v in baseline_kwargs:
#                 baseline_opts[k] = v
#             ax.axhline(iy * offset, **baseline_opts)

#         if style == 'pplot':
#             pplot(ax, x, y, linex=lx, liney=ly, color=color, **kw)
#         elif style == 'lines':
#             ax.plot(x, y, '-', color=color, **kw)


# def plot_wigner(ax, xs, ys, zs, norm=None, clim=None, **kw):
#     cmap = kw.pop('cmap', cm.bwr)
#     xticks = kw.pop('xticks', None)
#     yticks = kw.pop('yticks', None)
#     xticklabels = kw.pop('xticklabels', None)
#     yticklabels = kw.pop('yticklabels', None)

#     if norm is None and clim is None:
#         clim = max(abs(zs.min()), zs.max())
#     elif norm is None:
#         norm = colors.Normalize(vmin=-abs(clim), vmax=abs(clim))

#     im = ppcolormesh(ax, xs, ys, zs, norm=norm, cmap=cmap)

#     if xticks is None:
#         xtick = max(xs) // 1
#         xticks = [-xtick, 0, xtick]
#     if yticks is None:
#         ytick = max(ys) // 1
#         yticks = [-ytick, 0, ytick]

#     ax.set_xticks(xticks)
#     ax.set_yticks(yticks)

#     if xticklabels is not None:
#         ax.set_xticklabels(xticklabels)
#     if yticklabels is not None:
#         ax.set_yticklabels(yticklabels)

#     return im


# some common tools
# =================


# color management tools
def get_color_cycle(n, colormap, start=0.0, stop=1.0, format="hex"):
    if type(colormap) == str:
        colormap = getattr(cm, colormap)

    pts = np.linspace(start, stop, n)
    if format == "hex":
        colors = [rgb2hex(colormap(pt)) for pt in pts]
    return colors


# tools for color plots
def centers2edges(arr):
    e = (arr[1:] + arr[:-1]) / 2.0
    e = np.concatenate(([arr[0] - (e[0] - arr[0])], e))
    e = np.concatenate((e, [arr[-1] + (arr[-1] - e[-1])]))
    return e


def pcolorgrid(xaxis, yaxis):
    xedges = centers2edges(xaxis)
    yedges = centers2edges(yaxis)
    xx, yy = np.meshgrid(xedges, yedges)
    return xx, yy


# creating and formatting figures
def correctly_sized_figure(
    widths, heights, margins=0.5, dw=0.2, dh=0.2, make_axes=True
):
    """
    Create a figure and grid where all dimensions are specified in inches.
    Arguments:
        widths: list of column widths
        heights: list of row heights
        margins: either a scalar or a list of four numbers (l, r, t, b)
        dw: white space between subplots, horizontal
        dh: white space between subplots, vertical
        make_axes: bool; if True, create axes on the grid and return,
                   else return the gridspec.
    """
    wsum = sum(widths)
    hsum = sum(heights)
    nrows = len(heights)
    ncols = len(widths)
    if type(margins) == list:
        l, r, t, b = margins
    else:
        l = r = t = b = margins

    figw = wsum + (ncols - 1) * dw + l + r
    figh = hsum + (nrows - 1) * dh + t + b

    # margins in fraction of the figure
    top = 1.0 - t / figh
    bottom = b / figh
    left = l / figw
    right = 1.0 - r / figw

    # subplot spacing in fraction of the subplot size
    wspace = dw / np.average(widths)
    hspace = dh / np.average(heights)

    fig = plt.figure(figsize=(figw, figh))
    gs = gridspec.GridSpec(nrows, ncols, height_ratios=heights, width_ratios=widths)
    gs.update(
        top=top, bottom=bottom, left=left, right=right, wspace=wspace, hspace=hspace
    )

    if make_axes:
        axes = []
        for i in range(nrows):
            for j in range(ncols):
                axes.append(fig.add_subplot(gs[i, j]))

        return fig, axes

    else:
        return fig, gs


def format_ax(
    ax,
    top=False,
    right=False,
    xlog=False,
    ylog=False,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    xticks=3,
    yticks=3,
):
    ax.tick_params(
        axis="x",
        which="both",
        pad=2,
        top=top,
        labeltop=top,
        bottom=not top,
        labelbottom=not top,
    )
    if top:
        ax.xaxis.set_label_position("top")

    ax.tick_params(
        axis="y",
        which="both",
        pad=2,
        right=right,
        labelright=right,
        left=not right,
        labelleft=not right,
    )
    if right:
        ax.yaxis.set_label_position("right")

    if isinstance(xticks, list):
        ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
        if xlim is not None:
            ax.set_xlim(xlim)
    elif xlim is not None:
        ax.xaxis.set_major_locator(ticker.LinearLocator(xticks))
        ax.set_xlim(xlim)
    else:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(xticks))

    if isinstance(yticks, list):
        ax.yaxis.set_major_locator(ticker.FixedLocator(yticks))
        if ylim is not None:
            ax.set_ylim(ylim)
    elif ylim is not None:
        ax.yaxis.set_major_locator(ticker.LinearLocator(yticks))
        ax.set_ylim(ylim)
    else:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(yticks))

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2


def format_right_cb(cb):
    cb.outline.set_visible(False)
    cb.ax.xaxis.set_visible(True)
    cb.ax.xaxis.set_label_position("top")


def add_legend(
    ax, anchor_point=(1, 1), legend_ref_point="lower right", **labels_and_handles
):
    if len(labels_and_handles) > 0:
        handles = []
        labels = []
        for l, h in labels_and_handles.items():
            handles.append(h)
            labels.append(l)
        ax.legend(
            handles,
            labels,
            bbox_to_anchor=anchor_point,
            borderpad=0,
            loc=legend_ref_point,
        )
    else:
        ax.legend(bbox_to_anchor=anchor_point, borderpad=0, loc=legend_ref_point)


def setup_plotting(sns_style="whitegrid", rcparams={}):
    # some sensible defaults for sizing, those are for a typical print-plot
    sns.set_style(sns_style)

    mpl.rcParams["figure.constrained_layout.use"] = True
    mpl.rcParams["figure.dpi"] = 200
    mpl.rcParams["figure.figsize"] = (3, 2)
    mpl.rcParams["font.size"] = 6
    # mpl.rcParams['lines.marker'] = 'o'
    mpl.rcParams["lines.markersize"] = 3
    mpl.rcParams["lines.linewidth"] = 1
    mpl.rcParams["axes.linewidth"] = 0.5
    mpl.rcParams["axes.titlesize"] = "medium"
    mpl.rcParams["grid.linewidth"] = 0.5
    mpl.rcParams["image.cmap"] = "viridis"
    mpl.rcParams["legend.fontsize"] = 5
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["xtick.major.width"] = 0.5
    mpl.rcParams["ytick.major.width"] = 0.5
    mpl.rcParams["xtick.major.size"] = 2
    mpl.rcParams["ytick.major.size"] = 2
    mpl.rcParams["mathtext.fontset"] = "dejavusans"

    mpl.rcParams.update(rcparams)
