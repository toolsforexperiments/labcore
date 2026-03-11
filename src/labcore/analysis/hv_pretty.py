from pathlib import Path
from typing import Any, Optional

import holoviews as hv
import hvplot
import seaborn as sns
from PIL import Image

hv.extension("bokeh")


# Convert inches to pixels
def correctly_sized_figure(width: float = 6, height: float = 4) -> dict[str, int]:
    """Returns width and height in pixels from inches."""
    return {"width": int(width * 300), "height": int(height * 300)}


# Set Arial font for all text elements
def set_arial_font(plot: Any, element: Any = None) -> None:
    """Applies Arial font to all textual elements of a bokeh-based hvplot."""
    p = plot.state
    p.title.text_font = "Arial"
    p.title.text_font_style = "normal"

    for ax in p.axis:
        ax.axis_label_text_font = "Arial"
        ax.axis_label_text_font_style = "normal"
        ax.major_label_text_font = "Arial"
        ax.major_label_text_font_style = "normal"

    if hasattr(p, "legend"):
        for item in p.legend:
            item.label_text_font = "Arial"
            item.label_text_font_style = "normal"


# Axis and label formatting
def format_ax(
    plot: Any,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    fontsize: int = 12,
    title_fontsize: int = 14,
    xticks: Any = None,
    yticks: Any = None,
    xlim: Any = None,
    ylim: Any = None,
    axes_pad: float = 0.05,
    tick_fontsize: int = 10,
) -> Any:
    """Apply axis and label formatting options to a hvplot object."""
    opts_dict = {
        "title": title,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "xlim": xlim,
        "ylim": ylim,
        "xticks": xticks,
        "yticks": yticks,
        "padding": axes_pad,
        "fontsize": {
            "title": f"{title_fontsize}pt",
            "labels": f"{fontsize}pt",
            "xticks": f"{tick_fontsize}pt",
            "yticks": f"{tick_fontsize}pt",
            "legend": f"{fontsize}pt",
        },
    }

    plot = plot.opts(**{k: v for k, v in opts_dict.items() if v is not None})
    plot.opts(responsive=False)
    return plot


# Add legend to Overlay or Layouts
def add_legend(plot: Any, location: str = "top_right", show: bool = True) -> Any:
    """Configure legend visibility and position."""
    if isinstance(plot, hv.Overlay) or isinstance(plot, hv.Layout):
        plot = plot.opts(show_legend=show, legend_position=location)
    return plot


# Setup seaborn style
def setup_plotting(
    style: str = "whitegrid", context: str = "notebook", font_scale: float = 1.2
) -> None:
    """Sets up seaborn styling globally for consistency with matplotlib-style aesthetics."""
    sns.set_style(style)
    sns.set_context(context, font_scale=font_scale)


def save_plot_as_png(
    plot: Any,
    filename: Any,
    width_in: float = 6,
    height_in: float = 4,
    dpi: int = 300,
    embed_dpi: bool = True,
) -> None:
    """
    Save a Holoviews plot to a high-resolution PNG with embedded DPI metadata.

    Parameters:
    - plot      : Holoviews object (e.g. from hvplot)
    - filename  : Target output PNG file (e.g. 'figure.png')
    - width_in  : Width in inches (default: 6)
    - height_in : Height in inches (default: 4)
    - dpi       : Dots per inch (default: 300)
    - embed_dpi : Whether to embed DPI metadata using PIL (default: True)
    """
    # Convert filename to Path
    output_path = Path(filename).resolve()
    tmp_path = output_path.with_name("_tmp_hvplot_export.png")

    # Calculate pixel dimensions
    width_px = int(width_in * dpi)
    height_px = int(height_in * dpi)

    # Apply size to plot
    plot = plot.opts(width=width_px, height=height_px)

    # Save to temporary file using Holoviews
    hvplot.save(plot, tmp_path, fmt="png")

    if embed_dpi:
        img = Image.open(tmp_path)
        img.save(output_path, dpi=(dpi, dpi))
        tmp_path.unlink()  # Delete temp file
    else:
        tmp_path.rename(output_path)
