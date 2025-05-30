import seaborn as sns
import holoviews as hv
import hvplot
from PIL import Image
from pathlib import Path

hv.extension('bokeh')

# Convert inches to pixels
def correctly_sized_figure(width=6, height=4):
    """Returns width and height in pixels from inches."""
    return {'width': int(width * 300), 'height': int(height * 300)}

# Set Arial font for all text elements
def set_arial_font(plot, element=None):
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
def format_ax(plot, title=None, xlabel=None, ylabel=None,
              fontsize=12, title_fontsize=14,
              xticks=None, yticks=None, xlim=None, ylim=None,
              axes_pad=0.05, tick_fontsize=10):
    """Apply axis and label formatting options to a hvplot object."""
    opts_dict = {
        'title': title,
        'xlabel': xlabel,
        'ylabel': ylabel,
        'xlim': xlim,
        'ylim': ylim,
        'xticks': xticks,
        'yticks': yticks,
        'padding': axes_pad,
        'fontsize': {
            'title': f"{title_fontsize}pt",
            'labels': f"{fontsize}pt",
            'xticks': f"{tick_fontsize}pt",
            'yticks': f"{tick_fontsize}pt",
            'legend': f"{fontsize}pt"
        }
    }
    
    plot = plot.opts(**{k: v for k, v in opts_dict.items() if v is not None})
    plot.opts(responsive=False)
    return plot

# Add legend to Overlay or Layouts
def add_legend(plot, location='top_right', show=True):
    """Configure legend visibility and position."""
    if isinstance(plot, hv.Overlay) or isinstance(plot, hv.Layout):
        plot = plot.opts(show_legend=show, legend_position=location)
    return plot

# Setup seaborn style
def setup_plotting(style='whitegrid', context='notebook', font_scale=1.2):
    """Sets up seaborn styling globally for consistency with matplotlib-style aesthetics."""
    sns.set_style(style)
    sns.set_context(context, font_scale=font_scale)


def save_plot_as_png(plot, filename, width_in=6, height_in=4, dpi=300, embed_dpi=True):
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
    hvplot.save(plot, tmp_path, fmt='png')

    if embed_dpi:
        img = Image.open(tmp_path)
        img.save(output_path, dpi=(dpi, dpi))
        tmp_path.unlink()  # Delete temp file
    else:
        tmp_path.rename(output_path)