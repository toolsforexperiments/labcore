from datetime import datetime
from pathlib import Path
from typing import Any, Union
from collections import OrderedDict
import logging

import asyncio
import nest_asyncio
nest_asyncio.apply()

import hvplot
import holoviews as hv

from bokeh.io.export import export_png
import pandas
import param
import panel as pn
from panel.widgets import Select
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import re

from ..data.datadict_storage import find_data, timestamp_from_path, datadict_from_hdf5
from ..data.datadict import (
    DataDict,
    dd2df,
    datadict_to_meshgrid,
    dd2xr,
)
from ..utils.misc import add_end_number_to_repeated_file
from .hvplotting import Node, labeled_widget, PlotNode

logger = logging.getLogger(__name__)


class Handler(FileSystemEventHandler):
    def __init__(self, update_callback):
        self.update_callback = update_callback

    def on_created(self, event):
        if not event.is_directory:
            # Get file extension and compare to data file type, ddh5
            file_type = Path(event.src_path).suffix
            # TODO: Generalize to other data file types
            if file_type == '.ddh5':
                self.update_callback(event)


class DataSelect(pn.viewable.Viewer):

    SYM = {
        'complete': '✅',
        'star': '😁',
        'bad': '😭',
        'trash': '❌',
        'poop': '💩',
    }
    DATAFILE = 'data.ddh5'

    selected_path = param.Parameter(None)
    search_term = param.Parameter(None)
    group_options = param.Parameter(None)

    # Used to combat Watchdogs duplicate calling events
    event_lock = False

    @staticmethod
    def date2label(date_tuple):
        return "-".join((str(k) for k in date_tuple))

    @staticmethod
    def label2date(label):
        return tuple(int(l) for l in label[:10].split("-"))

    @staticmethod
    def group_data(data_list):
        ret = {}
        for path, info in data_list.items():
            ts = timestamp_from_path(path)
            date = (ts.year, ts.month, ts.day)
            if date not in ret:
                ret[date] = {}
            ret[date][path] = info
        return ret

    def __panel__(self):
        return self.layout

    def __init__(self, data_root, size=15):
        super().__init__()

        self.size = size

        # this contains a dict with the structure:
        # { date (as tuple):
        #    { path of the dataset folder : (list of subdirs, list of files) }
        # }
        self.data_root = data_root
        self.data_sets = self.group_data(find_data(root=data_root))

        self.layout = pn.Column()

        # a search bar for data
        self.search_label = pn.widgets.StaticText(value="Search:", align='center')
        self.text_input = pn.widgets.TextInput(
            placeholder='Enter a search term here...'
        )

        # Refresh button beside search bar
        self.refresh_button = pn.widgets.Button(
            name='🔄 Refresh',
            width=100,
            button_type='default'
        )
        self.refresh_button.on_click(self._on_refresh_clicked)

        # Tag buttons
        self.star_button = pn.widgets.Button(
            name='😁 Star',
            width=100,
            button_type='default'
        )
        self.star_button.on_click(self._on_star_clicked)

        self.trash_button = pn.widgets.Button(
            name='❌ Trash',
            width=100,
            button_type='default'
        )
        self.trash_button.on_click(self._on_trash_clicked)

        self.bad_button = pn.widgets.Button(
            name='😭 Bad',
            width=100,
            button_type='default'
        )
        self.bad_button.on_click(self._on_bad_clicked)

        self.poop_button = pn.widgets.Button(
            name='💩 Error',
            width=100,
            button_type='default'
        )
        self.poop_button.on_click(self._on_poop_clicked)

        # Add search bar, refresh button, and tag buttons in a row
        self.layout.append(pn.Row(self.search_label, self.text_input, self.refresh_button, self.star_button, self.trash_button, self.bad_button, self.poop_button))

        # Display the current search term
        self.typed_value = pn.widgets.StaticText(
            stylesheets=[selector_stylesheet],
            css_classes=['ttlabel'],
        )
        self.layout.append(self.text_input_repeater)

        self.image_feed_width = 400  # The width of images in the feed
        self.feed_scroll_width = 40  # Extra width of the feed itself for the scroll bar

        # two selectors for data selection
        self._group_select_widget = pn.widgets.CheckBoxGroup(
            name='Date',
            width=200-self.feed_scroll_width,
            stylesheets=[selector_stylesheet]
        )
        # Wrap the CheckBoxGroup in a feed so that it can't get too long
        self._group_select_feed = pn.layout.Feed(
            objects=[self._group_select_widget],
            height=(self.size - 1) * 20,
            width=200
        )
        # Add a title to match the multiselect widget style
        self._group_select = pn.Column(
            pn.widgets.StaticText(
                stylesheets=[selector_stylesheet],
                css_classes=['ttlabel'],
                value="Date"
            ),
            self._group_select_feed
        )
        # Data select panel
        self._data_select_widget = Select(
            name='Data set',
            size=self.size,
            width=800,
            stylesheets=[selector_stylesheet]
        )

        # Scrollable feed of images stored with this data
        self.data_images_feed = pn.layout.Feed(None, sizing_mode="fixed")
        # Data frame showing axes & dependencies
        self.data_info = pn.pane.DataFrame(None)

        self.layout.append(pn.Row(
            self._group_select, self.data_select, self.data_info, self.data_images_feed))

        # a simple info panel about the selection
        self.lbl = pn.widgets.StaticText(
            stylesheets=[selector_stylesheet],
            css_classes=['ttlabel'],
        )

        self.layout.append(pn.Row(self.info_panel))

        opts = OrderedDict()
        for k in sorted(self.data_sets.keys())[::-1]:
            lbl = self.date2label(k) + f' [{len(self.data_sets[k])}]'
            opts[lbl] = k
        self._group_select_widget.options = opts

        # WATCHDOG INCORPORATION
        # This allows for monitoring when files are created
        self.DIRECTORY_TO_WATCH = r"."
        self.observer = Observer()
        self.handler = Handler(self.update_group_options)
        self.start()

    def start(self):
        self.observer.schedule(
            self.handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()

    @pn.depends("_group_select_widget.value")
    def data_select(self):
        # setup global variables for search function
        active_search = False
        r = re.compile(".*")
        if hasattr(self, "text_input"):
            if self.text_input.value_input is not None and self.text_input.value_input != "":
                # Make the Regex expression for the searched string
                r = re.compile(".*" + str(self.text_input.value_input) + ".*")
                active_search = True

        opts = self.get_data_options(active_search, r)

        self._data_select_widget.options = opts
        return self._data_select_widget

    def get_data_options(self, active_search=True, r=re.compile('.*')):
        if isinstance(r, str):
            r = re.compile(r)
        opts = OrderedDict()
        for d in self._group_select_widget.value:
            for dset in sorted(self.data_sets[d].keys())[::-1]:
                if active_search and not r.match(str(dset) + " " + str(timestamp_from_path(dset))):
                    # If Active search and this term doesn't match it, don't show
                    continue
                (dirs, files) = self.data_sets[d][dset]
                ts = timestamp_from_path(dset)
                time = f"{ts.hour:02d}:{ts.minute:02d}:{ts.second:02d}"
                uuid = f"{dset.stem[18:26]}"
                name = f"{dset.stem[27:]}"
                date = f"{ts.date()}"
                lbl = f"{date} - {time} - {uuid} - {name} "
                for k in ['complete', 'star', 'bad', 'trash', 'poop']:
                    if f'__{k}__.tag' in files:
                        lbl += self.SYM[k]
                opts[lbl] = dset
        return opts

    @pn.depends("_data_select_widget.value")
    def info_panel(self):
        path = self._data_select_widget.value
        # Setup data preview panel
        if path is not None:
            abs_path = path.absolute()
            # Defer heavy operations - only load images if visible, not data dict
            # This speeds up file selection significantly
            images = []
            try:
                for file in Path.iterdir(abs_path):
                    # Check if the file ends with png, jpg, or jpeg
                    file = str(file)
                    img = ''
                    if file.endswith(".png"):
                        img = pn.pane.PNG(file, sizing_mode="fixed",
                                          width=self.image_feed_width)
                    elif file.endswith(".jpg") or file.endswith(".jpeg"):
                        img = pn.pane.JPG(file, sizing_mode="fixed",
                                          width=self.image_feed_width)
                    else:
                        continue
                    images.append(img)
                    images.append(pn.Spacer(height=img.height))
            except Exception as e:
                logger.warning(f"Could not load images from {abs_path}: {e}")

            self.data_images_feed.objects = images
            self.data_images_feed.width = self.image_feed_width + self.feed_scroll_width

            # Load datadict lazily only when needed (shown) and in a thread
            try:
                dd = datadict_from_hdf5(str(abs_path) + "/data")
                dict_for_dataframe = {}
                for key in dd.keys():
                    if len(key) < 2 or key[0:2] != "__":
                        depends_on = dd[key]["axes"] if dd[key]["axes"] != [
                        ] else "Independent"
                        dict_for_dataframe[key] = [
                            dd[key]["__shape__"], depends_on]
                # Convert to data frame and display
                df = pandas.DataFrame.from_dict(
                    data=dict_for_dataframe, orient="index", columns=['Shape', 'Depends on'])
                self.data_info.object = df
            except Exception as e:
                logger.warning(f"Could not load data from {abs_path}: {e}")
                self.data_info.object = f"Error loading data: {e}"
        else:
            self.data_images_feed.objects = []
            self.data_info.object = None

        # Get the path
        if isinstance(path, Path):
            path = path / self.DATAFILE
        # Display path under the dataframe
        self.lbl.value = f"Path: {path}"
        self.selected_path = path
        return self.lbl

    @pn.depends("text_input.value_input")
    def text_input_repeater(self):
        self.typed_value.value = f"Current Search: {self.text_input.value_input}"
        self.search_term = self.text_input.value_input
        return self.typed_value

    def update_group_options(self, event):
        # Refresh self.data_sets
        new_data_set = self.group_data(find_data(root=self.data_root))
        # Repull data group options
        new_opts = OrderedDict()
        for k in sorted(new_data_set.keys())[::-1]:
            lbl = self.date2label(k) + f' [{len(new_data_set[k])}]'
            new_opts[lbl] = k
        # Set the group and data options
        self.data_sets = new_data_set
        self._group_select_widget.options = new_opts
        self._data_select_widget.options = self.get_data_options()
        self._group_select_feed.objects = [self._group_select_widget]

    def _on_refresh_clicked(self, event):
        """Callback for refresh button click. Refreshes the entire data selection."""
        # Re-scan data directory
        new_data_set = self.group_data(find_data(root=self.data_root))
        # Repull data group options
        new_opts = OrderedDict()
        for k in sorted(new_data_set.keys())[::-1]:
            lbl = self.date2label(k) + f' [{len(new_data_set[k])}]'
            new_opts[lbl] = k
        # Set the group and data options
        self.data_sets = new_data_set
        self._group_select_widget.options = new_opts
        self._data_select_widget.options = self.get_data_options()
        self._group_select_feed.objects = [self._group_select_widget]

    def _toggle_tag(self, tag_name: str, event):
        """Toggle a tag file for the selected dataset."""
        selected_path = self._data_select_widget.value
        if selected_path is None:
            logger.warning(f"No dataset selected to add {tag_name} tag")
            return

        tag_file = selected_path / f"__{tag_name}__.tag"
        try:
            if tag_file.exists():
                # Remove the tag
                tag_file.unlink()
                logger.info(f"Removed {tag_name} tag from {selected_path}")
            else:
                # Create the tag
                tag_file.touch()
                logger.info(f"Added {tag_name} tag to {selected_path}")

            # Refresh the data display to update tag indicators
            self._data_select_widget.options = self.get_data_options()
        except Exception as e:
            logger.error(f"Error toggling {tag_name} tag: {e}")

    def _on_star_clicked(self, event):
        """Callback for star button click."""
        self._toggle_tag('star', event)

    def _on_trash_clicked(self, event):
        """Callback for trash button click."""
        self._toggle_tag('trash', event)

    def _on_bad_clicked(self, event):
        """Callback for bad button click."""
        self._toggle_tag('bad', event)

    def _on_poop_clicked(self, event):
        """Callback for poop/error button click."""
        self._toggle_tag('poop', event)


selector_stylesheet = """
:host .bk-input {
    font-family: monospace;
}

:host(.ttlabel) .bk-clearfix {
    font-family: monospace;
}
"""


class LoaderNodeBase(Node):
    """A node that loads data.

    the panel of the node consists of UI options for loading and pre-processing.

    Each subclass must implement ``LoaderNodeBase.load_data``.
    """

    def __init__(self, path=Path('.'), *args: Any, **kwargs: Any):
        """Constructor for ``LoaderNode``.

        Parameters
        ----------
        *args:
            passed to ``Node``.
        **kwargs:
            passed to ``Node``.
        """
        # LoaderNodes need a datapath- this lets the super access said path
        self.file_path = path

        # to be able to watch, this needs to be defined before super().__init__
        self.refresh = pn.widgets.Select(
            name="Auto-refresh",
            options={
                'None': None,
                '2 s': 2,
                '5 s': 5,
                '10 s': 10,
                '1 min': 60,
                '10 min': 600,
            },
            value="None",
            width=80,
        )
        self.task = None

        # Determines if save_buttons need to be disabled. Must be run early
        # to avoid an 'attribute does not exist' error.
        self.disable_buttons = not self.can_save()

        super().__init__(path=path, *args, **kwargs)

        # Store whether or not each graph type can be saved as html/png.
        # Must be created before plot_col column
        self.graph_type_savable = {}
        for k in self.graph_types:
            self.graph_type_savable[k] = hasattr(
                self.graph_types[k], 'get_plot')

        self.average_toggle = pn.widgets.Switch(
            value=True, name="Average", align="center"
        )
        self.average_toggle.param.watch(self.load_and_preprocess, "value")
        self.pre_process_dim_input = pn.widgets.TextInput(
            value="rep",
            name="Average dim.",
            width=100,
            align="end",
        )
        self.pre_process_dim_input.param.watch(self.load_and_preprocess, "value")
        self.rotate_iq_toggle = pn.widgets.Switch(
            value=False, name="Rotate IQ", align="center"
        )
        self.rotate_iq_toggle.param.watch(self.load_and_preprocess, "value")
        self.rotate_iq_angle_input = pn.widgets.FloatInput(
            value=0.0,
            name="Rotate angle (deg)",
            width=100,
            align="end",
        )
        self.rotate_iq_angle_input.param.watch(self.load_and_preprocess, "value")
        self.grid_on_load_toggle = pn.widgets.Switch(
            value=True, name="Auto-grid", align="end"
        )
        self.auto_load_toggle = pn.widgets.Switch(
            value=False, name="Auto-load on select", align="center"
        )
        self.generate_button = pn.widgets.Button(
            name="Load data", align="center", button_type="primary"
        )

        # Button to save graph as html
        self.generate_button.on_click(self.load_and_preprocess)
        self.html_button = pn.widgets.Button(
            name="Make HTML", align="end", button_type="default", disabled=True
        )
        self.html_button.on_click(self.save_html)

        # Button to save graph as png
        self.png_button = pn.widgets.Button(
            name="Make PNG", align="end", button_type="default", disabled=True
        )
        self.png_button.on_click(self.save_png)

        self.info_label = pn.widgets.StaticText(name="Info", align="start")
        self.info_label.value = "No data loaded."

        # This is needed to tell the site that it needs more vertical space.
        # If there's no buffer column, the screen will jitter up and down
        # when it auto refreshes.
        self.buffer_col = pn.Column(height=600, width=10)
        self.plot_col = pn.Column(objects=self.plot)

        # The Leading pn.Row is used to make the fit box appear at right
        self.layout =pn.Column(
                pn.Row(
                    self.generate_button,
                    pn.Card(
                        pn.Row(
                            pn.Column(self.grid_on_load_toggle,self.auto_load_toggle,self.refresh,align="center"),
                        ),
                        title="Loading Options",
                        collapsed=True,
                        ),
                        pn.Card(
                            pn.Row(
                                pn.Column(
                                    self.average_toggle,
                                    self.pre_process_dim_input,
                                    align="center",
                                ),
                                pn.Column(
                                    self.rotate_iq_toggle,
                                    self.rotate_iq_angle_input,
                                    align="center",
                                ),
                            ),
                            title="Pre-processing",
                            collapsed=True,
                        ),
                        pn.Card(
                        self.fit_obj,
                        title="Fit",
                        collapsed=True,
                        ),
                        pn.Card(
                            pn.Row(
                                self.html_button,
                                self.png_button,
                            ),
                            title="Save",
                            collapsed=True,
                        ),
                    ),
                self.display_info,
                pn.Row(
                    self.buffer_col,
                    self.plot_col
                )
            )


        #Create var to collect the fitfunc inputs
        self.fit_inputs = None

        self.lock = asyncio.Lock()

    async def load_and_preprocess(self, *events: param.parameterized.Event) -> None:
        """Call load data and perform pre-processing.

        Function is triggered by clicking the "Load data" button.
        """
        async with self.lock:
            t0 = datetime.now()
            dd = self.load_data()  # this is simply a datadict now.

            # if there wasn't data selected, we can't process it
            if dd is None:
                return

            # this is the case for making a pandas DataFrame
            if not self.grid_on_load_toggle.value:
                data = self.split_complex(dd2df(dd))
                indep, dep = self.data_dims(data)

                if self.average_toggle.value and self.pre_process_dim_input.value in indep:
                    data = self.mean(
                        data, self.pre_process_dim_input.value)
                    indep.pop(indep.index(
                        self.pre_process_dim_input.value))

                if self.rotate_iq_toggle.value:
                    data = self.rotate_iq(data, self.rotate_iq_angle_input.value)

            # when making gridded data, can do things slightly differently
            # TODO: what if gridding goes wrong?
            else:
                mdd = datadict_to_meshgrid(dd)

                if self.average_toggle.value and self.pre_process_dim_input.value in mdd.axes():
                    mdd = mdd.mean(self.pre_process_dim_input.value)

                data = self.split_complex(dd2xr(mdd))
                indep, dep = self.data_dims(data)

                if self.rotate_iq_toggle.value:
                    data = self.rotate_iq(data, self.rotate_iq_angle_input.value)

            for dim in indep + dep:
                self.units_out[dim] = dd.get(dim, {}).get("unit", None)

            self.data_out = data
            t1 = datetime.now()
            self.info_label.value = f"Loaded data at {t1.strftime('%Y-%m-%d %H:%M:%S')} (in {(t1-t0).microseconds*1e-3:.0f} ms)."

            # Select None so that save buttons disabled/enabled works
            # I have no clue why but there's a bug if this doesn't happen
            save_val = self.plot_type_select.value
            self.plot_type_select.value = 'None'
            self.plot_type_select.value = save_val

    @pn.depends("data_out", "plot_type_select.value")
    def plot_obj(self):
        ret = super().plot_obj()
        self.toggle_save_buttons()
        return ret

    def can_save(self):
        # Returns TRUE is the necessary packages for saving html and images are installed
        # Returns FALSE and prints a notice about the uninstalled packages otherwise

        # Try and save an image to check packages
        try:
            test_file = Path("SAVETEST.png")
            save_as = add_end_number_to_repeated_file(test_file)
            self.refresh.save(save_as)
            save_as.unlink(missing_ok=True)
            has_packages = True
        except RuntimeError as e:
            logger.error(e)
            has_packages = False

        # Reset Plot object
        self._plot_obj = None

        if not has_packages:
            logger.warning("SAVING IMAGES DISABLED: You have not installed the necessary packages to allow for the saving of "
                           "images. To allow this functionality, please install Selenium, PhantomJS, Firefox, and "
                           "Geckodriver")

        has_packages = True
        return has_packages

    def toggle_save_buttons(self):
        if (self.disable_buttons) or (self.file_path == ""):
            # If the packages aren't installed, keep button disabled
            # If no file is selected, keep button disabled
            return
        # Checks if the graphs can be saved and disables buttons accordingly
        _disabled = not self.graph_type_savable[self.plot_type_select.value]
        self.html_button.disabled = _disabled
        self.png_button.disabled = _disabled

    def save_html(self, *events: param.parameterized.Event):
        if isinstance(self._plot_obj, PlotNode):
            file_name = add_end_number_to_repeated_file(self.file_path.parent / f"{self.file_path.parent.name}.html")
            hvplot.save(self._plot_obj.get_plot(), file_name)

    def save_png(self, *events: param.parameterized.Event):
        def save_p(p, suffix=""):
            path = add_end_number_to_repeated_file(
                self.file_path.parent.joinpath(f"{self.file_path.parent.name}{suffix}.png"))
            # FIXME: We need all of this because different plot types return different objects.
            #  It might be worth unifying it and always returning the same object.
            # If p is a Panel HoloViews pane, extract the HoloViews object
            if hasattr(p, 'object') and hasattr(p.object, 'traverse'):
                # This is a Panel HoloViews pane - extract the HoloViews object
                hv_obj = p.object
            elif hasattr(p, 'traverse'):
                # This is already a HoloViews object
                hv_obj = p
            else:
                # Skip objects that are not HoloViews or Panel HoloViews panes
                logger.warning(f"Skipping object of type {type(p)} - not a HoloViews object")
                return

            bokeh_plot = hv.render(hv_obj)
            export_png(bokeh_plot, filename=path)

        # TODO: What happens when this is not a Node? What should happen then?
        if isinstance(self._plot_obj, PlotNode):

            plot = self._plot_obj.get_plot()
            if isinstance(plot, pn.Column):
                for i, p in enumerate(plot):
                    suffix = f"_plot{i+1}" if len(plot) > 1 else ""
                    save_p(p, suffix)
            else:
                save_p(plot)

    @pn.depends("info_label.value")
    def display_info(self):
        return self.info_label

    @pn.depends("refresh.value", watch=True)
    def on_refresh_changed(self):
        if self.refresh.value is None:
            self.task = None

        if self.refresh.value is not None:
            if self.task is None:
                self.task = asyncio.ensure_future(self.run_auto_refresh())

    async def run_auto_refresh(self):
        while self.refresh.value is not None:
            await asyncio.sleep(self.refresh.value)
            asyncio.run(self.load_and_preprocess())
        return

    def load_data(self) -> DataDict:
        """Load data. Needs to be implemented by subclasses.

        Raises
        ------
        NotImplementedError
            if not implemented by subclass.
        """
        raise NotImplementedError

Label_stylesheet = """
:host {
    font-family: monospace;
    font-size: 20px;
}
"""

class DDH5LoaderNode(LoaderNodeBase):
    """A node that loads data from a specified file location.

    the panel of the node consists of UI options for loading and pre-processing.

    """

    file_path = param.Parameter(None)

    def __init__(self, path: Union[str, Path] = "", *args: Any, **kwargs: Any):
        """Constructor for ``LoaderNodePath``.

        Parameters
        ----------
        *args:
            passed to ``Node``.
        **kwargs:
            passed to ``Node``.
        """
        super().__init__(path, *args, **kwargs)
        self.file_path = path

    def load_data(self) -> DataDict:
        """
        Load data from the file location specified
        """
        # Check in case no data is selected
        if str(self.file_path) == "":
            self.info_label.value = "Please select data to load. If there is no data, trying running in a higher directory."
            return None

        return datadict_from_hdf5(self.file_path.absolute())
