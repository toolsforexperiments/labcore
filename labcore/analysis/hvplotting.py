"""HoloViews-based plotting for labstack.
Contains a set of classes and functions that can be used to plot labstack-style data.

Important Classes:
    - Nodes:
        - Node: a base class for all nodes. Nodes are the basic blocks we use for
            processing data. They can be chained together to form a pipeline.
        - LoaderNode: a node that loads and preprocesses data.
        - ReduxNode: a node that reduces data dimensionality (e.g. by averaging).
        - ValuePlot: plots data values.
        - ComplexHist: plots histograms of complex data ('IQ readout histograms').

    - Widgets:
        - XYSelect: a widget for selecting x and y axes.


#TODO:

    * When fitting real and imaginary data, the color of the dot and line between data crosses. For example if the imaginary line is red with dots red, after adding a fit to the real data, the dots of imaginary will remain red, but the lines will be orange and the dots of the fit will orange but the lines red
    * Program refuses to open if it cannot find the config file.
    * Program refuses to open if it cannot find data properly formatted. This might be ok but a nice error message should appear.
"""

import copy
import importlib
import inspect
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import panel as pn
import param
import ruamel.yaml
import xarray as xr
from panel.widgets import RadioButtonGroup as RBG

from ..data.datadict_storage import NumpyEncoder
from ..data.tools import data_dims, split_complex
from ..utils.misc import add_end_number_to_repeated_file
from .fit import Fit, FitResult, plot_ds_2d_with_fit

logger = logging.getLogger(__name__)

pn.extension(notifications=True)

Data = Union[xr.Dataset, pd.DataFrame]
"""Type alias for valid data. Can be either a pandas DataFrame or an xarray Dataset."""

DataDisplay = Optional[Union[pn.pane.DataFrame, xr.Dataset, str]]
"""Type alias for displaying raw data."""


class Node(pn.viewable.Viewer):
    """Node base class.

    A simple wrapper class that we use to standardize the way we process data.
    Aim: whenever input data is set/updated, ``Node.process`` is called (in the
    base class it simply sets output equal to input).
    User-defined nodes may watch ``data_in`` and ``data_out`` to update UI or
    anything else.
    Pipelines are formed by appending nodes to each other using ``Node.append``.

    Nodes with graphs must implement a ``get_plot()`` function that returns
    a holoviews graph object in order for that graph to be able to be saved
    as either an html or png file.

    Params
    ------
    data_in
        input data. Must be either a pandas DataFrame or an xarray Dataset.
    data_out
        processed output data. Must be either a pandas DataFrame or an xarray Dataset.
    units_in
        units of input data. Must be a dictionary with keys corresponding to dimensions,
        and values corresponding to units.
    units_out
        units of output data. Must be a dictionary with keys corresponding to dimensions,
        and values corresponding to units.
    meta_in
        any input metadata. Arbitrary keys/value.
    meta_out
        any output metadata. Arbitrary keys/value.
    """

    data_in = param.Parameter(None)
    data_out = param.Parameter(None)

    # -- important metadata
    units_in = param.Parameter({})
    units_out = param.Parameter({})
    meta_in = param.Parameter({})
    meta_out = param.Parameter({})

    def __panel__(self) -> pn.viewable.Viewable:
        return self.layout

    def __init__(self, path=None, data_in: Optional[Data] = None, *args: Any, **kwargs: Any):
        """Constructor for ``Node``.

        Parameters
        ----------
        data_in
            Optional input data.
        *args:
            passed to ``pn.viewable.Viewer``.
        **kwargs:
            passed to ``pn.viewable.Viewer``.

        """
        self._watchers: Dict[Node, param.parameterized.Watcher] = {}

        super().__init__(*args, **kwargs)
        self.layout = pn.Column()

        # -- options for plotting
        self.graph_types = {"None": None,
                            "Value": ValuePlot,
                            "Readout hist.": ComplexHist,
                            "Magnitude & Phase": MagnitudePhasePlot}

        self.plot_type_select = RBG(
            options=list(self.graph_types.keys()),
            value="Value",
            name="View as",
        )
        self._plot_obj: Optional[Node] = None

        if data_in is not None:
            self.data_in = data_in
            self.process()

        # file_path is needed here so it can be passed to plots for fit saving
        self.file_path = path

    @staticmethod
    def render_data(data: Optional[Data]) -> DataDisplay:
        """Shows data as renderable object.

        Raises
        ------
        NotImplementedError
            if data is not a pandas DataFrame or an xarray Dataset.
        """
        if data is None:
            return "No data"

        if isinstance(data, pd.DataFrame):
            return pn.pane.DataFrame(data, max_rows=20, show_dimensions=True)
        elif isinstance(data, xr.Dataset):
            return data
        else:
            raise NotImplementedError

    @staticmethod
    def data_dims(data: Optional[Data]) -> tuple[list[str], list[str]]:
        """Returns the dimensions of the data.

        Format: (independents, dependents); both as lists of strings.

        Raises
        ------
        NotImplementedError
            if data is not a pandas DataFrame or an xarray Dataset.
        """
        return data_dims(data)

    @staticmethod
    def mean(data: Data, *dims: str) -> Data:
        """Takes the mean of data along the given dimensions.

        Parameters
        ----------
        data
            input data.
        *dims
            dimensions to take the mean along

        Returns
        -------
        data after taking the mean

        Raises
        ------
        NotImplementedError
            if data is not a pandas DataFrame or an xarray Dataset.
        """
        indep, dep = Node.data_dims(data)
        if isinstance(data, pd.DataFrame):
            for d in dims:
                i = indep.index(d)
                indep.pop(i)
            return data.groupby(level=tuple(indep)).mean()
        elif isinstance(data, xr.Dataset):
            for d in dims:
                data = data.mean(d)
            return data
        else:
            raise NotImplementedError

    @staticmethod
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
        return split_complex(data)

    @staticmethod
    def complex_dependents(data: Optional[Data]) -> dict[str, dict[str, str]]:
        """Returns a dictionary of complex dependents and their real/imaginary parts.

        Requires that complex data has already been split.

        Parameters
        ----------
        data
            input data.

        Returns
        -------
        dictionary of the form:
            "`dependent`": {"real": "`dependent_Re`", "imag": "`dependent_Im`"}}
            `dependent_Re` and `dependent_Im` are the dimensions actually present
            in the data.
        """
        ret = {}
        _, dep = Node.data_dims(data)
        for d in dep:
            if d[-3:] == "_Re":
                im_dep = d[:-3] + "_Im"
                if im_dep in dep:
                    ret[d[:-3]] = dict(real=d, imag=im_dep)
        return ret

    def dim_label(self, dim: str, which: str = "out") -> str:
        """Generate dimension label for use in plots.

        Parameters
        ----------
        dim
            dimension name.
        which
            Either "in" or "out", depending on whether we want the input or
            output data of the Node. Default is "out".

        Returns
        -------
        dimension label, including units if available.
        """
        if which == "out":
            units = self.units_out
        else:
            units = self.units_in

        if dim in units and units[dim] is not None:
            return f"{dim} ({units[dim]})"
        else:
            return f"{dim} (a.u.)"

    def dim_labels(self, which: str = "out") -> dict[str, str]:
        """Generate dimension labels for use in plots.

        Generates all dimension labels for the data.
        See ``Node.dim_label`` for more information.
        """
        if which == "out":
            indep, dep = self.data_dims(self.data_out)
        else:
            indep, dep = self.data_dims(self.data_in)
        dims = indep + dep
        return {d: self.dim_label(d, which=which) for d in dims}

    def update(self, *events: param.parameterized.Event) -> None:
        """Update the node using external events.

        If event contains ``data_out``, ``units_out``, or ``meta_out``,
        will set them as ``data_in``, ``units_in``, or ``meta_in`` respectively.
        """
        for e in events:
            if e.name == "data_out":
                self.data_in = e.new
            elif e.name == "units_out":
                self.units_in = e.new
            elif e.name == "meta_out":
                self.meta_in = e.new

    @pn.depends("data_in", watch=True)
    def process(self) -> None:
        """Process data.

        By default, simply sets ``data_out`` equal to ``data_in``.

        Can/Should be overridden by subclasses to do more complicated things.
        """
        self.data_out = self.data_in

    @pn.depends("data_in")
    def data_in_view(self) -> DataDisplay:
        """Updating view of input data (as table; as provided by the data type).

        Updates on change of ``data_in``.
        """
        return self.render_data(self.data_in)

    @pn.depends("data_out")
    def data_out_view(self) -> DataDisplay:
        """Updating view of output data (as table; as provided by the data type).

        Updates on change of ``data_out``.
        """
        return self.render_data(self.data_out)

    @pn.depends("data_out")
    def plot(self) -> pn.viewable.Viewable:
        """A reactive panel object that allows selecting a plot type, and shows the plot.

        Updates on change of ``data_out``.
        """
        return [labeled_widget(self.plot_type_select),
                self.plot_obj]

    @pn.depends("data_out", "plot_type_select.value")
    def plot_obj(self) -> Optional["Node"]:
        """The actual plot object.

        Updates on change of ``data_out`` or the selection of the plot value.

        Returns
        -------
        A dedicated plotting node.
        """
        if self.plot_type_select.value == "Value":
            if not isinstance(self._plot_obj, ValuePlot):
                if self._plot_obj is not None:
                    self.detach(self._plot_obj)
                self._plot_obj = ValuePlot(
                    name="plot", data_in=self.data_out, path=self.file_path)
                self.append(self._plot_obj)
                self._plot_obj.data_in = self.data_out

        elif self.plot_type_select.value == "Magnitude & Phase":
            if not isinstance(self._plot_obj, MagnitudePhasePlot):
                if self._plot_obj is not None:
                    self.detach(self._plot_obj)
                self._plot_obj = MagnitudePhasePlot(
                    name="plot", data_in=self.data_out, path=self.file_path)
                self.append(self._plot_obj)
                self._plot_obj.data_in = self.data_out

        elif self.plot_type_select.value == "Readout hist.":
            if not isinstance(self._plot_obj, ComplexHist):
                if self._plot_obj is not None:
                    self.detach(self._plot_obj)
                self._plot_obj = ComplexHist(
                    name="plot", data_in=self.data_out, path=self.file_path)
                self.append(self._plot_obj)
                self._plot_obj.data_in = self.data_out

        else:
            if self._plot_obj is not None:
                self.detach(self._plot_obj)
            self._plot_obj = self.data_out_view

        return self._plot_obj

    def append(self, other: "Node") -> None:
        watcher = self.param.watch(
            other.update, ["data_out", "units_out", "meta_out"])
        self._watchers[other] = watcher

    def detach(self, other: "Node") -> None:
        if other in self._watchers:
            self.param.unwatch(self._watchers[other])
            del self._watchers[other]

    @pn.depends("data_out", "plot_type_select.value")
    def fit_obj(self):
        # Returns the panel to select fit variables
        if isinstance(self._plot_obj, PlotNode):
            return self._plot_obj.get_fit_panel
        return pn.Column()


class ReduxNode(Node):
    OPTS = ["None", "Mean"]

    coordinates = param.List(default=[])
    operations = param.List(default=[])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._widgets = {}
        self.layout = pn.Column()

    def __panel__(self):
        return self.layout

    @pn.depends("data_in", watch=True)
    def on_input_update(self):
        self.coords = list(self.data_in.dims.keys())
        for c in self.coords:
            if c not in self._widgets:
                w = RBG(name=c, options=self.OPTS, value=self.OPTS[0])
                ui = pn.Row(f"{c}", w)
                self.layout.append(ui)
                self._widgets[c] = {
                    "widget": w,
                    "ui": ui,
                    "change_cb": w.param.watch(self.on_widget_change, ["value"]),
                }

        for c in list(self._widgets.keys()):
            if c not in self.coords:
                self.layout.remove(self._widgets[c]["ui"])
                del self._widgets[c]

        self.on_widget_change()

    @pn.depends("operations")
    def on_operations_change(self):
        for c, o in zip(self.coords, self.operations):
            self._widgets[c].value = o

    def on_widget_change(self, *events):
        self.operations = [self._widgets[c]
                           ["widget"].value for c in self.coords]

    @pn.depends("data_in", "operations", watch=True)
    def process(self):
        out = self.data_in
        for c, o in zip(self.coords, self.operations):
            if o == "Mean":
                out = out.mean(c)
        self.data_out = out


class XYSelect(pn.viewable.Viewer):
    value = param.Tuple(default=("None", "None"))
    options = param.List(
        default=[
            "None",
        ]
    )

    def __init__(self):
        self._xrbg = RBG(options=self.options, name="x")
        self._yrbg = RBG(options=self.options, name="y")
        super().__init__()
        self._layout = pn.Column(
            labeled_widget(self._xrbg),
            labeled_widget(self._yrbg),
        )

        self._sync_x()
        self._sync_y()

    def __panel__(self):
        return self._layout

    @param.depends("options", watch=True)
    def on_option_change(self):
        self._xrbg.options = self.options
        self._yrbg.options = self.options

    @param.depends("value", watch=True)
    def _sync_widgets(self):
        if self.value[0] == self.value[1] and self.value[0] != "None":
            self.value = self.value[0], "None"
        self._xrbg.name = self.name
        self._xrbg.value = self.value[0]
        self._yrbg.value = self.value[1]

    @param.depends("_xrbg.value", watch=True)
    def _sync_x(self):
        x = self._xrbg.value
        y = self.value[1]
        if y == x:
            y = "None"
        self.value = (x, y)

    @param.depends("_yrbg.value", watch=True)
    def _sync_y(self):
        y = self._yrbg.value
        x = self.value[0]
        if y == x:
            x = "None"
        self.value = (x, y)


# -- generic plot functions

class PlotNode(Node):
    """PlotNode, Node subclass

    A superclass of all nodes that make plots of some sort. Mostly
    deals with define/creating fits for whatever graph subnode is instantiated.
    """
    refresh_graph = param.Parameter(None)

    FITS = None

    @staticmethod
    def load_fits_from_config():
        PlotNode.FITS = {}
        yaml = ruamel.yaml.YAML()
        cwd = Path.cwd()
        config_path = cwd / "autoplotConfig.yml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Could not find config file at {config_path}. Please add a file called `autoplotConfig.yml` wherever you are running autoplot."
            )
        raw_config = yaml.load(config_path)
        # Add fits to PlotNode.FITS if fits in the Config
        if 'fits' in raw_config:
            for ff in raw_config['fits']:
                # Module and Class name from the string
                modname = str(ff).rsplit('.', 1)
                mod = modname[0]
                try:
                    module = importlib.import_module(mod)
                    name = modname[1]
                    # Add to FITS
                    PlotNode.FITS[name] = getattr(module, name)
                except Exception as e:
                    msg = f"Could not access Class {modname[1]} from module {modname[0]}. Exception: {e}"
                    logger.error(msg)

    def __init__(self, path=None, *args, **kwargs):

        self.path = path

        if PlotNode.FITS is None:
            self.load_fits_from_config()

        self.fit_dict: dict = {}
        # Create fit_dict from json based on path
        if path == '.' or path == '' or path is None:
            self.fit_data_path = None
        else:
            _dir = Path(path).parent
            self.fit_data_path = _dir.joinpath("fit_data.json")

        self.fit_result = None
        self.fit_dict = {}

        # Initialize _fit_axis_options to avoid errors when there's no data.
        # This should get properly set in the process() function
        self._fit_axis_options = []

        # Initialize fit layout variables so they can be checked
        self.fit_inputs = None
        self.save_fit_button = None
        self.reguess_fit_button = None
        self.model_fit_button = None
        self.refit_button = None
        self.fit_box = None

        # A toggle variable to refresh the graph
        self.refresh_graph = True

        super().__init__(*args, **kwargs)

        fit_options = list(PlotNode.FITS.keys())
        fit_options.append('None')
        self.fit_button = pn.widgets.MenuButton(
            name="Fit", items=fit_options, button_type='success', width=100
        )
        self.fit_button.on_click(self.set_fit_box)

        self.select_fit_axis = pn.widgets.Select(
            name='Fit Axis',
            options=self._fit_axis_options if self._fit_axis_options is not None else [],
        )
        self.select_fit_axis.param.watch(self.set_fit_box, 'value')

        self.fit_layout = pn.Column(
            pn.Row(self.fit_button,
                   self.select_fit_axis),
        )

    def get_fit_panel(self):
        return self.fit_layout
    
    def process(self):
        """Make a copy of the data so that changes (added fits) don't carry
        to other graphs/other analysis.
        Add saved arguments for all fits/axes."""
        self.data_out = copy.copy(self.data_in)
        # Set fit_axis_options based on the function. Default to [] if returns None
        self._fit_axis_options = self.fit_axis_options()
        # Draw any fits that already exist
        for axis in self.fit_axis_options():
            if axis in self.fit_dict.keys():
                func_name = self.fit_dict[axis]['fit_function']
                saved_args = self.get_values(axis)
                if func_name not in PlotNode.FITS:
                    msg = f"Axis {axis} has a fit of type {func_name} saved, which you don't have access to."
                    print(msg)
                    pn.state.notifications.error(msg, duration=0)
                else:
                    self.update_dataset_by_fit_and_axis(PlotNode.FITS[func_name], saved_args, axis, True)

    def plot_panel(self):
        """Creates and returns a panel with the class's plot

        Should be overridden by subclasses to return the desired plot.
        """
        return NotImplementedError

    def get_plot(self):
        """Returns the plot as a holoviews object

        Used for saving the plot as html or png.

        Can/Should be overridden by subclasses to return the appropriate object.
        """
        return self.plot_panel()

    def fit_axis_options(self) -> list:
        """Returns a list of the different axes you can 
        make a fit for in this node.

        Should be overridden by subclasses to return the appropriate object.
        """
        return []

    def set_fit_box(self, *events: param.parameterized.Event, fitted:bool=None):
        if fitted is None:
            fitted = False
            if self.select_fit_axis.value in self.fit_dict.keys():
                if 'start_params' not in self.fit_dict[self.select_fit_axis.value].keys():
                    fitted = True
        # Delete start parameters when refreshing the fit box. Creating the 
        # fit box will regenerate these.
        if self.select_fit_axis.value in self.fit_dict.keys():
            if 'start_params' in self.fit_dict[self.select_fit_axis.value].keys():
                del self.fit_dict[self.select_fit_axis.value]['start_params']
        self.set_fit_box_helper(self.fit_button.clicked !=
                                'None', self.fit_button.clicked, fitted=fitted)

    def set_fit_box_helper(self, new_box: bool, fit_func_name: str, fitted: bool=False):
        """Removes and/or creates a fit box. If new_box == True this
        will (re)create the fit box."""
        # Check if fit_box exists & get fit_box
        if self.fit_box is None:
            if not new_box:
                return
            self.fit_box = self.add_fit_box(fit_func_name, fitted=fitted)
        else:
            self.remove_fit_box()
            if new_box:
                self.fit_box = self.add_fit_box(fit_func_name, fitted=fitted)

    def remove_fit_box(self):
        fit_box = self.fit_layout.objects[len(self.fit_layout.objects)-1]
        # Get all fit objects other than layout and set as the current objects
        no_fit_objects = self.fit_layout.objects[:-1]
        self.fit_layout.objects = no_fit_objects
        self.fit_inputs = None
        self.save_fit_button = None
        self.fit_box = None

    def add_fit_box(self, selected=None, fitted=False):
        """Create a widget box for creating a fit."""
        if selected is None:
            selected = self.fit_button.clicked
        if self.select_fit_axis.value in self.fit_dict.keys():
            if self.fit_dict[self.select_fit_axis.value]['fit_function'] != selected:
                fitted = False
        # Create a widget box, add the name of the fit function at top
        objs = [pn.widgets.StaticText(
            name='FITTED' if fitted else 'Setup',
            value=selected,
            align="center",
        )]
        fit_class = PlotNode.FITS[selected]
        # Get guesses or saved values for all variables and make inputs
        saved_args = self.get_arguments()
        for i, var in enumerate(inspect.signature(fit_class.model).parameters.keys()):
            if var == "coordinates":
                # User wont input coords
                continue
            name = var
            if fitted:
                objs.append( pn.widgets.StaticText(
                name=var,
                value=str(saved_args[var]) + " +/- " + str(self.fit_dict[self.select_fit_axis.value]['params'][var]['stderr']),
                align="start",
                ))
            else:
                objs.append(pn.widgets.FloatInput(
                    name=name,
                    # Set value to the saved_args (or Ansatz) or to 0
                    value=saved_args[var] if var in list(saved_args.keys()) else 0,
                )
            )
            objs[i].param.watch(self.update_fit_args, 'value')
        # Add buttons to model the fit, reset the fit
        self.reguess_fit_button = pn.widgets.Button(
            name="Reguess", align="center", button_type="default", disabled=False
        )
        self.reguess_fit_button.on_click(self.reguess_fit)

        self.save_fit_button = pn.widgets.Button(
            name="Save", align="center", button_type="success", disabled=False
        )
        self.save_fit_button.on_click(self.save_fit)

        self.model_fit_button = pn.widgets.Button(
            name="Run Fit", align="center", button_type="default", disabled=False
        )
        self.model_fit_button.on_click(self.model_fit)

        self.refit_button = pn.widgets.Button(
            name="Refit", align="center", button_type="default", disabled=False
        )
        self.refit_button.on_click(self.set_fit_box)

        if fitted:
            objs.append(pn.Row(self.save_fit_button, self.refit_button))
        else:
            objs.append(pn.Row(self.model_fit_button, self.reguess_fit_button))
        # Add to the layout
        self.fit_inputs = pn.WidgetBox(name=selected,
                                       objects=objs
                                       )
        self.fit_layout.append(pn.Row(objects=[self.fit_inputs], name="fit_box"))

        # Save to fit_dict
        if self.select_fit_axis.value not in self.fit_dict.keys():
            self.fit_dict[self.select_fit_axis.value] = {
                'fit_function': self.fit_button.clicked, 'start_params': saved_args, 'params': {}}
        # Resave to json for case of new fit class
        elif 'start_params' not in self.fit_dict[self.select_fit_axis.value]:
            self.fit_dict[self.select_fit_axis.value]['start_params'] = saved_args
        self.fit_dict[self.select_fit_axis.value]['fit_function'] = self.fit_button.clicked
        self.update_fit_args(None)
        return self.fit_inputs

    def save_fit(self, *events: param.parameterized.Event):
        """Saves only the currently selected fit axis to the json file"""
        params_dict = self.fit_result.params_to_dict()
        params_path = add_end_number_to_repeated_file(self.path.parent.joinpath("fit_params.json"))

        fit_result = self.fit_result.lmfit_result.fit_report()
        result_path = add_end_number_to_repeated_file(self.path.parent.joinpath("fit_result.txt"))

        with open(params_path, "w") as outfile:
            json.dump(params_dict, outfile, cls=NumpyEncoder)

        with open(result_path, "w") as outfile:
            outfile.write(fit_result)

    def reguess_fit(self, event):
        """Resets the current args to the results of the Fit.guess() function for the
        current fit class."""
        # Save the parameters from the current fit
        store_params = self.fit_dict[self.select_fit_axis.value]
        del self.fit_dict[self.select_fit_axis.value]
        # Redo the box (will add the axis back to fit_dict)
        self.set_fit_box_helper(
            True, store_params['fit_function'])
        # Restore parameters of current saved fit and refresh graph
        self.fit_dict[self.select_fit_axis.value]['params'] = store_params['params']
        self.refresh_graph = True

    def model_fit(self, *events: param.parameterized.Event):
        """Models the fit starting with the arguments already created"""
        # Get fit class, axis name, coordinate values
        fit_class = PlotNode.FITS[self.fit_button.clicked]
        data_key = self.select_fit_axis.value
        np_data = [self.data_out[var].values for var in self.data_out.coords]
        coords = np_data[0]
        coord_dim = self.indep_dims()
        if coord_dim == 2:
            coords = np_data[0:2]
        vals = self.data_out.data_vars[data_key].to_numpy()
        # Run the fit on the fit class
        fit = fit_class(coords, vals)
        run_kwargs = self.fit_dict[self.select_fit_axis.value]['start_params']
        self.fit_result = fit.run(**run_kwargs)
        # Get the Fit Result's arguments
        params_dict = self.fit_result.params_to_dict()
        fit_params = {}
        for k, v in params_dict.items():
            fit_params[k] = v['value']
        # Update the dataset with the new data
        name = self.select_fit_axis.value
        self.update_dataset_by_fit_and_axis(fit_class, fit_params, name, saved=True)
        self.fit_dict[self.select_fit_axis.value]['params'] = params_dict
        # switch to fitted fit_box
        self.set_fit_box(None, fitted=True) 
        self.refresh_graph = True

    def get_arguments(self):
        """Gets argument values for the currently selected fit and
        fit axis. Pulls data from the json if one exists, otherwise
        runs fit.guess and takes those values."""
        axis = self.select_fit_axis.value
        fit_name = self.fit_button.clicked
        if axis in self.fit_dict.keys() and self.fit_dict[axis]['fit_function'] == fit_name:
            return self.get_values(axis)
        else:
            return self.get_ansatz()

    def get_ansatz(self):
        fit_class = PlotNode.FITS[self.fit_button.clicked]
        # Get the guess for this data and x
        # Set data_key to first data key & make numpy data
        data_key = self.select_fit_axis.value
        np_data = [self.data_out[var].values for var in self.data_out.coords]
        # Get Ansatz using fit's 'guess' function
        coord_dim = self.indep_dims()
        coords = np_data[0]
        if coord_dim == 2:
            coords = np_data[0:2]
        return fit_class.guess(coords, self.data_out.data_vars[data_key].to_numpy())

    def update_fit_args(self, event):
        """Updates the temporary saved value for all of the fit's starting arguments.

        Called whenever a float input's value is changed, when the fitbox is 
        created, or when the fit_axis changes. """
        if self.select_fit_axis.value not in self.fit_dict.keys():
            self.fit_dict[self.select_fit_axis.value] = {
                'fit_function': self.fit_button.clicked, 'start_params': {}}
        for i, obj in enumerate(self.fit_inputs.objects):
            if isinstance(obj, pn.widgets.FloatInput):
                self.fit_dict[self.select_fit_axis.value]['start_params'][obj.name] = self.fit_inputs[i].value

        fit_class = PlotNode.FITS[self.fit_button.clicked]
        params = self.fit_dict[self.select_fit_axis.value]['start_params']
        self.update_dataset_by_fit_and_axis(fit_class, params, self.select_fit_axis.value)

        self.refresh_graph = True

    def update_dataset_by_fit_and_axis(self, fit_class: Fit,
                                       model_args: dict[str, float],
                                       model_axis_name: str, saved: bool = False):
        """Updates the data for fit in the self.data_out dataset based
        on the given arguments."""
        # Create np array of coordinates
        np_data = [self.data_out[var].values for var in self.data_out.coords]
        coords = np_data[0]
        coord_dim = self.indep_dims()
        if coord_dim == 2:
            coords = np_data[0:2]
        # Model the data, name it, and add to self.data_out
        fit_data = fit_class.model(coords, **model_args)
        fit_name = model_axis_name+"_fit"
        fit_name_temp = model_axis_name+"_fit*"
        if not saved:
            # If not saved, deleted save data and add * to name
            if fit_name in self.data_out.keys():
                del self.data_out[fit_name]
            fit_name = fit_name_temp
        else:
            # If saved, delete unsaved data
            if fit_name_temp in self.data_out.keys():
                del self.data_out[fit_name_temp]
        self.update_dataset_by_data(fit_data, fit_name)
    
    def update_dataset_by_data(self, fit_data:np.ndarray, name:str):
        # Get independent variable(s) and fit class
        indep, dep = self.data_dims(self.data_out)
        self.data_out[name] = (indep, fit_data)

    def get_data_fit_names(self, axis_name, omit_axes=None):
        if omit_axes is None:
            omit_axes = ['Magnitude', 'Phase']

        # Check if a fit axis exists. Return list of axis and fit axis (if it exists)
        if isinstance(axis_name, list):
            # If given name is a list, loop through all names in list
            ret = []
            for name in axis_name:
                ret = ret + self.get_data_fit_names(name)
            return ret
        # axis_name is a string:
        # Omit axis based on passed omissions
        if axis_name in omit_axes:
            return []
        # Check if a _fit or _fit* version of the name exists
        fit_name = axis_name + "_fit"
        ret = [axis_name]
        if fit_name+"*" in self.data_out.data_vars.keys():
            ret.append(fit_name+"*")
        elif fit_name in self.data_out.data_vars.keys():
            ret.append(fit_name)
        return ret

    def get_values(self, axis:str):
        """Gets a dictionary of values from the result of the FitResult's params_to_dict() function."""
        if 'params' not in self.fit_dict[axis].keys():
            if 'start_params' in self.fit_dict[axis].keys():
                return self.fit_dict[axis]['start_params']
            return []
        _dict = self.fit_dict[axis]['params']
        values = {}
        for k in _dict.keys():
            values[k] = _dict[k]['value']
        return values
    
    def indep_dims(self) -> int:
        indep, dep = self.data_dims(self.data_out)
        if isinstance(indep, list):
            return len(indep)
        if indep is not None:
            return 1
        return 0


class ValuePlot(PlotNode):
    def __init__(self, *args, **kwargs):
        self.xy_select = XYSelect()
        self._old_indep = []

        super().__init__(*args, **kwargs)

        self.layout = pn.Column(
            self.plot_options_panel,
            self.plot_panel,
        )

    def __panel__(self):
        return self.layout

    @pn.depends("data_out")
    def plot_options_panel(self):
        indep, dep = self.data_dims(self.data_out)

        opts = ["None"]
        if indep is not None:
            opts += indep
        self.xy_select.options = opts

        if indep != self._old_indep:
            if len(opts) > 2:
                self.xy_select.value = (opts[-2], opts[-1])
            elif len(opts) > 1:
                self.xy_select.value = (opts[-1], "None")
        self._old_indep = indep

        return self.xy_select

    @pn.depends("data_out", "xy_select.value", "refresh_graph")
    def plot_panel(self):
        self.refresh_graph = False

        plot = "*No valid options chosen.*"
        x, y = self.xy_select.value

        if x in ["None", None]:
            pass

        # case: a line or scatter plot (or multiple of these)
        elif y in ["None", None]:
            if isinstance(self.data_out, pd.DataFrame):
                plot = self.data_out.hvplot.line(
                    x=x, xlabel=self.dim_label(x),
                    y=self.get_data_fit_names(self.fit_axis_options()),
                ) * self.data_out.hvplot.scatter(x=x)

            elif isinstance(self.data_out, xr.Dataset):
                plot = self.data_out.hvplot.line(
                    x=x,
                    xlabel=self.dim_label(x),
                    y=self.get_data_fit_names(self.fit_axis_options()),
                ) * self.data_out.hvplot.scatter(x=x)
            else:
                raise NotImplementedError

        # case: if x and y are selected, we make a 2d plot of some sort
        else:
            if isinstance(self.data_out, pd.DataFrame):
                plot = plot_df_as_2d(self.data_out, x, y,
                                     dim_labels=self.dim_labels(),
                                     graph_axes=self.get_data_fit_names(self.fit_axis_options())
                                     )
            elif isinstance(self.data_out, xr.Dataset):
                plot = plot_xr_as_2d(self.data_out, x, y,
                                     dim_labels=self.dim_labels(),
                                     graph_axes=self.get_data_fit_names(self.fit_axis_options())
                                     )
            else:
                raise NotImplementedError
            plot = plot.cols(2)
        return plot

    def fit_axis_options(self):
        indep, dep = self.data_dims(self.data_out)
        ret = []
        for d in dep:
            if d[-4:] != "_fit" and d[-5:] != "_fit*":
                ret.append(d)
        return list(dep)
    

class ComplexHist(PlotNode):
    def __init__(self, *args, **kwargs):
        self.gb_select = pn.widgets.CheckButtonGroup(
            name="Group by",
            options=[],
            value=[],
        )
        super().__init__(*args, **kwargs)

        self.layout = pn.Column(
            labeled_widget(self.gb_select),
            self.plot_panel,
        )

    def __panel__(self):
        return self.layout

    @pn.depends("data_out", watch=True)
    def _sync_options(self):
        indep, dep = self.data_dims(self.data_out)
        if isinstance(indep, list):
            self.gb_select.options = indep

    @pn.depends("data_out", "gb_select.value", "refresh_graph")
    def plot_panel(self):
        self.refresh_graph = False

        t0 = time.perf_counter()

        plot = "*No valid options chosen.*"

        layout = pn.Column()
        for k, v in self.complex_dependents(self.data_out).items():
            xlim = float(self.data_out[v["real"]].min()), float(
                self.data_out[v["real"]].max()
            )
            ylim = float(self.data_out[v["imag"]].min()), float(
                self.data_out[v["imag"]].max()
            )
            p = self.data_out.hvplot(
                kind="hexbin",
                aspect=1,
                groupby=self.gb_select.value,
                x=v["real"],
                y=v["imag"],
                xlim=xlim,
                ylim=ylim,
                clabel="count",
            )
            layout.append(p)
            plot = layout

        return plot

    def get_plot(self):
        plt = self.plot_panel()
        return plt[0].object

    def fit_axis_options(self):
        _dict = self.complex_dependents(self.data_out).items()
        if not isinstance(_dict, dict):
            _dict = dict(_dict)
        return list(_dict.keys())


class MagnitudePhasePlot(PlotNode):
    def __init__(self, *args, **kwargs):
        self.xy_select = XYSelect()
        self._old_indep = []

        super().__init__(*args, **kwargs)

        self.layout = pn.Column(
            self.plot_options_panel,
            self.plot_panel,
        )

    def process(self):
        assert isinstance(
            self.data_in, xr.Dataset), "MagnitudePhasePlot needs an xr.Dataset, did not receive one."
        # Convert the current dataset to one that has Magnitude and Phase columns
        indep, dep = self.data_dims(self.data_in)
        # Assign labels. This assumes the first column is the real coefficients.
        keylist = list(self.data_in.data_vars.keys())
        real = self.data_in.variables[keylist[0]]
        imaginary = self.data_in.variables[keylist[1]]
        # Calculate magnitude and phase
        magnitude = np.sqrt(np.square(real) + np.square(imaginary))
        phase = np.arctan(imaginary / real)
        super().process()
        # Add magnitude and phase data (don't effect self.data_in)
        self.data_out['Magnitude'] = (indep, magnitude)
        self.data_out['Phase'] = (indep, phase)

    def __panel__(self):
        return self.layout

    @pn.depends("data_out")
    def plot_options_panel(self):
        indep, dep = self.data_dims(self.data_out)

        opts = ["None"]
        if indep is not None:
            opts += indep
        self.xy_select.options = opts

        if indep != self._old_indep:
            if len(opts) > 2:
                self.xy_select.value = (opts[-2], opts[-1])
            elif len(opts) > 1:
                self.xy_select.value = (opts[-1], "None")
        self._old_indep = indep

        return self.xy_select

    @pn.depends("data_out", "xy_select.value", "refresh_graph")
    def plot_panel(self):
        self.refresh_graph = False

        plot = "*No valid options chosen.*"
        x, y = self.xy_select.value

        if x in ["None", None]:
            pass

        else:
            # case: a line or scatter plot (or multiple of these)
            # NOTE: if these have shared_axes=True, then both graphs cannot
            # simultaneously have fits without one graph being unreadable due to
            # it's axis dimensions.
            if y in ["None", None]:
                plot_m = self.data_out.hvplot.line(
                    x=x,
                    xlabel=self.dim_label(x),
                    y=self.get_data_fit_names("Magnitude"),
                    shared_axes=False,
                )
                plot_p = self.data_out.hvplot.line(
                    x=x,
                    xlabel=self.dim_label(x),
                    y=self.get_data_fit_names("Phase"),
                    shared_axes=False,
                )
                plot = pn.Column(plot_m, plot_p)

            # case: if x and y are selected, we make a 2d plot of some sort
            else:
                plot = plot_xr_as_2d(self.data_out, x, y,
                                     dim_labels=self.dim_labels(), 
                                     graph_axes=self.get_data_fit_names(self.fit_axis_options()))
                plot = plot.cols(2)

        return plot

    def fit_axis_options(self):
        return ['Magnitude', 'Phase']

    def get_data_fit_names(self, axis_name):
        return super().get_data_fit_names(axis_name, [])


def plot_df_as_2d(df, x, y, dim_labels=None, graph_axes=None):
    if graph_axes is None:
        graph_axes = []
    if dim_labels is None:
        dim_labels = {}

    indeps, deps = Node.data_dims(df)

    # Set deps to the passed axes so it graphs all desired data
    if graph_axes != []:
        deps = graph_axes

    if x in indeps and y in indeps:
        return pn.Column(
            *[
                df.hvplot.heatmap(
                    x=x,
                    y=y,
                    C=d,
                    xlabel=dim_labels.get(x, x),
                    ylabel=dim_labels.get(y, y),
                    clabel=f"Mean {dim_labels.get(d, d)}",
                ).aggregate(function=np.mean)
                for d in deps
            ]
        )
    elif x in deps + indeps and y in deps:
        return df.hvplot.scatter(
            x=x,
            y=y,
            xlabel=dim_labels.get(x, x),
            ylabel=dim_labels.get(y, y),
        )
    else:
        return "*that's currently not supported :(*"


def plot_xr_as_2d(ds, x, y, dim_labels=None, graph_axes=None):
    if graph_axes is None:
        graph_axes = []

    if dim_labels is None:
        dim_labels = {}

    if ds is None:
        return "Nothing to plot."

    indeps, deps = Node.data_dims(ds)
    plot = None

    # Set deps to the passed axes so it graphs all desired data
    if graph_axes:
        deps = graph_axes

    if x + '_fit' in ds:
        return plot_ds_2d_with_fit(ds, dim_labels.get(x, x), x, y)
    # plotting stuff vs two independent -- heatmap
    if x in indeps and y in indeps:
        for d in deps:
            if plot is None:
                plot = ds.get(d).hvplot.quadmesh(
                    x=x,
                    y=y,
                    xlabel=dim_labels.get(x, x),
                    ylabel=dim_labels.get(y, y),
                    clabel=f"Mean {dim_labels.get(d, d)}",
                )
            else:
                plot += ds.get(d).hvplot.quadmesh(
                    x=x,
                    y=y,
                    xlabel=dim_labels.get(x, x),
                    ylabel=dim_labels.get(y, y),
                    clabel=f"Mean {dim_labels.get(d, d)}",
                )
        # FIXME: QuadMesh object has no attribute 'cols' error for longsweep
        try:
            return plot.cols(1)
        except AttributeError:
            return "*Not a valid plot* Attribute Error occurred"

    else:
        return "*Not a valid plot*"


# -- specific plot functions


# -- various tool functions


def labeled_widget(w, lbl=None):
    m = w.margin

    if lbl is None:
        lbl = w.name

    lbl_w = pn.widgets.StaticText(value=lbl, margin=(m[0], m[1], 0, m[1]))
    w.margin = (0, m[1], m[0], m[1])
    return pn.Column(
        lbl_w,
        w,
    )


# -- convenience functions


def plot_data(data: Union[pd.DataFrame, xr.Dataset]) -> pn.viewable.Viewable:
    n = Node(data, name="plot")
    return pn.Column(
        n,
        n.plot,
    )