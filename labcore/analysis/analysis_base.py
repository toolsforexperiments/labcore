from typing import Optional, Type, Any, Dict, List
from types import TracebackType
from pathlib import Path
from datetime import datetime
import json
import logging
import pickle

import numpy as np
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import xarray as xr
import pandas as pd

# Needed to generate hvplot from a script
import hvplot.xarray
import holoviews as hv

from ..data.datadict_storage import NumpyEncoder, timestamp_from_path
from .fit import AnalysisResult, FitResult


logger = logging.getLogger(__name__)


class AnalysisExistsError(Exception):
    pass


class DatasetAnalysis:

    figure_save_format = ["png", "pdf"]
    raise_on_earlier_analysis = None

    def __init__(
        self, datafolder, name, analysisfolder="./analysis/", has_period_in_name=False,
        raise_on_earlier_analysis=None,
    ):
        if raise_on_earlier_analysis is not None:
            self.raise_on_earlier_analysis = raise_on_earlier_analysis

        self.name = name
        # The folder that contains the data we are performing an analysis
        self.datafolder = datafolder
        if not isinstance(self.datafolder, Path):
            self.datafolder = Path(self.datafolder)
        self.analysisfolder = analysisfolder
        if not isinstance(self.analysisfolder, Path):
            self.analysisfolder = Path(self.analysisfolder)
        self.timestamp = str(
            datetime.now().replace(microsecond=0).isoformat().replace(":", "")
        )

        # saving redundantly with data, and a separate analysis folder.
        self.savefolders = []
        for i, f in enumerate([self.analysisfolder, self.datafolder]):
            for n in name.split("/"):
                f = f / n
            if not i:
                if not has_period_in_name:
                    f = f / self.datafolder.stem
                else:
                    end = self.datafolder.suffix
                    f = f / (self.datafolder.stem + end)

            self.savefolders.append(f)

        self.entities = {}
        self.files = []

    def __enter__(self):
        earlier_exist = False
        if self.raise_on_earlier_analysis is not None:
            earlier_exist = True
            for filename, suffix in self.raise_on_earlier_analysis:
                if not self.has_analysis_data(filename, suffix):
                    earlier_exist = False
            if earlier_exist:
                raise AnalysisExistsError

        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.save()

    def _new_file_path(self, folder: Path, name: str, suffix: str = "") -> Path:
        if suffix != "":
            name = name + "." + suffix
        return folder / f"{self.timestamp}_{name}"

    # --- loading measurement data --- #
    def load_metadata_from_json(
        self,
        file_name,
        key,
    ):
        """Load a parameter from a metadata json file

        Parameters
        ----------
        file_name
            file name inside the data directory.
        key
            which entry from the file we should get.
            only top-level keys supported currently.

        Returns
        -------
            the requested data.

        Raises
        ------
        ValueError
            if requested data is not present in the file.
        """
        fn = self.datafolder / file_name
        with open(fn, "r") as f:
            data = json.load(f)

        if key not in data:
            raise ValueError("this parameter was not found in the saved meta data.")

        return data[key]

    def load_saved_parameter(
        self,
        parameter_name,
        parameter_manager_name="parameter_manager",
        file_name="parameters.json",
    ):

        fn = self.datafolder / file_name
        with open(fn, "r") as f:
            data = json.load(f)

        parameter_path = f"{parameter_manager_name}.{parameter_name}"
        if parameter_path not in data:
            raise ValueError("this parameter was not found in the saved meta data.")

        return data[parameter_path]["value"]

    # --- Adding analysis results --- #
    def add(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self.entities:
                raise ValueError(
                    "element with that name already exists in this analysis."
                )
            self.entities[k] = v

    def add_figure(self, name, *arg, fig: Optional[Figure] = None, **kwargs) -> Figure:
        if name in self.entities:
            raise ValueError("element with that name already exists in this analysis.")
        if fig is None:
            fig = plt.figure(*arg, **kwargs)
        self.entities[name] = fig
        return fig

    make_figure = add_figure

    def to_table(self, name, data: Dict[str, Any]):
        data.update(
            {
                "data_loc": self.datafolder.name,
                "datetime": timestamp_from_path(self.datafolder),
            }
        )

        def make_table(data):
            row = {k: [v] for k, v in data.items()}
            index = row.pop("data_loc")
            return pd.DataFrame(row, index=index)

        def append_to_table(df, data, must_match=False):
            row = make_table(data)
            if must_match:
                if not np.all(row.columns == df.columns):
                    raise ValueError(
                        f"existing table columns ({df.columns}) do not match"
                        f"data columns ({row.keys()})"
                    )

            if row.index[0] in df.index:
                df.loc[row.index[0]] = row.loc[row.index[0]]
            else:
                df = pd.concat([df, row], axis=0)
            return df

        path = self.savefolders[0].parent / (name + ".csv")
        if not path.parent.exists():
            path.parent.mkdir(exist_ok=True, parents=True)

        if path.exists():
            df = pd.read_csv(path, index_col=0)
            df = append_to_table(df, data)
        else:
            df = make_table(data)

        df.to_csv(path)

    @staticmethod
    def load_table(path):
        df = pd.read_csv(path, index_col=0)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    # --- Saving analysis results --- #
    def save(self):
        for folder in self.savefolders:
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)

            for name, element in self.entities.items():
                fp = None
                try:
                    if isinstance(element, Figure):
                        fp = self.save_mpl_figure(element, name, folder)

                    elif isinstance(element, AnalysisResult):
                        fp = [
                            self.save_dict_data(
                                element.params_to_dict(), name + "_params", folder
                            )
                        ]
                        if isinstance(element, FitResult):
                            fp.append(
                                self.save_str(
                                    element.lmfit_result.fit_report(),
                                    name + "_lmfit_report",
                                    folder,
                                )
                            )

                    elif isinstance(element, xr.Dataset):
                        fp = self.save_ds(
                            element,
                            name + "_xrdataset",
                            folder,
                        )

                    elif isinstance(element, xr.DataArray):
                        fp = self.save_da(
                            element,
                            name + "_xrdataarray",
                            folder,
                        )

                    elif isinstance(element, hv.core.Dimensioned):
                        fp = self.save_hv_plot(
                            element,
                            name + "_hvplot",
                            folder,
                        )

                    elif isinstance(element, pd.DataFrame):
                        fp = self.save_df(element, name, folder)

                    elif isinstance(element, np.ndarray):
                        fp = self.save_np(element, name, folder)

                    elif isinstance(element, dict):
                        fp = self.save_dict_data(element, name, folder)

                    elif isinstance(element, str):
                        fp = self.save_str(element, name, folder)

                    else:
                        logger.warning(
                            f"additional data '{name}', type {type(element)}"
                            f"is not supported for saving, pickle instead."
                        )
                        try:
                            n = name + f"_{str(type(element))}"
                            fp = self.save_pickle(element, n, folder)
                        except:
                            logger.error(f"Could not pickle {name}.")
                except:
                    logger.warning(
                        f"data '{name}', type {type(element)}"
                        f"could not be saved regularly, try pickle instead..."
                    )
                    try:
                        n = name + f"_{str(type(element))}"
                        fp = self.save_pickle(element, n, folder)
                    except:
                        logger.error(f"Could not pickle {name}.")

                if fp is not None:
                    if isinstance(fp, list):
                        self.files.extend(set(fp) - set(self.files))
                    elif fp not in self.files:
                        self.files.append(fp)

    def save_mpl_figure(self, fig: Figure, name: str, folder: Path) -> List[Path]:
        """save a figure in a standard way to the dataset directory.

        Parameters
        ----------
        fig
            the figure instance
        name
            name to give the figure
        fmt
            file format (defaults to png)

        Returns
        -------
        A list of file paths where the figure was saved.

        """
        fmts = self.figure_save_format
        if not isinstance(fmts, list):
            fmts = [fmts]

        fig.suptitle(f"{self.datafolder.name}: {name}", fontsize="small")

        fps = []
        for f in fmts:
            fp = self._new_file_path(folder, name, f)
            fig.savefig(fp)
            fps.append(fp)

        return fps

    def save_hv_plot(self, plot: hv.core.Dimensioned, name: str, folder: Path) -> Path:
        """save a hvplot in a standard way to the dataset directory.

        Parameters
        ----------
        plot
            the hvplot instance
        name
            name to give the figure
        fmt
            file format (defaults to png)

        Returns
        -------
            Path of the saved file.

        """
        fp = self._new_file_path(folder, name, "html")
        hv.save(plot, fp)
        return fp

    def save_dict_data(self, data: dict, name: str, folder: Path) -> Path:
        fp = self._new_file_path(folder, name, "json")
        # d = dict_arrays_to_list(data)
        # with open(fp, "x") as f:
        with open(fp, "w") as f:
            json.dump(data, f, cls=NumpyEncoder)
        return fp

    def save_str(self, data: str, name: str, folder: Path) -> Path:
        fp = self._new_file_path(folder, name, "txt")
        with open(fp, "x") as f:
            f.write(data)
        return fp

    def save_np(self, data: np.ndarray, name: str, folder: Path) -> Path:
        fp = self._new_file_path(folder, name, "json")
        with open(fp, "x") as f:
            json.dump({name: data}, f, cls=NumpyEncoder)
        return fp

    def save_ds(self, data: xr.Dataset, name: str, folder: Path) -> Path:
        fp = self._new_file_path(folder, name, "nc")
        data.to_netcdf(fp)
        return fp

    def save_da(self, data: xr.DataArray, name: str, folder: Path) -> Path:
        fp = self._new_file_path(folder, name, "nc")
        data.to_netcdf(fp)
        return fp

    def save_df(self, data: pd.DataFrame, name: str, folder: Path) -> Path:
        fp = self._new_file_path(folder, name, "csv")
        data.to_csv(fp)
        return fp

    def save_pickle(self, data: Any, name: str, folder: Path) -> Path:
        fp = self._new_file_path(folder, name, "pickle")
        with open(fp, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        return fp

    # --- loading (and managing) earlier analysis results --- #
    def get_analysis_data_file(self, name: str, format=["json"]):
        files = list(self.savefolders[0].glob(f"*{name}*"))
        files = [f for f in files if f.suffix[1:] in format]
        if len(files) == 0:
            raise ValueError(f"no analysis data found for '{name}'")
        return files[-1]

    def has_analysis_data(self, name: str, format=["json"]):
        try:
            self.get_analysis_data_file(name, format)
            return True
        except ValueError:
            return False

    def load_analysis_data(self, name: str, format=["json"]):
        fp = self.get_analysis_data_file(name, format)
        with open(fp, "r") as f:
            data = json.load(f)
        return data
