from typing import Optional, Type, Any
from types import TracebackType
from pathlib import Path
from datetime import datetime
import json
import logging

import numpy as np
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

from ..data.datadict_storage import NumpyEncoder
from .fit import AnalysisResult, FitResult


logger = logging.getLogger(__name__)


class DatasetAnalysis:

    figure_save_format = ["png", "pdf"]

    def __init__(self, datafolder, name, analysisfolder="./analysis/"):
        self.name = name
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
            for n in name.split('/'):
                f = f / n
            if not i:
                f = f / self.datafolder.stem
            self.savefolders.append(f)

        self.entities = {}

    def __enter__(self):
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

    # --- Saving analysis results --- #
    def save(self):
        for f in self.savefolders:
            if not f.exists():
                f.mkdir(parents=True, exist_ok=True)

            for name, element in self.entities.items():
                if isinstance(element, Figure):
                    fp = self.save_mpl_figure(element, name, f)

            # elif isinstance(element, AnalysisResult):
            #     fp = self.save_add_dict_data(element.params_to_dict(), name + "_params")
            #     if isinstance(element, FitResult):
            #         fp = self.save_add_str(
            #             element.lmfit_result.fit_report(), name + "_lmfit_report"
            #         )

            # elif isinstance(element, np.ndarray):
            #     fp = self.save_add_np(element, name)

            # elif isinstance(element, dict):
            #     fp = self.save_add_dict_data(element, name)

            # elif isinstance(element, str):
            #     fp = self.save_add_str(element, name)

                else:
                    logger.error(f"additional data '{name}', type {type(element)}"
                                f"is not supported for saving, ignore.")

    def save_mpl_figure(self, fig: Figure, name: str, folder: Path):
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
        ``None``

        """
        fmts = self.figure_save_format
        if not isinstance(fmts, list):
            fmts = [fmts]

        fig.suptitle(f"{self.datafolder.name}: {name}", fontsize="small")

        for f in fmts:
            fp = self._new_file_path(folder, name, f)
            fig.savefig(fp)

        return fp

    # def save_add_dict_data(self, data: dict, name: str):
    #     fp = self._new_file_path(name, "json")
    #     # d = dict_arrays_to_list(data)
    #     with open(fp, "x") as f:
    #         json.dump(data, f, cls=NumpyEncoder)
    #     return fp

    # def save_add_str(self, data: str, name: str):
    #     fp = self._new_file_path(name, "txt")
    #     with open(fp, "x") as f:
    #         f.write(data)
    #     return fp

    # def save_add_np(self, data: np.ndarray, name: str):
    #     fp = self._new_file_path(name, "json")
    #     with open(fp, "x") as f:
    #         json.dump({name: data}, f, cls=NumpyEncoder)

    # --- loading (and managing) earlier analysis results --- #
    # def get_analysis_data_file(self, name: str, format=["json"]):
    #     files = list(self.folder.glob(f"*{name}*"))
    #     files = [f for f in files if f.suffix[1:] in format]
    #     if len(files) == 0:
    #         raise ValueError(f"no analysis data found for '{name}'")
    #     return files[-1]

    # def has_analysis_data(self, name: str, format=["json"]):
    #     try:
    #         self.get_analysis_data_file(name, format)
    #         return True
    #     except ValueError:
    #         return False

    # def load_analysis_data(self, name: str, format=["json"]):
    #     fp = self.get_analysis_data_file(name, format)
    #     with open(fp, "r") as f:
    #         data = json.load(f)
    #     return data
