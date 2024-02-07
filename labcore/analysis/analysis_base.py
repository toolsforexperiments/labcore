from typing import Optional, Type, Any, Dict
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

from ..data.datadict_storage import NumpyEncoder, timestamp_from_path
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
            for n in name.split("/"):
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


    def to_table(self, name, data: Dict[str, Any]):
        data.update(
            {'data_loc': self.datafolder.name,
             'datetime': timestamp_from_path(self.datafolder),}
        )
        
        def make_table(data):
            row = {k: [v] for k, v in data.items()}
            index = row.pop('data_loc')
            return pd.DataFrame(row, index=index)
        
        def append_to_table(df, data, must_match=False):
            row = make_table(data)
            if must_match:
                if not np.all(row.columns == df.columns):
                    raise ValueError(f"existing table columns ({df.columns}) do not match"
                                    f"data columns ({row.keys()})")
            
            if row.index[0] in df.index:
                df.loc[row.index[0]] = row.loc[row.index[0]]
            else:
                df = pd.concat([df, row], axis=0)
            return df


        path = self.savefolders[0].parent / (name+'.csv')
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
        df['datetime'] = pd.to_datetime(df['datetime']) 
        return df


    # --- Saving analysis results --- #
    def save(self):
        for folder in self.savefolders:
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)

            for name, element in self.entities.items():
                try: 
                    if isinstance(element, Figure):
                        fp = self.save_mpl_figure(element, name, folder)

                    elif isinstance(element, AnalysisResult):
                        fp = self.save_dict_data(
                            element.params_to_dict(), name + "_params", folder
                        )
                        if isinstance(element, FitResult):
                            fp = self.save_str(
                                element.lmfit_result.fit_report(),
                                name + "_lmfit_report",
                                folder,
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
                            logger.error(f'Could not pickle {name}.')
                except:
                    logger.warning(
                        f"data '{name}', type {type(element)}"
                        f"could not be saved regularly, try pickle instead..."
                    )
                    try:
                        n = name + f"_{str(type(element))}"
                        fp = self.save_pickle(element, n, folder)
                    except:
                        logger.error(f'Could not pickle {name}.')


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

    def save_dict_data(self, data: dict, name: str, folder: Path):
        fp = self._new_file_path(folder, name, "json")
        # d = dict_arrays_to_list(data)
        with open(fp, "x") as f:
            json.dump(data, f, cls=NumpyEncoder)
        return fp

    def save_str(self, data: str, name: str, folder: Path):
        fp = self._new_file_path(folder, name, "txt")
        with open(fp, "x") as f:
            f.write(data)
        return fp

    def save_np(self, data: np.ndarray, name: str, folder: Path):
        fp = self._new_file_path(folder, name, "json")
        with open(fp, "x") as f:
            json.dump({name: data}, f, cls=NumpyEncoder)

    def save_ds(self, data: xr.Dataset, name: str, folder: Path):
        fp = self._new_file_path(folder, name, "nc")
        data.to_netcdf(fp)

    def save_da(self, data: xr.DataArray, name: str, folder: Path):
        fp = self._new_file_path(folder, name, "nc")
        data.to_netcdf(fp)

    def save_df(self, data: pd.DataFrame, name: str, folder: Path):
        fp = self._new_file_path(folder, name, "csv")
        data.to_csv(fp)

    def save_pickle(self, data: Any, name: str, folder: Path):
        fp = self._new_file_path(folder, name, "pickle")
        with open(fp, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
