"""plottr.data.datadict_storage

Provides file-storage tools for the DataDict class.

.. note::
    Any function in this module that interacts with a ddh5 file, will create a lock file while it is using the file.
    The lock file has the following format: ~<file_name>.lock. The file lock will get deleted even if the program
    crashes. If the process is suddenly stopped however, we cannot guarantee that the file lock will be deleted.
"""
import os
import logging
import time
import datetime
import uuid
import json
import shutil
import re
from enum import Enum
from typing import Any, Union, Optional, Dict, Type, Collection, List
from types import TracebackType
from pathlib import Path

import numpy as np
import h5py
import xarray as xr

from .tools import split_complex
from .datadict import (
    DataDict,
    is_meta_key,
    DataDictBase,
    dd2xr,
    datadict_to_meshgrid,
    dd2df,
    datasets_are_equal,
)

__author__ = "Wolfgang Pfaff"
__license__ = "MIT"

DATAFILEXT = "ddh5"
TIMESTRFORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger(__name__)

# FIXME: need correct handling of dtypes and list/array conversion

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


class AppendMode(Enum):
    """How/Whether to append data to existing data."""

    #: Data that is additional compared to already existing data is appended.
    new = 0
    #: All data is appended to existing data.
    all = 1
    #: Data is overwritten.
    none = 2


# tools for working on hdf5 objects


def h5ify(obj: Any) -> Any:
    """
    Convert an object into something that we can assign to an HDF5 attribute.

    Performs the following conversions:
    - list/array of strings -> numpy chararray of unicode type

    :param obj: Input object.
    :return: Object, converted if necessary.
    """
    if isinstance(obj, list):
        all_string = True
        for elt in obj:
            if not isinstance(elt, str):
                all_string = False
                break
        if not all_string:
            obj = np.array(obj)

    if type(obj) == np.ndarray and obj.dtype.kind == "U":
        return np.char.encode(obj, encoding="utf8")

    return obj


def deh5ify(obj: Any) -> Any:
    """Convert slightly mangled types back to more handy ones.

    :param obj: Input object.
    :return: Object
    """
    if type(obj) == bytes:
        return obj.decode()

    if type(obj) == np.ndarray and obj.dtype.kind == "S":
        return np.char.decode(obj)

    return obj


def set_attr(h5obj: Any, name: str, val: Any) -> None:
    """Set attribute `name` of object `h5obj` to `val`

    Use :func:`h5ify` to convert the object, then try to set the attribute
    to the returned value. If that does not succeed due to a HDF5 typing
    restriction, set the attribute to the string representation of the value.
    """
    try:
        h5obj.attrs[name] = h5ify(val)
    except TypeError:
        newval = str(val)
        h5obj.attrs[name] = h5ify(newval)


def add_cur_time_attr(
    h5obj: Any, name: str = "creation", prefix: str = "__", suffix: str = "__"
) -> None:
    """Add current time information to the given HDF5 object, following the format of:
    ``<prefix><name>_time_sec<suffix>``.

    :param h5obj: The HDF5 object.
    :param name: The name of the attribute.
    :param prefix: Prefix of the attribute.
    :param suffix: Suffix of the attribute.
    """

    t = time.localtime()
    tsec = time.mktime(t)
    tstr = time.strftime(TIMESTRFORMAT, t)

    set_attr(h5obj, prefix + name + "_time_sec" + suffix, tsec)
    set_attr(h5obj, prefix + name + "_time_str" + suffix, tstr)


# elementary reading/writing


def _data_file_path(file: Union[str, Path], init_directory: bool = False) -> Path:
    """Get the full filepath of the data file.
    If `init_directory` is True, then create the parent directory."""

    path = Path(file)

    if path.suffix != f".{DATAFILEXT}":
        path = Path(path.parent, path.stem + f".{DATAFILEXT}")
    if init_directory:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def datadict_to_hdf5(
    datadict: DataDict,
    path: Union[str, Path],
    groupname: str = "data",
    append_mode: AppendMode = AppendMode.new,
    file_timeout: Optional[float] = None,
) -> None:
    """Write a DataDict to DDH5

    Note: Meta data is only written during initial writing of the dataset.
    If we're appending to existing datasets, we're not setting meta
    data anymore.

    :param datadict: Datadict to write to disk.
    :param path: Path of the file (extension may be omitted).
    :param groupname: Name of the top level group to store the data in.
    :param append_mode:
        - `AppendMode.none` : Delete and re-create group.
        - `AppendMode.new` : Append rows in the datadict that exceed
          the number of existing rows in the dataset already stored.
          Note: we're not checking for content, only length!

        - `AppendMode.all` : Append all data in datadict to file data sets.
    :param file_timeout: How long the function will wait for the ddh5 file to unlock. Only relevant if you are
        writing to a file that already exists and some other program is trying to read it at the same time.
        If none uses the default value from the :class:`FileOpener`.

    """
    filepath = _data_file_path(path, True)
    if not filepath.exists():
        append_mode = AppendMode.none

    with FileOpener(filepath, "a", file_timeout) as f:
        if append_mode is AppendMode.none:
            init_file(f, groupname)
        assert groupname in f
        grp = f[groupname]

        # add top-level meta data.
        for k, v in datadict.meta_items(clean_keys=False):
            set_attr(grp, k, v)

        for k, v in datadict.data_items():
            data = v["values"]
            shp = data.shape
            nrows = shp[0]

            # create new dataset, add axes and unit metadata
            if k not in grp:
                maxshp = tuple([None] + list(shp[1:]))
                ds = grp.create_dataset(k, maxshape=maxshp, data=data)

                # add meta data
                add_cur_time_attr(ds)

                if v.get("axes", []):
                    set_attr(ds, "axes", v["axes"])
                if v.get("unit", "") != "":
                    set_attr(ds, "unit", v["unit"])

                for kk, vv in datadict.meta_items(k, clean_keys=False):
                    set_attr(ds, kk, vv)
                ds.flush()

            # if the dataset already exits, append data according to
            # chosen append mode.
            else:
                ds = grp[k]
                dslen = ds.shape[0]

                if append_mode == AppendMode.new:
                    newshp = tuple([nrows] + list(shp[1:]))
                    ds.resize(newshp)
                    ds[dslen:] = data[dslen:]
                elif append_mode == AppendMode.all:
                    newshp = tuple([dslen + nrows] + list(shp[1:]))
                    ds.resize(newshp)
                    ds[dslen:] = data[:]
                ds.flush()


def init_file(f: h5py.File, groupname: str = "data") -> None:
    if groupname in f:
        del f[groupname]
        f.flush()
        grp = f.create_group(groupname)
        add_cur_time_attr(grp)
        f.flush()
    else:
        grp = f.create_group(groupname)
        add_cur_time_attr(grp)
        f.flush()


def datadict_from_hdf5(
    path: Union[str, Path],
    groupname: str = "data",
    startidx: Union[int, None] = None,
    stopidx: Union[int, None] = None,
    structure_only: bool = False,
    ignore_unequal_lengths: bool = True,
    file_timeout: Optional[float] = None,
) -> DataDict:
    """Load a DataDict from file.

    :param path: Full filepath without the file extension.
    :param groupname: Name of hdf5 group.
    :param startidx: Start row.
    :param stopidx: End row + 1.
    :param structure_only: If `True`, don't load the data values.
    :param ignore_unequal_lengths: If `True`, don't fail when the rows have
        unequal length; will return the longest consistent DataDict possible.
    :param file_timeout: How long the function will wait for the ddh5 file to unlock. If none uses the default
        value from the :class:`FileOpener`.
    :return: Validated DataDict.
    """
    filepath = _data_file_path(path)
    if not filepath.exists():
        raise ValueError(f"Specified file '{filepath}' does not exist.")

    if startidx is None:
        startidx = 0

    res = {}
    with FileOpener(filepath, "r", file_timeout) as f:
        if groupname not in f:
            raise ValueError("Group does not exist.")

        grp = f[groupname]
        keys = list(grp.keys())
        lens = [len(grp[k][:]) for k in keys]

        if len(set(lens)) > 1:
            if not ignore_unequal_lengths:
                raise RuntimeError("Unequal lengths in the datasets.")

            if stopidx is None or stopidx > min(lens):
                stopidx = min(lens)
        else:
            if stopidx is None or stopidx > lens[0]:
                stopidx = lens[0]

        for attr in grp.attrs:
            if is_meta_key(attr):
                res[attr] = deh5ify(grp.attrs[attr])

        for k in keys:
            ds = grp[k]
            entry: Dict[str, Union[Collection[Any], np.ndarray]] = dict(
                values=np.array([]),
            )

            if "axes" in ds.attrs:
                entry["axes"] = deh5ify(ds.attrs["axes"]).tolist()
            else:
                entry["axes"] = []

            if "unit" in ds.attrs:
                entry["unit"] = deh5ify(ds.attrs["unit"])

            if not structure_only:
                entry["values"] = ds[startidx:stopidx]

            entry["__shape__"] = ds[:].shape

            # and now the meta data
            for attr in ds.attrs:
                if is_meta_key(attr):
                    _val = deh5ify(ds.attrs[attr])
                    entry[attr] = deh5ify(ds.attrs[attr])

            res[k] = entry

    dd = DataDict(**res)
    dd.validate()
    return dd


def all_datadicts_from_hdf5(
    path: Union[str, Path], file_timeout: Optional[float] = None, **kwargs: Any
) -> Dict[str, Any]:
    """
    Loads all the DataDicts contained on a single HDF5 file. Returns a dictionary with the group names as keys and
    the DataDicts as the values of that key.

    :param path: The path of the HDF5 file.
    :param file_timeout: How long the function will wait for the ddh5 file to unlock. If none uses the default
        value from the :class:`FileOpener`.
    :return: Dictionary with group names as key, and the DataDicts inside them as values.
    """
    filepath = _data_file_path(path)
    if not os.path.exists(filepath):
        raise ValueError("Specified file does not exist.")

    ret = {}
    with FileOpener(filepath, "r", file_timeout) as f:
        keys = [k for k in f.keys()]
    for k in keys:
        ret[k] = datadict_from_hdf5(
            path=path, groupname=k, file_timeout=file_timeout, **kwargs
        )
    return ret


def reconstruct_safe_write_data(path: Union[str, Path],
                          unification_from_scratch: bool = True,
                          file_timeout: Optional[float] = None) -> DataDictBase:
    """
    Creates a new DataDict from the data saved in the .tmp folder. This is used when the data is saved in the safe
    writing mode. The data is saved in individual files in the .tmp folder. This function reconstructs the data from
    these files and returns a DataDict with the data.

    :param path: The path to the folder containing the .tmp path
    :param unification_from_scratch: If True, will reconstruct the data from scratch. If False, will try to load the
        data from the last reconstructed file.
    :param file_timeout: How long the function will wait for the ddh5 file to unlock. If none uses the default value
    """

    path = Path(path)

    tmp_path = path.parent / ".tmp"

    # FIXME: This should probably raise a warning more than a crash, but will leave it as crash for now
    if not tmp_path.exists():
        raise ValueError("Specified folder does not exist.")

    files = []
    for dirpath, dirnames, filenames in os.walk(str(tmp_path)):
        files.extend([(Path(dirpath)/file) for file in filenames if file.endswith(".ddh5")])

    files = sorted(files, key=lambda x: int(x.stem.split("#")[-1]))

    # Checks if data is already there.
    # If there is, loads it from the latest loaded file not to have to load unnecessary data
    if path.exists() and not unification_from_scratch:
        dd = datadict_from_hdf5(path, file_timeout=file_timeout)
        if not dd.has_meta("last_reconstructed_file"):
            raise ValueError("The file does not have the meta data 'last_reconstructed_file', "
                             "could not know where to reconstruct from.")
        last_reconstructed_file = Path(dd.meta_val("last_reconstructed_file"))
        if not last_reconstructed_file.exists() or last_reconstructed_file not in files:
            raise ValueError("When reconstructing the data, could find the last reconstructed file. "
                             "This indicates that something wrong happened in the tmp folder.")
        starting_index = files.index(last_reconstructed_file) + 1
    else:
        first = files.pop(0)
        dd = datadict_from_hdf5(first, file_timeout=file_timeout)
        starting_index = 0

    for file in files[starting_index:]:
        d = datadict_from_hdf5(file, file_timeout=file_timeout)
        # Create a dictionary with just the keys and values to add to the original one.
        dd.add_data(**{x[0]: d.data_vals(x[0]) for x in d.data_items()})

    # Add shape to axes
    for name, datavals in dd.data_items():
        datavals["__shape__"] = tuple(np.array(datavals["values"][:]).shape,)

    # Catches the edge case where there is a single file in the .tmp folder.
    # This will not happen other than the first time, so it is ok to have that first variable there.
    if len(files) > 0:
        dd.add_meta("last_reconstructed_file", str(files[-1]))
    else:
        dd.add_meta("last_reconstructed_file", str(first))

    return dd

# File access with locking


class FileOpener:
    """
    Context manager for opening files, creates its own file lock to indicate other programs that the file is being
    used. The lock file follows the following structure: "~<file_name>.lock".

    :param path: The file path.
    :param mode: The opening file mode. Only the following modes are supported: 'r', 'w', 'w-', 'a'. Defaults to 'r'.
    :param timeout: Time, in seconds, the context manager waits for the file to unlock. Defaults to 30.
    :param test_delay: Length of time in between checks. I.e. how long the FileOpener waits to see if a file got
        unlocked again
    """

    def __init__(
        self,
        path: Union[Path, str],
        mode: str = "r",
        timeout: Optional[float] = None,
        test_delay: float = 0.1,
    ):
        self.path = Path(path)
        self.lock_path = self.path.parent.joinpath("~" + str(self.path.stem) + ".lock")
        if mode not in ["r", "w", "w-", "a"]:
            raise ValueError("Only 'r', 'w', 'w-', 'a' modes are supported.")
        self.mode = mode
        self.default_timeout = 300.0
        if timeout is None:
            self.timeout = self.default_timeout
        else:
            self.timeout = timeout
        self.test_delay = test_delay

        self.file: Optional[h5py.File] = None

    def __enter__(self) -> h5py.File:
        self.file = self.open_when_unlocked()
        return self.file

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ) -> None:
        try:
            assert self.file is not None
            self.file.close()
        finally:
            if self.lock_path.is_file():
                self.lock_path.unlink()

    def open_when_unlocked(self) -> h5py.File:
        t0 = time.time()
        while True:
            if not self.lock_path.is_file():
                try:
                    self.lock_path.touch(exist_ok=False)
                # This happens if some other process beat this one and created the file beforehand
                except FileExistsError:
                    continue

                while True:
                    try:
                        f = h5py.File(str(self.path), self.mode)
                        return f
                    except (OSError, PermissionError, RuntimeError):
                        pass
                    time.sleep(
                        self.test_delay
                    )  # don't overwhelm the FS by very fast repeated calls.
                    if time.time() - t0 > self.timeout:
                        raise RuntimeError("Waiting or file unlock timeout")

            time.sleep(
                self.test_delay
            )  # don't overwhelm the FS by very fast repeated calls.
            if time.time() - t0 > self.timeout:
                raise RuntimeError("Lock file remained for longer than timeout time")


class DDH5Writer(object):
    """Context manager for writing data to DDH5.
    Based on typical needs in taking data in an experimental physics lab.

    Creates lock file when writing data.

    Can be used in safe_write_mode to make sure the experiment and data will be saved even if the ddh5 is being used by
    other programs. In this mode, the data is individually saved in files in a .tmp folder. When the experiment is
    finished, the data is unified and saved in the original file.
    If the data is correctly reconstructed, the .tmp folder is deleted. If not you can use the function unify_safe_write_data
    to reconstruct the data.


    :param basedir: The root directory in which data is stored.
        :meth:`.create_file_structure` is creating the structure inside this root and
        determines the file name of the data. The default structure implemented here is
        ``<root>/YYYY-MM-DD/YYYY-mm-dd_THHMMSS_<ID>-<name>/<filename>.ddh5``,
        where <ID> is a short identifier string and <name> is the value of parameter `name`.
        To change this, re-implement :meth:`.data_folder` and/or
        :meth:`.create_file_structure`.
    :param datadict: Initial data object. Must contain at least the structure of the
        data to be able to use :meth:`add_data` to add data.
    :param groupname: Name of the top-level group in the file container. An existing
        group of that name will be deleted.
    :param name: Name of this dataset. Used in path/file creation and added as meta data.
    :param filename: Filename to use. Defaults to 'data.ddh5'.
    :param file_timeout: How long the function will wait for the ddh5 file to unlock. If none uses the default
        value from the :class:`FileOpener`.
    :param safe_write_mode: If True, will save the data in the safe writing mode. Defaults to False.
    """

    # TODO: need an operation mode for not keeping data in memory.
    # TODO: a mode for working with pre-allocated data

    # Sets how many files before the writer creates a new folder in its safe writing mode
    n_files_per_dir = 1000

    # Controls how often the writer reconstructs the data in its safe writing mode.
    # It will reconstruct the data every `n_files_per_reconstruction` files or every `n_seconds_per_reconstruction`
    # seconds, whichever comes first.
    n_files_per_reconstruction = 1000
    n_seconds_per_reconstruction = 10

    def __init__(
        self,
        datadict: DataDict,
        basedir: Union[str, Path] = ".",
        groupname: str = "data",
        name: Optional[str] = None,
        filename: str = "data",
        filepath: Optional[Union[str, Path]] = None,
        file_timeout: Optional[float] = None,
        safe_write_mode: Optional[bool] = False,
    ):
        """Constructor for :class:`.DDH5Writer`"""

        self.basedir = Path(basedir)
        self.datadict = datadict

        if name is None:
            name = ""
        self.name = name

        self.groupname = groupname
        self.filename = Path(filename)

        self.filepath: Optional[Path] = None
        if filepath is not None:
            self.filepath = Path(filepath)

        self.datadict.add_meta("dataset.name", name)
        self.file_timeout = file_timeout
        self.uuid = uuid.uuid1()

        self.safe_write_mode = safe_write_mode
        # Stores how many individual data files have been written for safe_write_mode
        self.n_files = 0
        self.last_update_n_files = 0
        self.last_reconstruction_time = time.time()

    def __enter__(self) -> "DDH5Writer":
        if self.filepath is None:
            self.filepath = _data_file_path(self.data_file_path(), True)
        logger.info(f"Data location: {self.filepath}")

        nrecords: Optional[int] = self.datadict.nrecords()
        if nrecords is not None and nrecords > 0:
            datadict_to_hdf5(
                self.datadict,
                str(self.filepath),
                groupname=self.groupname,
                append_mode=AppendMode.none,
                file_timeout=self.file_timeout,
            )
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ) -> None:
        assert self.filepath is not None

        if self.safe_write_mode:
            try:
                logger.debug("Starting reconstruction of data")
                dd = reconstruct_safe_write_data(self.filepath, file_timeout=self.file_timeout)

                # Makes sure the reconstructed data matches the one in the .tmp folder
                assert datasets_are_equal(dd, self.datadict, ignore_meta=True)

                datadict_to_hdf5(dd, self.filepath, groupname=self.groupname, file_timeout=self.file_timeout, append_mode=AppendMode.none)
                shutil.rmtree(self.filepath.parent / ".tmp")

            except Exception as e:
                logger.error(f"Error while unifying data. Data should be located in the .tmp directory: {e}")
                self.add_tag("__not_reconstructed__")
                raise e

        with FileOpener(self.filepath, "a", timeout=self.file_timeout) as f:
            add_cur_time_attr(f.require_group(self.groupname), name="close")
        if exc_type is None:
            # exiting because the measurement is complete
            self.add_tag("__complete__")
        else:
            # exiting because of an exception
            self.add_tag("__interrupted__")

    def data_folder(self) -> Path:
        """Return the folder, relative to the data root path, in which data will
        be saved.

        Default format:
        ``<basedir>/YYYY-MM-DD/YYYY-mm-ddTHHMMSS_<ID>-<name>``.
        In this implementation we use the first 8 characters of a UUID as ID.

        :returns: The folder path.
        """
        ID = str(self.uuid).split("-")[0]
        parent = f"{datetime.datetime.now().replace(microsecond=0).isoformat().replace(':', '')}_{ID}"
        if self.name:
            parent += f"-{self.name}"
        path = Path(time.strftime("%Y-%m-%d"), parent)
        return path

    def data_file_path(self) -> Path:
        """Determine the filepath of the data file.

        :returns: The filepath of the data file.
        """
        data_folder_path = Path(self.basedir, self.data_folder())
        appendix = ""
        idx = 2
        while data_folder_path.exists():
            appendix = f"-{idx}"
            data_folder_path = Path(self.basedir, str(self.data_folder()) + appendix)
            idx += 1

        return Path(data_folder_path, self.filename)

    def _generate_next_safe_write_path(self):
        """
        Generates the next path for the data to be saved in the safe writing mode. Should not be used for other things.
        """

        now = datetime.datetime.now()

        # Creates tmp folder
        tmp_folder = self.filepath.parent / ".tmp"
        tmp_folder.mkdir(exist_ok=True)

        # Creates today folder
        today_folder = tmp_folder / now.strftime("%Y-%m-%d")
        today_folder.mkdir(exist_ok=True)

        # Creates hour folder
        hour_folder = today_folder / now.strftime("%H")
        hour_folder.mkdir(exist_ok=True)

        # Creates minute folder
        minute_folder = hour_folder / now.strftime("%M")
        minute_folder.mkdir(exist_ok=True)

        n_secs = 0
        second_folder = minute_folder / (now.strftime("%S") + f"_#{str(n_secs)}")
        if second_folder.exists():
            n_data_files = len(list(second_folder.iterdir())) + 1
            if n_data_files >= self.n_files_per_dir:
                keep_searching = True
                while keep_searching:
                    n_secs += 1
                    second_folder = minute_folder / (now.strftime("%S") + f"_#{str(n_secs)}")
                    if not second_folder.exists():
                        keep_searching = False
                        second_folder.mkdir()
                    else:
                        n_data_files = len(list(second_folder.iterdir())) + 1
                        if n_data_files < self.n_files_per_dir:
                            keep_searching = False

        # Creates the filename that follows the structure: yyyy-mm-dd-HHMM-SS#_#<total_number_of_files>.ddh5
        filename = now.strftime("%Y-%m-%d-%H_%M_%S") + f"_{n_secs}_#{self.n_files}.ddh5"
        self.n_files += 1

        return second_folder/filename

    def add_data(self, **kwargs: Any) -> None:
        """Add data to the file (and the internal `DataDict`).

        Requires one keyword argument per data field in the `DataDict`, with
        the key being the name, and value the data to add. It is required that
        all added data has the same number of 'rows', i.e., the most outer dimension
        has to match for data to be inserted faithfully.
        If some data is scalar and others are not, then the data should be reshaped
        to (1, ) for the scalar data, and (1, ...) for the others; in other words,
        an outer dimension with length 1 is added for all.
        """
        self.datadict.add_data(**kwargs)

        if self.safe_write_mode:
            clean_dd_copy = self.datadict.structure()
            clean_dd_copy.add_data(**kwargs)
            filepath = self._generate_next_safe_write_path()

            datadict_to_hdf5(
                clean_dd_copy,
                filepath,
                groupname=self.groupname,
                append_mode=AppendMode.new,
                file_timeout=self.file_timeout,
            )

            delta_t = time.time() - self.last_reconstruction_time

            # Reconstructs the data every n_files_per_reconstruction files or every n_seconds_per_reconstruction seconds
            if (self.n_files - self.last_update_n_files >= self.n_files_per_reconstruction or
                    delta_t > self.n_seconds_per_reconstruction):
                try:
                    dd = reconstruct_safe_write_data(self.filepath, unification_from_scratch=False,
                                                     file_timeout=self.file_timeout)
                    datadict_to_hdf5(dd, self.filepath, groupname=self.groupname, file_timeout=self.file_timeout, append_mode=AppendMode.none)
                except RuntimeError as e:
                    logger.warning(f"Error while unifying data: {e} \nData is still getting saved in .tmp directory.")

                with FileOpener(self.filepath, "a", timeout=self.file_timeout) as f:
                    add_cur_time_attr(f, name="last_change")
                    add_cur_time_attr(f[self.groupname], name="last_change")

                # Even if I fail at reconstruction, I want to wait the same amount as if it was successful to try again.
                self.last_reconstruction_time = time.time()
                self.last_update_n_files = self.n_files

        else:
            nrecords = self.datadict.nrecords()
            if nrecords is not None and nrecords > 0:
                datadict_to_hdf5(
                    self.datadict,
                    str(self.filepath),
                    groupname=self.groupname,
                    file_timeout=self.file_timeout,
                )

                assert self.filepath is not None
                with FileOpener(self.filepath, "a", timeout=self.file_timeout) as f:
                    add_cur_time_attr(f, name="last_change")
                    add_cur_time_attr(f[self.groupname], name="last_change")

    # convenience methods for saving things in the same directory as the ddh5 file

    def add_tag(self, tags: Union[str, Collection[str]]) -> None:
        assert self.filepath is not None
        if isinstance(tags, str):
            tags = [tags]
        for tag in tags:
            open(self.filepath.parent / f"{tag}.tag", "x").close()

    def backup_file(self, paths: Union[str, Collection[str]]) -> None:
        assert self.filepath is not None
        if isinstance(paths, str):
            paths = [paths]
        for path in paths:
            shutil.copy(path, self.filepath.parent)

    def save_text(self, name: str, text: str) -> None:
        assert self.filepath is not None
        with open(self.filepath.parent / name, "x") as f:
            f.write(text)

    def save_dict(self, name: str, d: dict) -> None:
        assert self.filepath is not None
        with open(self.filepath.parent / name, "x") as f:
            json.dump(d, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)


def data_info(folder: str, fn: str = "data.ddh5", do_print: bool = True):
    fn = Path(folder, fn)
    dataset = datadict_from_hdf5(fn)
    if do_print:
        print(dataset)
    else:
        return str(dataset)


def timestamp_from_path(p: Path) -> datetime.datetime:
    """Return a `datetime` timestamp from a standard-formatted path.
    Assumes that the path stem has a timestamp that begins in ISO-like format
    ``YYYY-mm-ddTHHMMSS``.
    """
    timestring = str(p.stem)[:13] + ":" + str(p.stem)[13:15] + ":" + str(p.stem)[15:17]
    return datetime.datetime.fromisoformat(timestring)


def find_data(
    root,
    newer_than: Optional[datetime.datetime] = None,
    older_than: Optional[datetime.datetime] = None,
    folder_filter: Optional[str] = None,
) -> List[Path]:
    if not isinstance(root, Path):
        root = Path(root)

    folders = {}
    for f, dirs, files in os.walk(root):
        if "data.ddh5" in files:
            fp = Path(f)
            ts = timestamp_from_path(fp)
            if newer_than is not None and ts <= newer_than:
                continue
            if older_than is not None and ts >= older_than:
                continue
            if folder_filter is not None:
                pattern = re.compile(folder_filter)
                if not pattern.search(str(fp.stem)):
                    continue

            folders[fp] = (dirs, files)
    return folders


def most_recent_data_path(
    root,
    older_than: Optional[datetime.datetime] = None,
    folder_filter: Optional[str] = None,
) -> Path:
    folders = find_data(root, older_than=older_than, folder_filter=folder_filter)
    return sorted(folders.keys())[-1]


def load_as_xr(
    folder: Path, fn="data.ddh5", fields: Optional[List[str]] = None
) -> xr.Dataset:
    """Load ddh5 data as xarray (only for gridable data).

    Parameters
    ----------
    folder :
        data folder
    fn : str, optional
        filename, by default 'data.ddh5'

    Returns
    -------
    _type_
        _description_
    """
    fn = folder / fn
    dd = datadict_from_hdf5(fn)
    if fields is not None:
        dd = dd.extract(fields)
    xrdata = split_complex(dd2xr(datadict_to_meshgrid(dd)))
    xrdata.attrs["raw_data_folder"] = str(folder.resolve())
    xrdata.attrs["raw_data_fn"] = str(fn)
    return xrdata


def load_as_df(folder, fn="data.ddh5"):
    fn = folder / fn
    dfdata = split_complex(dd2df(datadict_from_hdf5(fn)))
    return dfdata
