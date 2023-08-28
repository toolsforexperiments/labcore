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
from enum import Enum
from typing import Any, Union, Optional, Dict, Type, Collection
from types import TracebackType
from pathlib import Path

import numpy as np
import h5py

from qcodes.utils import NumpyJSONEncoder
#from plottr import QtGui, Signal, Slot, QtWidgets, QtCore
'''
from ..node import (
    Node, NodeWidget, updateOption, updateGuiFromNode,
    emitGuiUpdate,
)
'''
from .datadict import DataDict, is_meta_key, DataDictBase

__author__ = 'Wolfgang Pfaff'
__license__ = 'MIT'

DATAFILEXT = 'ddh5'
TIMESTRFORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger(__name__)

# FIXME: need correct handling of dtypes and list/array conversion

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

    if type(obj) == np.ndarray and obj.dtype.kind == 'U':
        return np.char.encode(obj, encoding='utf8')

    return obj


def deh5ify(obj: Any) -> Any:
    """Convert slightly mangled types back to more handy ones.

    :param obj: Input object.
    :return: Object
    """
    if type(obj) == bytes:
        return obj.decode()

    if type(obj) == np.ndarray and obj.dtype.kind == 'S':
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


def add_cur_time_attr(h5obj: Any, name: str = 'creation',
                      prefix: str = '__', suffix: str = '__') -> None:
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

    set_attr(h5obj, prefix + name + '_time_sec' + suffix, tsec)
    set_attr(h5obj, prefix + name + '_time_str' + suffix, tstr)


# elementary reading/writing

def _data_file_path(file: Union[str, Path], init_directory: bool = False) -> Path:
    """Get the full filepath of the data file.
    If `init_directory` is True, then create the parent directory."""

    path = Path(file)

    if path.suffix != f'.{DATAFILEXT}':
        path = Path(path.parent, path.stem + f'.{DATAFILEXT}')
    if init_directory:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def datadict_to_hdf5(datadict: DataDict,
                     path: Union[str, Path],
                     groupname: str = 'data',
                     append_mode: AppendMode = AppendMode.new,
                     file_timeout: Optional[float] = None) -> None:
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

    with FileOpener(filepath, 'a', file_timeout) as f:
        if append_mode is AppendMode.none:
            init_file(f, groupname)
        assert groupname in f
        grp = f[groupname]

        # add top-level meta data.
        for k, v in datadict.meta_items(clean_keys=False):
            set_attr(grp, k, v)

        for k, v in datadict.data_items():
            data = v['values']
            shp = data.shape
            nrows = shp[0]

            # create new dataset, add axes and unit metadata
            if k not in grp:
                maxshp = tuple([None] + list(shp[1:]))
                ds = grp.create_dataset(k, maxshape=maxshp, data=data)

                # add meta data
                add_cur_time_attr(ds)

                if v.get('axes', []):
                    set_attr(ds, 'axes', v['axes'])
                if v.get('unit', "") != "":
                    set_attr(ds, 'unit', v['unit'])

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


def init_file(f: h5py.File,
              groupname: str = 'data') -> None:

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


def datadict_from_hdf5(path: Union[str, Path],
                       groupname: str = 'data',
                       startidx: Union[int, None] = None,
                       stopidx: Union[int, None] = None,
                       structure_only: bool = False,
                       ignore_unequal_lengths: bool = True,
                       file_timeout: Optional[float] = None) -> DataDict:
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
        raise ValueError("Specified file does not exist.")

    if startidx is None:
        startidx = 0

    res = {}
    with FileOpener(filepath, 'r', file_timeout) as f:
        if groupname not in f:
            raise ValueError('Group does not exist.')

        grp = f[groupname]
        keys = list(grp.keys())
        lens = [len(grp[k][:]) for k in keys]

        if len(set(lens)) > 1:
            if not ignore_unequal_lengths:
                raise RuntimeError('Unequal lengths in the datasets.')

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
            entry: Dict[str, Union[Collection[Any], np.ndarray]] = dict(values=np.array([]), )

            if 'axes' in ds.attrs:
                entry['axes'] = deh5ify(ds.attrs['axes']).tolist()
            else:
                entry['axes'] = []

            if 'unit' in ds.attrs:
                entry['unit'] = deh5ify(ds.attrs['unit'])

            if not structure_only:
                entry['values'] = ds[startidx:stopidx]

            entry['__shape__'] = ds[:].shape

            # and now the meta data
            for attr in ds.attrs:
                if is_meta_key(attr):
                    _val = deh5ify(ds.attrs[attr])
                    entry[attr] = deh5ify(ds.attrs[attr])

            res[k] = entry

    dd = DataDict(**res)
    dd.validate()
    return dd


def all_datadicts_from_hdf5(path: Union[str, Path], file_timeout: Optional[float] = None, **kwargs: Any) -> Dict[str, Any]:
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
    with FileOpener(filepath, 'r', file_timeout) as f:
        keys = [k for k in f.keys()]
    for k in keys:
        ret[k] = datadict_from_hdf5(path=path, groupname=k, file_timeout=file_timeout, **kwargs)
    return ret


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

    def __init__(self, path: Union[Path, str],
                 mode: str = 'r',
                 timeout: Optional[float] = None,
                 test_delay: float = 0.1):
        self.path = Path(path)
        self.lock_path = self.path.parent.joinpath("~" + str(self.path.stem) + '.lock')
        if mode not in ['r', 'w', 'w-', 'a']:
            raise ValueError("Only 'r', 'w', 'w-', 'a' modes are supported.")
        self.mode = mode
        self.default_timeout = 30.
        if timeout is None:
            self.timeout = self.default_timeout
        else:
            self.timeout = timeout
        self.test_delay = test_delay

        self.file: Optional[h5py.File] = None

    def __enter__(self) -> h5py.File:
        self.file = self.open_when_unlocked()
        return self.file

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 exc_traceback: Optional[TracebackType]) -> None:
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
                    time.sleep(self.test_delay)  # don't overwhelm the FS by very fast repeated calls.
                    if time.time() - t0 > self.timeout:
                        raise RuntimeError('Waiting or file unlock timeout')

            time.sleep(self.test_delay)  # don't overwhelm the FS by very fast repeated calls.
            if time.time() - t0 > self.timeout:
                raise RuntimeError('Lock file remained for longer than timeout time')
