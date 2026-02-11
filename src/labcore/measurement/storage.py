"""plottr.data.datadict_storage

Provides file-storage tools for the DataDict class.

Description of the HDF5 storage format
======================================

We use a simple mapping from DataDict to the HDF5 file. Within the file,
a single DataDict is stored in a (top-level) group of the file.
The data fields are datasets within that group.

Global meta data of the DataDict are attributes of the group; field meta data
are attributes of the dataset (incl., the `unit` and `axes` values). The meta
data keys are given exactly like in the DataDict, i.e., incl the double
underscore pre- and suffix.
"""
import os
import time
from enum import Enum
from typing import Any, Union, Optional, Dict, Type, Collection, List, Tuple
from types import TracebackType
from pathlib import Path
import json
import pickle
import shutil
import glob
import logging

import numpy as np
import h5py

from ..data.datadict import DataDict, is_meta_key
from ..data.datadict_storage import DDH5Writer

from .sweep import Sweep

__author__ = 'Wolfgang Pfaff'
__license__ = 'MIT'

TIMESTRFORMAT = "%Y-%m-%dT%H%M%S"

logger = logging.getLogger(__name__)


def _create_datadict_structure(sweep: Sweep) -> DataDict:
    """
    Returns a structured DataDict from the DataSpecs of a Sweep.

    :param sweep: Sweep object from which the DataDict is created.
    """

    data_specs = sweep.get_data_specs()
    data_dict = DataDict()
    for spec in data_specs:

        depends_on = spec.depends_on
        unit = spec.unit
        name = spec.name

        # Checks which fields have information and which ones are None.
        if depends_on is None:
            if unit is None:
                data_dict[name] = dict()
            else:
                data_dict[name] = dict(unit=unit)
        else:
            if unit is None:
                data_dict[name] = dict(axes=depends_on)
            else:
                data_dict[name] = dict(axes=depends_on, unit=unit)

    data_dict.validate()

    return data_dict


def _check_none(line: Dict, all: bool = True) -> bool:
    """
    Checks if the values in a Dict are all None.
    :returns: True if all values are None, False otherwise.
    """
    if all:
        for k, v in line.items():
            if v is None:
                return True
        return False

    if len(set(line.values())) == 1:
        for k, v in line.items():
            if v is None:
                return True
    return False


def _save_dictionary(dict: Dict, filepath: str) -> None:
    with open(filepath, 'w') as f:
        json.dump(dict, f, indent=2, sort_keys=True, cls=NumpyEncoder)


def _pickle_and_save(obj, filepath: str) -> None:
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    except TypeError as pickle_error:
        print(f'Object could not be pickled: {pickle_error.args}')


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def run_and_save_sweep(sweep: Sweep,
                       data_dir: str,
                       name: str,
                       ignore_all_None_results: bool = True,
                       save_action_kwargs: bool = False,
                       add_timestamps = False,
                       archive_files: Optional[List[str]] = None,
                       return_data: bool = False,
                       safe_write_mode: bool = False,
                       **extra_saving_items) -> Tuple[Union[str, Path], Optional[DataDict]]:
    """
    Iterates through a sweep, saving the data coming through it into a file called <name> at <data_dir> directory.

    :param sweep: Sweep object to iterate through.
    :param data_dir: Directory of file location.
    :param name: Name of the file.
    :param ignore_all_None_results: if ``True``, don't save any records that contain a ``None``.
        if ``False``, only do not save records that are all-``None``.
    :param  save_action_kwargs: If ``True``, the action_kwargs of the sweep will be saved as a json file named after
        the first key of the kwargs dctionary followed by '_action_kwargs' in the same directory as the data.
    :param archive_files: List of files to copy into a folder called 'archived_files' in
        the same directory that the data is saved. It should be a list of paths (str), regular expressions are supported.
        If a folder is passed, it will copy the entire folder and all of its subdirectories and files into the
        archived_files folder. If one of the arguments could not be found, a message will be printed and the measurement
        will be performed without the file being archived. An exception is raised if the type is invalid.

        e.g. archive_files=['*.txt', 'calibration_files', '../test_file.py'].  '*.txt' will copy every txt file
        located in the working directory. 'calibration_files' will copy the entire folder called calibration_files from
        the working directory into the archived_files folder. '../test_file.py' will copy the script test_file.py from
        one directory above the working directory.
    :param extra_saving_items: Kwargs for extra objects that should be saved. If the kwarg is a dictionary, the function
        will try and save it as a JSON file. If the dictionary contains objects that are not JSON serializable it will
        be pickled. Any other kind of object will be pickled too. The files will have their keys as names.
    :param safe_write_mode: Indicates if the data should be written in safe mode or not. Look into ddh5 writer for more
        info.

    :raises TypeError: A Typerror is raised if the object passed for archive_files is not correct
    """
    data_dict = _create_datadict_structure(sweep)

    # Creates a file even when it fails.
    with DDH5Writer(data_dict, data_dir, name=name, safe_write_mode=safe_write_mode) as writer:

        # Saving meta-data
        dir: Path = writer.filepath.parent
        if add_timestamps:
            t = time.localtime()
            time_stamp = time.strftime(TIMESTRFORMAT, t) + '_'

        for key, val in extra_saving_items.items():
            if callable(val):
                value = val()
            else:
                value = val

            if add_timestamps:
                pickle_path_file = os.path.join(dir, time_stamp + key + '.pickle')
                json_path_file = os.path.join(dir, time_stamp + key + '.json')
            else:
                pickle_path_file = os.path.join(dir, key + '.pickle')
                json_path_file = os.path.join(dir, key + '.json')

            if isinstance(value, dict):
                try:
                    _save_dictionary(value, json_path_file)
                except TypeError as error:
                    # Delete the file created by _save_dictionary. This file does not contain the complete dictionary.
                    if os.path.isfile(json_path_file):
                        os.remove(json_path_file)

                    logging.info(f'{key} has not been able to save to json: {error.args}.'
                                 f' The item will be pickled instead.')
                    _pickle_and_save(value, pickle_path_file)
            else:
                _pickle_and_save(value, pickle_path_file)

        # Save the kwargs
        if save_action_kwargs:
            if add_timestamps:
                json_path_file = os.path.join(dir, time_stamp + 'sweep_action_kwargs.json')
            else:
                json_path_file = os.path.join(dir, 'sweep_action_kwargs.json')
            _save_dictionary(sweep.action_kwargs, json_path_file)

        # Save archive_files
        if archive_files is not None:
            archive_files_dir = os.path.join(dir, 'archive_files')
            os.mkdir(archive_files_dir)
            if not isinstance(archive_files, list) and not isinstance(archive_files, tuple):
                if isinstance(archive_files, str):
                    archive_files = [archive_files]
                else:
                    raise TypeError(f'{type(archive_files)} is not a list.')
            for path in archive_files:
                if os.path.isdir(path):
                    folder_name = os.path.basename(path)
                    if folder_name == '':
                        folder_name = os.path.basename(os.path.dirname(path))

                    shutil.copytree(path, os.path.join(archive_files_dir, folder_name), dirs_exist_ok=True)
                elif os.path.isfile(path):
                    shutil.copy(path, archive_files_dir)
                else:
                    matches = glob.glob(path, recursive=True)
                    if len(matches) == 0:
                        logging.info(f'{path} could not be found. Measurement will continue without archiving {path}')
                    for file in matches:
                        shutil.copy(file, archive_files_dir)

        # Save data.
        try:
            for line in sweep:
                if not _check_none(line, all=ignore_all_None_results):
                    writer.add_data(**line)
        except KeyboardInterrupt:
            logger.warning('Sweep stopped by Keyboard interrupt. Data completed before interrupt should be saved.')
            ret = (dir, data_dict) if return_data else (dir, None)
            return ret

    logger.info('The measurement has finished successfully and all of the data has been saved.')
    ret = (dir, data_dict) if return_data else (dir, None)
    return ret

