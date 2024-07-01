"""
Reconstructs the safe write data from a .tmp folder and saves it to a ddh5 file. Used by the CLI command reconstruct-data.
Meant to be a backup way of reconstructing data if something goes wrong.
"""

import logging
import argparse
from pathlib import Path

from labcore.data.datadict_storage import reconstruct_safe_write_data, datadict_to_hdf5


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Reconstructing the safe write data')

    parser.add_argument("path",
                        help="path to directory containing a .tmp folder. .tmp doesn't have to be in the path",
                        default=None)
    parser.add_argument("--filename",
                        help="Name for the newly created ddh5 file. Defaults to data.ddh5",
                        default="data.ddh5")
    parser.add_argument("--file_timeout",
                        help="time before a timeout error is raised when interacting with files, in seconds",
                        default=None)

    args = parser.parse_args()

    path = Path(args.path)
    file_timeout = args.file_timeout

    # Checks if the path ends in .tmp, if not, adds it.
    if path.suffix != ".tmp":
        ddh5_path = path / args.filename
        path = path / ".tmp"
    else:
        ddh5_path = path.parent / args.filename

    if not path.exists():
        raise FileNotFoundError(f"No .tmp folder found in {path}")
    if ddh5_path.exists():
        raise FileExistsError(f"File {ddh5_path} already exists. Remove it or change filename before continuing.")

    dd = reconstruct_safe_write_data(path, file_timeout=file_timeout)
    
    datadict_to_hdf5(dd, ddh5_path)

    logger.info(f"Reconstruction of safe write data in {path} completed.")




