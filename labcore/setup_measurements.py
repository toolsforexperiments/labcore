import os
import sys
import logging
from typing import Optional, Any, Union, List, Dict, Tuple
from functools import partial
from dataclasses import dataclass
from pathlib import Path

from instrumentserver.client import Client, ProxyInstrument

from .data.datadict import DataDict
from .data.datadict_storage import data_info
from .measurement.storage import run_and_save_sweep
from .measurement import Sweep


# constants
WD = os.getcwd()
DATADIR = os.path.join(WD, 'data')


@dataclass
class Options:
    instrument_clients: Optional[Dict[str, Client]] = None
    parameters: Optional[ProxyInstrument] = None

options = Options()


# this function sets up our general logging
def setup_logging() -> logging.Logger:
    """Setup logging in a reasonable way. Note: we use the root logger since
    our measurements typically run in the console directly and we want
    logging to work from scripts that are directly run in the console.

    Returns
    -------
    The logger that has been setup.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for h in logger.handlers:
        logger.removeHandler(h)
        del h

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d\t| %(name)s\t| %(levelname)s\t| %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    fh = logging.FileHandler('measurement.log')
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    fmt = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] [%(name)s: %(levelname)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    streamHandler = logging.StreamHandler(sys.stdout)
    streamHandler.setFormatter(fmt)
    streamHandler.setLevel(logging.INFO)
    logger.addHandler(streamHandler)
    logger.info(f"Logging set up for {logger}.")
    return logger

# Create the logger
logger = setup_logging()

def find_or_create_remote_instrument(cli: Client, ins_name: str, ins_class: Optional[str]=None,
                                     *args: Any, **kwargs: Any) -> ProxyInstrument:
    """Finds or creates an instrument in an instrument server.

    Parameters
    ----------
    cli
        instance of the client pointing to the instrument server
    ins_name
        name of the instrument to find or to create
    ins_class
        the class of the instrument (import path as string) if creating a new instrument
    args
        will be passed to the instrument creation call
    kwargs
        will be passed to the instrument creation call

    Returns
    -------
    Proxy to the remote instrument
    """
    if ins_name in cli.list_instruments():
        return cli.get_instrument(ins_name)

    if ins_class is None:
        raise ValueError('Need a class to create a new instrument')

    ins = cli.create_instrument(
        instrument_class=ins_class,
        name=ins_name, *args, **kwargs)

    return ins


def run_measurement(sweep: Sweep, name: str, **kwargs) -> Tuple[Union[str, Path], Optional[DataDict]]:
    if options.instrument_clients is None:
        raise RuntimeError('it looks like options.instrument_clients is not configured.')
    if options.parameters is None:
        raise RuntimeError('it looks like options.parameters is not configured.')

    for n, c in options.instrument_clients.items():
        kwargs[n] = c.snapshot
    kwargs['parameters'] = options.parameters.toParamDict

    data_location, data = run_and_save_sweep(
        sweep=sweep,
        data_dir=DATADIR,
        name=name,
        save_action_kwargs=True,
        **kwargs)

    info = data

    logger.info(f"""
==========
Saved data at {data_location}:
{data_info(data_location, do_print=False)}
=========""")
    return data_location, data


