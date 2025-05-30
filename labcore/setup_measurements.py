import os
import sys
import logging
from typing import Optional, Any, Union, List, Dict, Tuple
from functools import partial
from dataclasses import dataclass
from pathlib import Path

from instrumentserver.client import Client, ProxyInstrument
from instrumentserver.helpers import nestedAttributeFromString

from .data.datadict import DataDict
from .data.datadict_storage import data_info
from .measurement.storage import run_and_save_sweep
from .measurement import Sweep
from .utils.misc import get_environment_packages, commit_changes_in_repo

# constants
WD = os.getcwd()
DATADIR = os.path.join(WD, 'data')


@dataclass
class Options:
    instrument_clients: Optional[Dict[str, Client]] = None
    parameters: Optional[ProxyInstrument] = None
    qubit_defaults: Optional[callable]= lambda: None


options = Options()


def param_from_name(name: str, ):
    return nestedAttributeFromString(options.parameters, name)


def getp(name: str, default=None, raise_if_missing=False):
    if options.parameters is None:
        logger.error("No parameter manager defined. cannot get/set params!")
        return None
    
    try: 
        p = param_from_name(name)
        return p()
    except AttributeError:
        if raise_if_missing:
            raise
        else:
            return default


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


def run_measurement(sweep: Sweep, name: str, safe_write_mode: bool = False, **kwargs) -> Tuple[Union[str, Path], Optional[DataDict]]:
    """
    Wrapper function around run_and_save_sweep that makes sure you are saving your measurement with all the necessary
    metadata around it.
    This includes the current git commit hash, the python environment, and the snapshot of all
    instruments and parameters.
    It is considered bad practice to have uncommitted changes in a repository that is installed locally.
    If you have to change something from a repo, please open up a PR, this usually means that something is wrong there.

    Assumptions
    -----------

    To use this function, you need:
    * Have an instrumentserver instance running with an instantiated client in options.instrument_clients
    * Have a parameter manager proxy instrument in options.parameters. You get this simply
      by passing the usually called params variable to options.

    How-to set up auto-committing
    ---------------------------

    To get this function to commit your measurement code before running the measurement,
    you need to follow the following steps:

    * Run `git init` in the directory where you are running your measurements.
      If you are getting an error saying "could not set core.filemode to 'false'", run `sudo git init` instead.
    * In your measurement folder copy the file :doc:`../doc/example.gitignore` and rename it to '.gitignore'.
    * Commit all the files in your measurement folder running the command `git add -a -m "Initial commit"`.

    After those 3 steps have been taken, every time a measurement is run, the folder should autocommit before any measurement.
    """
    if options.instrument_clients is None:
        raise RuntimeError('it looks like options.instrument_clients is not configured.')
    if options.parameters is None:
        raise RuntimeError('it looks like options.parameters is not configured.')

    for n, c in options.instrument_clients.items():
        # Signature for snapshot changed, we need to check for both now.
        if hasattr(c, 'get_snapshot'):
            kwargs[n] = c.get_snapshot
        elif hasattr(c, 'snapshot'):
            kwargs[n] = c.snapshot
        else:
            raise RuntimeError(f"Could not find snapshot method for client {n}. Please update all packages.")

    kwargs['parameters'] = options.parameters.toParamDict
    
    py_env = get_environment_packages()

    current_dir = Path.cwd()
    commit_hash = commit_changes_in_repo(current_dir)
   
    if commit_hash is None:
        logger.warning("The current directory is not a git repository, your measurement code will not be tracked.")
    
    save_kwargs = {
        'sweep': sweep,
        'data_dir': DATADIR,
        'name': name,
        'save_action_kwargs': True,
        'python_environment': py_env,
        'safe_write_mode': safe_write_mode,
        **kwargs
    }
    if commit_hash is not None:
        save_kwargs['current_commit'] = {"measurement-hash": commit_hash}

    data_location, data = run_and_save_sweep(**save_kwargs)

    logger.info(f"""
==========
Saved data at {data_location}:
{data_info(data_location, do_print=False)}
=========""")
    return data_location, data


