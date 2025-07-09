from typing import Callable, Dict, Generator, List, Optional
from functools import wraps
from dataclasses import dataclass
import time
import logging

import numpy as np

from qm.qua import *
try:
    from qm.QuantumMachinesManager import QuantumMachinesManager
except:
    from qm.quantum_machines_manager import QuantumMachinesManager

from labcore.measurement import *
from labcore.measurement.record import make_data_spec
from labcore.measurement.sweep import AsyncRecord

from .config import QMConfig

# --- Options that need to be set by the user for the OPX to work ---
# config object that when called returns the config dictionary as expected by the OPX
config: Optional[QMConfig] = None  # OPX config dictionary

# WARNING: DO NOT TOUCH THIS VARIABLE. IT IS GLOBAL AND HANDLED BY THE CONTEXT MANAGER.
_qmachine_context = None


logger = logging.getLogger(__name__)


class QuantumMachineContext:
    """
    Context manager for the Quantum Machine. It will open the machine when entering the context and close it when
    exiting, after all measurement completed. This is used via a with statement, i.e.:

    ```
    with QuantumMachineContext() as qmc:
        [your measurement code here]
    ```

    This does not need to be used, but if measurements are done repeatedly and precompiling with the OPX is
    desired, it saves some time.

    Warning: Using a context manager doesn't let you update the config of the OPX. If you want to change the
    config, you need to open a new quantum machine or use precompiled measurements to overwrite aspects of the
    measurement on the fly.
    """

    def __init__(self, wf_overrides: Optional[Dict] = None, if_overrides: Optional[Dict] = None, *args, **kwargs):
        """
        Initializes the context manager with a function to be executed, its arguments, and optional overrides.

        :param fun: The function to be executed in the quantum machine.
        :param args: Positional arguments for the function.
        :param wf_overrides: Optional dictionary of overrides for the waveforms.
        :param if_overrides: Optional dictionary of overrides for the intermediate frequencies.
        :param kwargs: Keyword arguments for the function.
        """
        global config
        self.wf_overrides = wf_overrides
        self.if_overrides = if_overrides
        self.kwargs = kwargs

        self._qmachine_mgr = None
        self._qmachine = None
        self._program_id = None

    def __enter__(self):
        global config, _qmachine_context
        _qmachine_mgr = QuantumMachinesManager(
                host=config.opx_address,
                port=config.opx_port, 
                cluster_name=config.cluster_name,
                octave=config.octave
            )
    
        self._qmachine = _qmachine_mgr.open_qm(config(), close_other_machines=False)
        _qmachine_context = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global _qmachine_context
        if self._qmachine is not None:
            self._qmachine.close()
        _qmachine_context = None


@dataclass
class TimedOPXData(DataSpec):
    def __post_init__(self):
        super().__post_init__()
        if self.depends_on is None or len(self.depends_on) == 0:
            deps = []
        else:
            deps = list(self.depends_on)
        self.depends_on = [self.name+'_time_points'] + deps

@dataclass
class ComplexOPXData(DataSpec):
    i_data_stream: str = 'I'
    q_data_stream: str = 'Q'


class RecordOPXdata(AsyncRecord):
    """
    Implementation of AsyncRecord for use with the OPX machine.
    """

    def __init__(self, *specs):
        self.communicator = {}
        # self.communicator['raw_variables'] = []
        self.user_data = []
        self.specs = []
        for s in specs:
            spec = make_data_spec(s)
            self.specs.append(spec)
            if isinstance(spec, TimedOPXData):
                tspec = indep(spec.name + "_time_points")
                self.specs.append(tspec)
                self.user_data.append(tspec.name)

    def setup(self, fun, *args, **kwargs) -> None:
        """
        Establishes connection with the OPX and starts the measurement. The config of the OPX is passed through
        the module variable global_config. It saves the result handles and saves initial values to the communicator
        dictionary.
        """
        global _qmachine_context
        self.communicator["self_managed"] = False
        # Start the measurement in the OPX.
        if _qmachine_context is None:
            qmachine_mgr = QuantumMachinesManager(
                host=config.opx_address,
                port=config.opx_port, 
                cluster_name=config.cluster_name,
                octave=config.octave
            )

            qmachine = qmachine_mgr.open_qm(config(), close_other_machines=False)
            logger.info(f"current QM: {qmachine}, {qmachine.id}")
            self.communicator["self_managed"] = True
        else:
            qmachine_mgr = _qmachine_context._qmachine_mgr
            qmachine = _qmachine_context._qmachine

        job = qmachine.execute(fun(*args, **kwargs))
        result_handles = job.result_handles

        # Save the result handle and create initial parameters in the communicator used in the collector.
        self.communicator['result_handles'] = result_handles
        self.communicator['active'] = True
        self.communicator['counter'] = 0
        self.communicator['manager'] = qmachine_mgr
        self.communicator['qmachine'] = qmachine
        self.communicator['qmachine_id'] = qmachine.id

    # FIXME change this such that we make sure that we have enough data on all handles
    def _wait_for_data(self, batchsize: int) -> None:
        """
        Waits for the opx to have measured more data points than the ones indicated in the batchsize. Also checks that
        the OPX is still collecting data, when the OPX is no longer processing, turn communicator['active'] to False to
        exhaust the collector.

        :param batchsize: Size of batch. How many data-points is the minimum for the sweep to get in an iteration.
                          e.g. if 5, _control_progress will keep running until at least 5 new data-points
                          are available for collection.
        """

        # When ready becomes True, the infinite loop stops.
        ready = False

        # Collect necessary values from communicator.
        res_handle = self.communicator['result_handles']
        counter = self.communicator['counter']

        while not ready:
            statuses = []
            processing = []
            for name, handle in res_handle:
                current_datapoint = handle.count_so_far()

                # Check if the OPX is still processing.
                if res_handle.is_processing():
                    processing.append(True)

                    # Check if enough data-points are available.
                    if current_datapoint - counter >= batchsize:
                        statuses.append(True)
                    else:
                        statuses.append(False)

                else:
                    # Once the OPX is done processing turn ready True and turn active False to exhaust the generator.
                    statuses.append(True)
                    processing.append(False)

            if not False in statuses:
                ready = True
            if not True in processing:
                self.communicator['active'] = False

    def cleanup(self):
        """
        Functions in charge of cleaning up any software tools that needs cleanup.

        Currently, manually closes the _qmachine in the OPT so that simultaneous measurements can occur.
        """
        logger.info('Cleaning up')

        if self.communicator["self_managed"]:
            open_machines = self.communicator["manager"].list_open_quantum_machines()
            logger.info(f"currently open QMs: {open_machines}")
            machine_id = self.communicator["qmachine_id"]
            self.communicator["qmachine"].close()
            logger.info(f"QM with ID {machine_id} closed.")

            self.communicator["qmachine"] = None
            self.communicator["manager"] = None



    def collect(self, batchsize: int = 100) -> Generator[Dict, None, None]:
        """
        Implementation of collector for the OPX. Collects new data-points from the OPX and yields them in a dictionary
        with the names of the recorded variables as keywords and numpy arrays with the values. Raises ValueError if a
        stream name inside the QUA program has a different name than a recorded variable and if the amount of recorded
        variables and streams are different.

        :param batchsize: Size of batch. How many data-points is the minimum for the sweep to get in an iteration.
                          e.g. if 5, _control_progress will keep running until at least 5 new data-points
                          are available for collection.
        """

        # Get the result_handles from the communicator.
        result_handle = self.communicator['result_handles']
        try:
            while self.communicator['active']:
                # Restart values for each iteration.
                return_data = {}
                counter = self.communicator['counter']  # Previous iteration data-point number.
                first = True
                available_points = 0
                ds: Optional[DataSpec] = None

                # Make sure that the result_handle is active.
                if result_handle is None:
                    yield None

                # Waits until new data-points are ready to be gathered.
                self._wait_for_data(batchsize)

                def get_data_from_handle(name, up_to):
                    if up_to == counter:
                        return None
                    handle = result_handle.get(name)
                    handle.wait_for_values(up_to)
                    data = np.squeeze(handle.fetch(slice(counter, up_to))['value'])
                    return data

                for i, ds in enumerate(self.specs):
                    if isinstance(ds, ComplexOPXData):
                        iname = ds.i_data_stream
                        qname = ds.q_data_stream
                        if i == 0:
                            available_points = result_handle.get(iname).count_so_far()
                        idata = get_data_from_handle(iname, up_to=available_points)
                        qdata = get_data_from_handle(qname, up_to=available_points)
                        if (qdata is None or idata is None):
                            print(f'qdata is: {qdata}')
                            print(f'idata is: {idata}')
                            print(f'available points is:{available_points}')
                            print(f'i is: {i}')
                            print(f'ds is: {ds}')
                            print(f'iname is: {iname}')
                            print(f'qname is: {qdata}')
                            print(f'am I active: {self.communicator["active"]}')
                            print(f'counter is: {self.communicator["counter"]}')

                        if qdata is not None and idata is not None:
                            return_data[ds.name] = idata + 1j*qdata

                    elif ds.name in self.user_data:
                        continue

                    elif ds.name not in result_handle:
                        raise RuntimeError(f'{ds.name} specified but cannot be found in result handle.')

                    else:
                        name = ds.name
                        if i == 0:
                            available_points = result_handle.get(name).count_so_far()
                        return_data[name] = get_data_from_handle(name, up_to=available_points)

                    if isinstance(ds, TimedOPXData):
                        data = return_data[ds.name]
                        if data is not None:
                            tvals = np.arange(1, data.shape[-1]+1)
                            if len(data.shape) == 1:
                                return_data[name + '_time_points'] = tvals
                            elif len(data.shape) == 2:
                                return_data[name + '_time_points'] = np.tile(tvals, data.shape[0]).reshape(data.shape[0], -1)
                            else:
                                raise NotImplementedError('someone needs to look at data saving ASAP...')

                self.communicator['counter'] = available_points
                yield return_data

        finally:
            self.cleanup()


class RecordPrecompiledOPXdata(RecordOPXdata):
    """
    Implementation of AsyncRecord for use with precompiled OPX programs.
    
    To pass either waveform or IF overrides, use the QuantumMachineContext and set the overrides as attributes.
    The overrides must be passed as dictionaries.
    
    For the waveform overrides the keys are the names of the waveforms as defined in the OPX config file, and
    the values are the new waveform arrays. For an arbitrary (as defined in the qmconfig) waveform to be overridable,
    the waveform must have `"is_overridable": True` set. Constant waveforms do not need to be set as such: the override
    will simply be a constant value. Other waveform types are not overridable.

    For the IF overrides the keys are the names of the elements as defined in the OPX config file, and
    the values are the new intermediate frequencies in Hz.

    Usage example:
    ```
    def create_readout_wf(amp):
        wf_samples = [0.0] * int(params.q01.readout.short.buffer()) + [amp] * int(
                params.q01.readout.short.len()
                - 2 * params.q01.readout.short.buffer()
            ) + [0.0] * int(params.q01.readout.short.buffer())
        return wf_samples

    def create_drive_wf(amp):
        return amp

    with QuantumMachineContext() as qmc:
        loc = measure_time_rabi() 

        qmc.wf_overrides = {
            "waveforms": {
                f"q01_short_readout_wf": create_readout_wf(),
                f"q01_square_pi_pulse_iwf": create_drive_wf()
            }
        }
        qmc.if_overrides = {
            "q01": 80e6,
            "q01_readout": 80e6
        }

        loc = measure_time_rabi()
    ```
    This will perform a time Rabi measurement, redefine the IF and waveforms of the drive and readout elements,
    and then execute the same measurement with the new settings.

    There is no need to create a new quantum machine or recompile in between measurements.
    """

    def setup(self, fun, *args, **kwargs):
        """
        Starts the measurement using a provided _program_id. Compilation only happens if the _program_id is None.
        """
        global _qmachine_context
        self.communicator["self_managed"] = False

        if _qmachine_context is None:
            raise RuntimeError("No quantum machine manager or quantum machine found. "
                               "Please use a context manager for precompiled measurements.")

        if _qmachine_context._program_id is None:
            _qmachine_context._program_id = _qmachine_context._qmachine.compile(fun(*args, **kwargs))
        if _qmachine_context.wf_overrides is not None:
            print(f"Using waveform overrides: {_qmachine_context.wf_overrides}")
            pending_job = _qmachine_context._qmachine.queue.add_compiled(_qmachine_context._program_id, overrides=_qmachine_context.wf_overrides)
        else:
            print("No waveform overrides provided, using default waveforms.")
            pending_job = _qmachine_context._qmachine.queue.add_compiled(_qmachine_context._program_id)

        if _qmachine_context.if_overrides is not None:
            print(f"Using IF overrides: {_qmachine_context.if_overrides}")
            for element, frequency in _qmachine_context.if_overrides.items():
                _qmachine_context._qmachine.set_intermediate_frequency(element, frequency)
            _qmachine_context.if_overrides = None
        else:
            print("No IF overrides provided, using default IFs.")

        job = pending_job.wait_for_execution()
        result_handles = job.result_handles

        self.communicator["result_handles"] = result_handles
        self.communicator["active"] = True
        self.communicator["counter"] = 0
        self.communicator["manager"] = _qmachine_context._qmachine_mgr
        self.communicator["qmachine"] = _qmachine_context._qmachine
        self.communicator["qmachine_id"] = _qmachine_context._qmachine.id

