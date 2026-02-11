# -*- coding: utf-8 -*-
"""
A driver to control the Keysight VNA P9374A using pyVISA and qcodes

@author: Hatlab: Ryan Kaufman; UIUC: Wolfgang Pfaff

"""

import logging
from typing import Any, Union, Dict, List, Tuple

import numpy as np
from qcodes import (VisaInstrument, Parameter, ParameterWithSetpoints, InstrumentChannel, validators as vals)
from qcodes.instrument.parameter import ParamRawDataType

"""
Some basic concepts for this VNA:

Measurement: 
    The VNA object that actually holds the information of measurement of the device. It will not automatically 
    show up on the VNA software interface when created.  
Trace: 
    The VNA object that can be shown on the VNA software, needs to link to a certain measurement.  
    As a working definition in our setting, we can think the trace and measurement are equivalent as there are always 
    created together.     
Channel:
    One channel can hold many traces. So that the common settings (frequency range, number of points, etc...) 
    can be applied to all the measurement. In this driver we are only use one channel. More needs to do if we want to
    support multiple channel functions. 
Window:
    Place to show the selected traces on the VNA software. 

The settings that are independent to each trace/measurement (S-parameters, trace data) will be method/parameters for 
Trace object while common settings for the whole channel will be method/parameters for the Keysight_P9374A_SingleChannel
object. 

For example:
    vna.trace_1.s_parameter()
    get the S-parameter for the first trace/measurement. 

    vna.fstart()
    get the start frequency for the whole channel. 

"""


class SParameterData(Parameter):
    """
    SParameterData(Parameter)

    Qcodes parameter that can be used to set/get the S-parameter of this measurement.

    """

    def __init__(self, trace_number: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trace_number = trace_number

    def get_raw(self) -> ParamRawDataType:
        _, traces, _ = self.root_instrument.get_existing_traces()
        if self.instrument.npts() == 0 or self.trace_number not in traces:
            return 'Trace is not on'

        data = self.root_instrument.ask(f":CALC1:MEAS{self.trace_number}:PAR?").strip('"')
        return data

    def set_raw(self, S_parameter: str = 'S21') -> None:
        _, traces, _ = self.root_instrument.get_existing_traces()
        if self.instrument.npts() == 0 or self.trace_number not in traces:
            return 'Trace is not on'

        self.root_instrument.write(f":CALC1:MEAS{self.trace_number}:PAR {S_parameter}")


class FrequencyData(Parameter):
    """
    FrequencyData(Parameter)

    Qcodes parameter that can be used to get the frequency data of this measurment.
    """

    def __init__(self, trace_number: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trace_number = trace_number

    def get_raw(self) -> ParamRawDataType:
        _, traces, _ = self.root_instrument.get_existing_traces()
        if self.instrument.npts() == 0 or self.trace_number not in traces:
            return np.array([])

        data = self.root_instrument.ask(f"CALC:MEAS{self.trace_number}:X?")
        return np.array(data.split(',')).astype(float)


class TraceData(ParameterWithSetpoints):
    """
    TraceData(ParameterWithSetpoints)

    Qcodes ParameterWithSetpoints that can be used to get the data of this measurement.
    VNA will be set to polar format "POL" to acquire data by default.
    If the VNA is in a different measurement format, it will be reset to that format after measurement, other wise VNA
    will remain in format of self.data_fmt after the measurement.

    """

    def __init__(self, trace_number: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trace_number = trace_number
        self.data_fmt = "POL"

    def get_raw(self) -> ParamRawDataType:
        _, traces, _ = self.root_instrument.get_existing_traces()
        if self.instrument.npts() == 0 or self.trace_number not in traces:
            return np.array([])

        # get the values of relevant parameters before taking the trace
        prev_fmt = None
        if self.data_fmt is not None:
            prev_fmt = self.root_instrument.ask(f"CALC:MEAS{self.trace_number}:FORM?")
            self.root_instrument.write(f"CALC:MEAS{self.trace_number}:FORM {self.data_fmt}")

        # Code will check if VNA average trpe is SWEEP.
        # If not a type error will be raised.
        try:
            prev_trigger_source, prev_sweep_mode, prev_averaging = self.root_instrument.average()
            data = self.root_instrument.ask(f"CALC:MEAS{self.trace_number}:DATA:FDATA?")
            # set relevant parameters back to their old values
            self.root_instrument.trigger_source(prev_trigger_source)
            self.root_instrument.sweep_mode(prev_sweep_mode)
            self.root_instrument.averaging(prev_averaging)
            if prev_fmt is not None:
                self.root_instrument.write(f"CALC:MEAS{self.trace_number}:FORM {prev_fmt}")

            # process complex data correctly
            data = np.array(data.split(',')).astype(float)
            if self.data_fmt in ['POL'] and data.size % 2 == 0:
                data = data.reshape((int(data.size / 2), 2))
                data = data[:, 0] + 1j * data[:, 1]
            return data

        except Exception as e:
            print(f"Data taking failed. {type(e)}: {e.args}")
            return np.zeros(self.instrument.npts())


class Trace(InstrumentChannel):
    """
    A Qcode InstrumentChannel creates from the parameter instrument: Keysight_P9374A_SingleChannel.

    This InstrumentChannel contains all the parameters and functions that are unique for each different
    trace/measurement.
    """

    def __init__(self, parent: "Keysight_P9374A_SingleChannel", number: int, name: str, **kwargs: Any):
        self._number = number
        super().__init__(parent, name=name, **kwargs)

        self.add_parameter(
            name='npts',
            unit='',
            get_cmd=self._get_npts,
            docstring='number of points in the trace',
        )

        self.add_parameter(
            name='frequency',
            unit='Hz',
            parameter_class=FrequencyData,
            trace_number=self._number,
            vals=vals.Arrays(shape=(self.npts.get_latest,)),
            snapshot_exclude=True,
        )

        self.add_parameter(
            name='data',
            unit='',
            setpoints=(self.frequency,),
            parameter_class=TraceData,
            trace_number=self._number,
            vals=vals.Arrays(shape=(self.npts.get_latest,),
                             valid_types=(np.floating, np.complexfloating)),
            snapshot_exclude=True,
        )

        self.add_parameter(
            name='s_parameter',
            unit='',
            parameter_class=SParameterData,
            trace_number=self._number,
            vals=vals.Enum('S11', 'S12', 'S21', 'S22'),
            get_parser=str
        )

    def _get_npts(self):
        return len(self._get_xdata())

    def _get_xdata(self) -> np.ndarray:
        _, traces, _ = self.root_instrument.get_existing_traces()
        if self._number not in traces:
            return np.array([])
        data = self.ask(f"CALC:MEAS{self._number}:X?")
        return np.array(data.split(',')).astype(float)


class Keysight_P9374A_SingleChannel(VisaInstrument):
    """
    This is a very simple driver for the Keysight_P9374A Vector Network Analyzer
    Performs basic manipulations of parameters and data acquisition

    Note: this version does not include a way of averaging via a BUS trigger

    """

    def __init__(self, name, address=None, **kwargs):

        """
        Initializes the Keysight_P9374A, and communicates with the wrapper.

        Input:
          name (string)    : name of the instrument
          address (string) : GPIB address
          reset (bool)     : resets to default values, default=False
        """
        if address is None:
            raise Exception('TCP IP address needed')
        logging.info(__name__ + ' : Initializing instrument Keysight PNA')

        super().__init__(name, address, terminator='\n', **kwargs)

        self.write('CALC1:PAR:MNUM 1')  # sets the active msmt to the first channel/trace

        # Add in parameters
        self.add_parameter('fstart',
                           get_cmd=':SENS1:FREQ:STAR?',
                           set_cmd=':SENS1:FREQ:STAR {}',
                           vals=vals.Numbers(),
                           get_parser=float,
                           unit='Hz'
                           )
        self.add_parameter('fstop',
                           get_cmd=':SENS1:FREQ:STOP?',
                           set_cmd=':SENS1:FREQ:STOP {}',
                           vals=vals.Numbers(),
                           get_parser=float,
                           unit='Hz'
                           )
        self.add_parameter('fcenter',
                           get_cmd=':SENS1:FREQ:CENT?',
                           set_cmd=':SENS1:FREQ:CENT {}',
                           vals=vals.Numbers(),
                           get_parser=float,
                           unit='Hz'
                           )
        self.add_parameter('fspan',
                           get_cmd=':SENS1:FREQ:SPAN?',
                           set_cmd=':SENS1:FREQ:SPAN {}',
                           vals=vals.Numbers(),
                           get_parser=float,
                           unit='Hz'
                           )
        self.add_parameter('rfout',
                           get_cmd=':OUTP?',
                           set_cmd=':OUTP {}',
                           vals=vals.Ints(0, 1),
                           get_parser=int
                           )
        self.add_parameter('num_points',
                           get_cmd=':SENS1:SWE:POIN?',
                           set_cmd=':SENS1:SWE:POIN {}',
                           vals=vals.Ints(1, 100000),
                           get_parser=int
                           )
        self.add_parameter('ifbw',
                           get_cmd=':SENS1:BWID?',
                           set_cmd=':SENS1:BWID {}',
                           vals=vals.Numbers(10, 1.5e6),
                           get_parser=float)
        self.add_parameter('power',
                           get_cmd=":SOUR1:POW?",
                           set_cmd=":SOUR1:POW {}",
                           unit='dBm',
                           get_parser=float,
                           vals=vals.Numbers(-85, 20)
                           )
        self.add_parameter('power_start',
                           get_cmd=':SOUR1:POW:STAR?',
                           set_cmd=':SOUR1:POW:STAR {}',
                           unit='dBm',
                           get_parser=float,
                           vals=vals.Numbers(-85, 10)
                           )
        self.add_parameter('power_stop',
                           get_cmd=':SOUR:POW:STOP?',
                           set_cmd=':SOUR1:POW:STOP {}',
                           unit='dBm',
                           get_parser=float,
                           vals=vals.Numbers(-85, 10)),
        self.add_parameter('averaging',
                           get_cmd=':SENS1:AVER?',
                           set_cmd=':SENS1:AVER {}',
                           get_parser=int,
                           vals=vals.Ints(0, 1)
                           )
        # TODO: this throws an error currently.
        # self.add_parameter('average_trigger',
        #                    get_cmd=':TRIG:AVER?',
        #                    set_cmd=':TRIG:AVER {}',
        #                    get_parser=int,
        #                    vals=vals.Ints(0, 1)
        #                    )

        self.add_parameter('avg_num',
                           get_cmd='SENS1:AVER:COUN?',
                           set_cmd='SENS1:AVER:COUN {}',
                           vals=vals.Ints(1),
                           get_parser=int
                           )
        self.add_parameter('avg_type',
                           get_cmd='SENS1:AVER:MODE?',
                           set_cmd='SENS1:AVER:MODE {}',
                           vals=vals.Enum('POIN', 'SWEEP'),
                           get_parser=str
                           )
        self.add_parameter('phase_offset',
                           get_cmd='CALC1:CORR:OFFS:PHAS?',
                           set_cmd='CALC1:CORR:OFFS:PHAS {}',
                           get_parser=float,
                           vals=vals.Numbers())
        self.add_parameter('electrical_delay',
                           get_cmd='CALC1:CORR:EDEL:TIME?',
                           set_cmd='CALC1:CORR:EDEL:TIME {}',
                           unit='s',
                           get_parser=float,
                           vals=vals.Numbers()
                           )
        self.add_parameter('trigger_source',
                           get_cmd='TRIG:SOUR?',
                           set_cmd='TRIG:SOUR {}',
                           vals=vals.Enum('IMM', 'EXT', 'MAN')
                           )
        self.add_parameter('sweep_mode',
                           get_cmd='SENS1:SWE:MODE?',
                           set_cmd='SENS1:SWE:MODE {}',
                           vals=vals.Enum('HOLD',
                                          'CONT',
                                          'GRO',
                                          'SING'))
        self.add_parameter('trigger_mode',
                           get_cmd='SENS:SWE:TRIG:MODE?',
                           set_cmd='SENS:SWE:TRIG:MODE {}',
                           vals=vals.Enum('CHAN', 'SWE', 'POIN', 'TRAC')
                           )
        self.add_parameter('trform',
                           get_cmd=':CALC1:FORM?',
                           set_cmd=':CALC1:FORM {}',
                           vals=vals.Enum('MLOG', 'PHAS',
                                          'GDEL',
                                          'SCOM', 'SMIT', 'SADM',
                                          'POL', 'MLIN',
                                          'SWR', 'REAL', 'IMAG',
                                          'UPH', 'PPH', 'SLIN', 'SLOG', )
                           )
        self.add_parameter('math',
                           get_cmd=':CALC1:MATH:FUNC?',
                           set_cmd=':CALC1:MATH:FUNC {}',
                           vals=vals.Enum('ADD', 'SUBT', 'DIV', 'MULT', 'NORM')
                           )
        self.add_parameter('sweep_type',
                           get_cmd=':SENS1:SWE:TYPE?',
                           set_cmd=':SENS1:SWE:TYPE {}',
                           vals=vals.Enum('LIN', 'LOG', 'SEGM', 'POW')
                           )
        self.add_parameter('correction',
                           get_cmd=':SENS1:CORR:STAT?',
                           set_cmd=':SENS1:CORR:STAT {}',
                           get_parser=int)
        self.add_parameter('smoothing',
                           get_cmd=':CALC1:SMO:STAT?',
                           set_cmd=':CALC1:SMO:STAT {}',
                           get_parser=float
                           )
        self.add_parameter('sweep_time',
                           get_cmd=':SENS1:SWE:TIME?',
                           set_cmd=None,  # generally just adjust ifbw and number of pts to change it,
                           get_parser=float,
                           unit='s'
                           )

        for i in range(1, 17):
            trace = Trace(self, number=i, name=f"trace_{i}")
            self.add_submodule(f"trace_{i}", trace)

        self.connect_message()

    def clear_all_traces(self):
        """remove all currently defined traces."""
        self.write("CALC:MEAS:DEL:ALL")

    def get_existing_traces_by_channel(self) -> Dict[int, List[Tuple[int, str]]]:
        """Returns all currently available traces.
        Assumes that traces/measurements do not have custom names not ending with the
        measurement number.

        Returns
            A dictionary, with keys being the channel indices that have traces in them.
            values are tuples of trace/measurement number and parameter measured.
        """
        ret = {}
        for i in range(1, 9):
            traces = self.ask(f"CALC{i}:PAR:CAT:EXT?").strip('"')
            if traces == "NO CATALOG":
                continue
            else:
                ret[i] = []
            traces = traces.split(',')
            names = traces[::2]
            params = traces[1::2]
            for n, p in zip(names, params):
                ret[i].append((int(n.split('_')[-1]), p))
        return ret

    def get_existing_traces(self) -> Tuple[List[int], List[int], List[str]]:
        """
        Return three lists, with one item per current trace: channel, trace/measurement number, parameter
        """
        chans, numbers, params = [], [], []
        trace_dict = self.get_existing_traces_by_channel()
        for chan, traces in trace_dict.items():
            for number, param in traces:
                chans.append(chan)
                numbers.append(number)
                params.append(param)
        return chans, numbers, params

    def get_sweep_data(self):
        """
        Gets stimulus data in displayed range of active measurement, returns array
        Will return different data depending on sweep type.

        For example:
            power sweep: 1xN array of powers in dBm
            frequency sweep: 1xN array of freqs in Hz
        Input:
            None
        Output:
            sweep_values (Hz, dBm, etc...)
        """
        logging.info(__name__ + ' : get stim data')
        strdata = str(self.ask(':SENS1:X:VAL?'))
        return np.array(list(map(float, strdata.split(','))))

    def data_to_mem(self):
        """
        Calls for data to be stored in memory
        """
        logging.debug(__name__ + ": data to mem called")
        self.write(":CALC1:MATH:MEM")

    def remove_trace(self, number: int):
        """
        Remove selected trace

        Note that when removing a new trace, vna will restart average.
        """
        _, traces, _ = self.get_existing_traces()
        if number not in traces:
            print('Trace does not exist. Nothing happens.')
        else:
            logging.debug(__name__ + f": remove trace{number}")
            self.write(f"CALC1:MEAS{number}:DEL")
            print('Trace is successfully removed.')

    def add_trace(self, number: int = 1, s_parameter: str = "S21"):
        """
        Adds a trace with a specific s_parameter

        Note that when adding a new trace, vna will restart average.
        """
        _, traces, _ = self.get_existing_traces()
        if number in traces:
            print('Trace exist. Please use another trace number or remove the current one with remove_trace(number).')
        else:
            logging.debug(__name__ + f": add trace{number} with S-parameter {s_parameter}")
            self.write(f"CALC1:MEAS{number}:DEF '{s_parameter}'")
            self.write(f"DISP:MEAS{number}:FEED 1")  # always show this trace in the window 1 (FEED number).
            print('Trace is successfully created.')

    # TODO: add timout protection
    def average(self) -> Tuple[str, str, int]:
        """
        Do the average self.avg_num() times.

        Read the trigger settings (trigger source and mode) before doing the average and return them.
        During the average the VNA will be in manual trigger source with single trigger mode.
        A single trigger signal is generated with 'INIT:IMM' for self.avg_num() times to complete the whole average
        process.

        Note that after the average the VNA will remain in the trigger settings for the average process.
        This is to give time for user to use another command to take the data from the trace/measurement.

        The trace.data() will automatically take the old settings and put them back. But you can also just take the
        return values of this function and reset by yourself.

        Will check if VNA average type is in SWEEP. If not, a type error will be raised.
        """

        if self.avg_type() == 'POIN':
            raise TypeError(
                'VNA average type is set to POINT, neeed to be SWEEP. Use vna.avg_type() function to change')
        else:
            prev_trigger_source = self.trigger_source()
            prev_sweep_mode = self.sweep_mode()
            prev_averaging = self.averaging()
            # The following trigger settings are necessary for VNA to take the average
            self.trigger_source('MAN')
            self.sweep_mode('SING')
            self.averaging(1)
            self.write("SENS:AVER:CLE")  # does not apply to point averaging
            for i in range(self.avg_num()):
                self.write('INIT:IMM')
                averaged = 0
                while averaged == 0:
                    averaged = self.ask("*OPC?")
            print('Average completed')

        return prev_trigger_source, prev_sweep_mode, prev_averaging

    def clear_averages(self) -> None:
        """Reset averaging and wait for new trigger to start over."""
        self.write('SENS:AVER:CLE')
