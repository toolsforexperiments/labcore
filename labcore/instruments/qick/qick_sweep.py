""" Tools to enable the usage of QICK in the Sweep framework.

Required packages/hardware:
- FPGA configured with QICK
- QICK package

Example usage of this module:
>>> @QickBoardSweep(
>>>     independent("freqs"),
>>>     ComplexQICKData("signal",
>>>                     depends_on=["freqs"],
>>>                     i_data_stream='I', q_data_stream='Q'),
>>> )
>>> class SingleToneSpecstroscopyProgram(RAveragerProgram):
""" 

import numpy as np
from collections.abc import Iterable, Generator
from dataclasses import dataclass

from labcore.measurement import independent, dependent
from labcore.measurement.sweep import AsyncRecord
from labcore.measurement.record import make_data_spec

# config has to be set by the users of the program.
# Example:
# conf = QickConfig(paras=params)
# qick_sweep.config = conf
config = None

@dataclass
class ComplexQICKData(DataSpec):
    i_data_stream: str = 'I'
    q_data_stream: str = 'Q'


class QickBoardSweep(AsyncRecord):
    """
    Decorator class to communicate with QICK in the Sweeping framework.
    """
    def __init__(self, *specs, **kwargs):
        """
        Initialize the decorator class by saving incoming DataSpec variables.
        """
        self.communicator = {}
        self.specs = []
        for s in specs:
            spec = make_data_spec(s)
            self.specs.append(spec)

    def setup(self, func, *args, **kwargs):
        """
        Setup a QICK program.
        """
        # Checks that the config is not None
        if config is None:
            raise Exception("QickSweep: config is not set")

        self.config = config
        conf = config.config()
        qick_program = func(soccfg=conf[0], cfg=conf[1])
        self.communicator["qick_program"] = qick_program

    def collect(self, *args, **kwargs):
        """
        Get the measurement data.
        Note that one can overload the given acquire function if one needs
        to perform other specific tasks. e.g. Needs to plot each non-averaged 
        points in the I-Q plane for a readout fidelity experiment.
        Assumptions
        * Given DataSpecs are either independent, dependent, or ComplexQICKData.
        """

        # Run the program
        data = self.communicator["qick_program"].acquire(self.config.soc)
        
        # Parse data according to annotated DataSpecs
        return_data = ()  # tuple to be yielded

        if len(self.specs) != len(data):
            NotImplementedError("DataSpecs and returned data have different lengths.")

        for i, ds, da in enumerate(zip(self.specs, data)):
            if isinstance(ds, independent) or isinstance(ds, dependent):
                return_data.append(da)
            elif isinstance(ds, ComplexQICKData):
                i_data = da[0]
                q_data = da[1]

                if (i_data is not None) and (q_data is not None):
                    return_data.append(i_data + 1j*q_data) 
                else:
                    NotImplementedError("ComplexQICKData requires both I and Q data.")

        yield return_data


        
