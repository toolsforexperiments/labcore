""" Tools to enable the usage of QICK in the Sweep framework.

Required packages/hardware:
- FPGA configured with QICK
- QICK package

Example usage of this module:
@QickBoardSweep(
    independent("freqs"),
    ComplexQICKData("signal",
                    depends_on=["freqs"],
                    i_data_stream='I', q_data_stream='Q'),
)
class SingleToneSpecstroscopyProgram(RAveragerProgram):
""" 

import numpy as np
from collections.abc import Iterable, Generator
from dataclasses import dataclass

from labcore.measurement import independent, dependent, DataSpec
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
        # TODO: How can I extend this to multiple measurement rounds? (e.g. active reset)

        # Run the program
        data = self.communicator["qick_program"].acquire(self.config.soc)

        # Parse data according to annotated DataSpecs
        return_data = {}  # tuple to be yielded
        if len(data) == 2:  # No QICK-internal dependent variables
            ds = self.specs[0]
            return_data[ds.name] = data[0][0] + 1j*data[1][0]
        else:
            measIdx = 0  # To specify the index of the measured data to return
            sweepIdx = 0  # To specify the index of the sweep variable to return
            isNot1D = isinstance(self.communicator["qick_program"].get_expt_pts(), list)
            for ds in self.specs:
                if isinstance(ds, ComplexQICKData):
                    return_data[ds.name] = data[1][measIdx][0] + 1j*data[2][measIdx][0]
                    measIdx += 1
                elif isNot1D:
                    return_data[ds.name] = self.communicator["qick_program"].get_expt_pts()[sweepIdx]
                    sweepIdx += 1
                else:
                    return_data[ds.name] = self.communicator["qick_program"].get_expt_pts()
                    sweepIdx += 1

            # Reformat the independent variables
            # Independent (sweep) variables have to be reformatted such that xArray and plottr can
            # correctly recognize the axis being swept. Without reformatting the swept variable,
            # the program won't be able to correctly set up the axis being swept.
            shapeIdx = 0
            for ds in self.specs:
                if isinstance(ds, ComplexQICKData):
                    return_data[ds.name] = np.transpose(return_data[ds.name])
                else:
                    dimList = [1] * sweepIdx
                    dimList[shapeIdx] = len(return_data[ds.name])
                    dimTuple = tuple(dimList)
                    diffList = [1] * sweepIdx
                    diffIdx = 0
                    for dp in self.specs:
                        if isinstance(dp, ComplexQICKData):
                            pass
                        else:
                            if diffIdx != shapeIdx:
                                diffList[diffIdx] = len(return_data[dp.name])
                            diffIdx += 1
                    diffTuple = tuple(diffList)
                    return_data[ds.name] = np.reshape(return_data[ds.name], dimTuple)
                    return_data[ds.name] = np.tile(return_data[ds.name], diffTuple)

                    shapeIdx += 1

        yield return_data
