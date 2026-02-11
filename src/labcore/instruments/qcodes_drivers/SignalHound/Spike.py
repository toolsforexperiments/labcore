"""Basic driver to communicate with the SPIKE program through SCPI."""


__author__ = "Michael Mollenhauer, Abdullah Irfan"
__email__ = "mcm16@illinois.edu, irfan3@illinois.edu"


import logging

import numpy as np
import qcodes
from qcodes import (VisaInstrument, validators as vals)


class Spike(VisaInstrument):
    """
    Pfafflab SignalHound Driver using the qcodes package

    """

    def __init__(self, name, address=None, **kwargs):
        if address is None:
            raise Exception('TCP IP address needed')
        logging.info(__name__ + ' : Initializing instrument Spike')

        super().__init__(name, address, terminator='\n', **kwargs)

        # Checks and changes the mode
        self.add_parameter('mode',
                           get_cmd=':INSTRUMENT?',
                           set_cmd='INSTRUMENT {}',
                           vals=vals.Anything(),
                           get_parser=str,
                           )


        # Zero-span mode
        # Changes the reference level in zero span mode
        self.add_parameter('zs_ref_level',
                           get_cmd=':ZS:CAPTURE:RLEVEL?',
                           set_cmd=':ZS:CAPTURE:RLEVEL {}',
                           vals=vals.Numbers(),
                           get_parser=float,
                           unit='dB'
                           )

        # Changes the center frequency in zero span mode
        self.add_parameter('zs_fcenter',
                           get_cmd=':ZS:CAPTURE:CENTER?',
                           set_cmd=':ZS:CAPTURE:CENTER {}',
                           vals=vals.Numbers(),
                           get_parser=float,
                           unit='Hz'
                           )

        # Changes the sampling rate in zero span mode
        self.add_parameter('zs_sample_rate',
                           get_cmd=':ZS:CAPTURE:SRATE {}',
                           set_cmd=':ZS:CAPTURE:SRATE?',
                           vals=vals.Numbers(),
                           get_parser=float,
                           )

        # Changes the IF bandwidth in zero span mode, only works when AUTO is off
        self.add_parameter('zs_ifbw',
                           get_cmd=':ZS:CAPTURE:IFBWIDTH?',
                           set_cmd=':ZS:CAPTURE:IFBWIDTH {}',
                           vals=vals.Numbers(),
                           get_parser=float,
                           unit='Hz'
                           )

        # Enables the AUTO IF bandwidth option in zero span mode
        self.add_parameter('zs_ifbw_auto',
                           get_cmd=':ZS:CAPTURE:IFBWIDTH:AUTO?',
                           set_cmd=':ZS:CAPTURE:IFBWIDTH:AUTO {}',
                           vals=vals.Anything()
                           )

        # Changes the sweep time in zero span mode
        self.add_parameter('zs_sweep_time',
                           get_cmd=':ZS:CAPTURE:SWEEP:TIME?',
                           set_cmd=':ZS:CAPTURE:SWEEP:TIME {}',
                           vals=vals.Numbers(),
                           get_parser=float,
                           unit='s'
                           )

        self.add_parameter('zs_power',
                           get_cmd=self._measure_zs_power_dBm,
                           set_cmd=False,
                           unit='dBm')

        self.add_parameter('zs_iq_values',
                           get_cmd=self._measure_zs_iq_vals,
                           set_cmd=False,
                           unit='dBm^.5')

        # setting defaults
        self.mode('ZS')

    def _measure_zs_iq_vals(self):
        IQ_table = np.array(self.ask(':FETCH:ZS? 1').split(',')).astype(float).reshape(-1, 2)
        return IQ_table

    def _measure_zs_power_dBm(self):
        IQ_table = np.array(self.ask(':FETCH:ZS? 1').split(',')).astype(float).reshape(-1, 2)
        power = (IQ_table[:, 0] ** 2 + IQ_table[:, 1] ** 2).mean()
        return 10 * np.log10(power)