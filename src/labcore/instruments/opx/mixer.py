"""Tools for using (IQ) mixers with the QM OPX.

Required packages/hardware:
- QM OPX incl python software
- SignalHound USB SA124B + driver (comes with qcodes)
"""
from functools import partial
from typing import List, Tuple, Optional, Any, Dict, Callable
from time import sleep
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from ..qcodes_drivers.SignalHound.Spike import Spike
from ..qcodes_drivers.SignalCore.SignalCore_sc5511a import SignalCore_SC5511A
from qcodes.instrument_drivers.rohde_schwarz.SGS100A import RohdeSchwarz_SGS100A

from qm import QuantumMachine, QuantumMachinesManager
from qm.qua import *

from .config import QMConfig
from dataclasses import dataclass


class MixerCalibration:
    """Class for performing IQ mixer calibration.

    We assume that we control the I and Q with a QM OPX, and monitor the output of the mixer with a
    SignalHound spectrum analyzer.
    Requires that independently a correctly specified configuration for the OPX is available.

    Parameters
    ----------
    lo_frq
        LO frequency in Hz
    if_frq
        IF frequency in Hz (can be positive or negative, depending on the sideband you want)
    analyzer
        SignalHound qcodes driver instance
    qm
        Quantum Machine instance (with config applied)
    mixer_name
        the name of mixer we're tuning, as given in the QM config
    element_name
        the name of the element thats playing the IQ waveform, as given in the QM config
    pulse_name
        the name of the (CW) pulse we're playing to tune the mixer, as given in the QM config
    """

    def __init__(self, lo_frq: float, if_frq: float, analyzer: Spike,
                 qm: QuantumMachine, mixer_name: str, element_name: str, pulse_name: str
                 ):

        self.lo_frq = lo_frq
        self.if_frq = if_frq
        self.analyzer = analyzer
        self.qm = qm
        self.mixer_name = mixer_name
        self.element_name = element_name
        self.pulse_name = pulse_name
        self.do_plot = True

        self.analyzer.mode('ZS')
        sleep(0.5)

    def play_wf(self) -> None:
        """Play an infinite loop waveform on the OPX.
        We're scaling the amplitude of the pulse used by 0.5.
        """
        with program() as const_pulse:
            with infinite_loop_():
                play(self.pulse_name * amp(0.5), self.element_name)

        _ = self.qm.execute(const_pulse)

    def setup_analyzer(self, f: float) -> None:
        """Set up the analyzer to measure at the given frequency `f` and sweep time.

        Signalhound driver is automatically put in zero-span mode when called.

        Parameters
        ----------
        f
            frequency to measure at, in Hz
        mode
            set the mode for the spectrum analyzer
        """
        self.analyzer.zs_fcenter(f)
        sleep(1.0)

    def measure_leakage(self) -> float:
        """Measure max. signal power at the LO frequency."""
        sleep(0.1)
        return self.analyzer.zs_power()

    def measure_upper_sb(self) -> float:
        """Measure max. signal power at LO frequency + IF frequency"""
        sleep(0.1)
        return self.analyzer.zs_power()

    def measure_lower_sb(self) -> float:
        """Measure max. signal power at LO frequency - IF frequency"""
        sleep(0.1)
        return self.analyzer.zs_power()

    @staticmethod
    def IQ_imbalance_correction(g, phi) -> List:
        """returns in the IQ mixer correction matrix as exepcted by the QM mixer config.

        Parameters
        ----------
        g
            relative amplitude imbalance between I and Q channels
        phi
            relative phase imbalance between I and Q channels
        """
        c = np.cos(phi)
        s = np.sin(phi)
        N = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
        return [float(N * x) for x in [(1 - g) * c, (1 + g) * s,
                                       (1 - g) * s, (1 + g) * c]]

    # TODO: properly implement the range for the COBYLA function to be bounded in
    def _optimize2d(self, func, initial_guess, ranges,
                    cb_options=None, maxit=200):

        """
        Performs minimization through COBYLA algorithm.
        It starts with an initial simplex (triangle in current 2D case).
        The initial simplex is a regular triangle centered around 'initial_guess'.
        The 'initial_ranges[0]' (which is side of square for 'scan2D') is the diameter of the circumscribing circle.

        Parameters
        ----------
        func
        initial_guess
        initial_ranges
        cb_options
        maxit

        Returns
        -------
        res (coordinates of the found minimum)

        """

        if cb_options is None:
            cb_options = dict()

        nit = 0

        x, y, z = [], [], []

        def cb(vec):
            val = func(vec)
            x.append(vec[0])
            y.append(vec[1])
            z.append(val)

            print(f'vector: {vec}, result: {val}, iteration: {len(y)}')

        try:
            res = minimize(func, initial_guess, method='COBYLA',
            callback=cb, options=dict(**cb_options, maxiter=maxit))
            nit = len(y)
             
        except KeyboardInterrupt:
            res = np.array([x[-1], y[-1]])
            print('optimization stopped by user')

        return res, nit

    def _scan2d(self, func, center, ranges, steps,
                title='', xtitle='', ytitle='', ztitle='Power'):

        xvals = center[0] + np.linspace(-ranges[0] / 2., ranges[0] / 2., steps)
        yvals = center[1] + np.linspace(-ranges[1] / 2., ranges[1] / 2., steps)
        xx, yy = np.meshgrid(xvals, yvals, indexing='ij')
        zz = np.ones_like(xx) * np.nan

        try:
            for k, x in enumerate(xvals):
                for l, y in enumerate(yvals):
                    p = func(np.array([x, y]))
                    zz[k, l] = p
                    print(f'{p:5.0f}', end='')
                print()

        except KeyboardInterrupt:
            print('scan stopped by user.')

        if self.do_plot:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            im = ax.pcolormesh(xx, yy, zz, shading='auto')
            cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
            ax.set_title(title + f" {datetime.now().isoformat()}", fontsize='small')
            cb.set_label(ztitle)
            ax.set_xlabel(xtitle)
            ax.set_ylabel(ytitle)
            plt.show()

        min_idx = np.argmin(zz.flatten())
        return np.array([xx.flatten()[min_idx], yy.flatten()[min_idx]], dtype=float)

    def lo_leakage(self, iq_offsets: np.ndarray) -> float:
        """Set the I and Q DC offsets and measure the leakage power (in dBm).

        Parameters
        ----------
        iq_offsets
            array with 2 elements (I and Q offsets), with dtype = float
        """
        self.qm.set_output_dc_offset_by_element(
            self.element_name, 'I', iq_offsets[0])
        self.qm.set_output_dc_offset_by_element(
            self.element_name, 'Q', iq_offsets[1])

        power = self.measure_leakage()
        return power

    def lo_leakage_scan(self, center: np.ndarray = np.array([0., 0.]),
                        ranges: Tuple = (0.5, 0.5), steps: int = 11) -> np.ndarray:
        """Scan the I and Q DC offsets and measure the leakage at each point.

        if `MixerCalibration.do_plot` is `True` (default), then this generates a live plot of this measurement.

        Parameters
        ----------
        center
            center coordinate [I_of, Q_of]
        ranges
            scan range on I and Q
        steps
            how many steps (will be used for both I and Q)
        Returns
        -------
        np.ndarray
            the I/Q offset coordinate at which the smallest leakage was found
        """
        self.setup_analyzer(self.lo_frq)
        res = self._scan2d(self.lo_leakage,
                           center=center, ranges=ranges, steps=steps,
                           title='Leakage scan', xtitle='I offset',
                           ytitle='Q offset')
        return res

    def optimize_lo_leakage(self, initial_guess: np.ndarray = np.array([0., 0.]),
                            ranges: Tuple[float, float] = (0.1, 0.1),
                            cb_options: Optional[Dict[str, Any]] = None):
        """Optimize the IQ DC offsets using COBYLA.

        The initial guess and ranges are used to specify the initial simplex
        for the COBYLA algorithm.
        initial guess is the starting point, and initial ranges are the distances
        along the two coordinates for the remaining two vertices of the initial simplex.

        Parameters
        ----------
        initial_guess
            x0 of the COBYLA algorithm, an array with two elements, for I and Q offset
        ranges
            distance for I and Q vectors to complete the initial simplex
        cb_options
            Options to pass to the `scipy.optimize.minimize(method='COBYLA')`.
            Will be passed via the `options` dictionary.
        """

        self.setup_analyzer(self.lo_frq)
        res, nit = self._optimize2d(self.lo_leakage,
                               initial_guess,
                               ranges=ranges,
                               cb_options=cb_options)
        return res, nit

    def sb_imbalance(self, imbalance: np.ndarray) -> float:
        """Set mixer imbalance and measure the upper SB power.

        Parameters
        ----------
        imbalance
            values for relative amplitude and phase imbalance

        Returns
        -------
        float
            upper SB power [dBm]

        """
        mat = self.IQ_imbalance_correction(imbalance[0], imbalance[1])
        self.qm.set_mixer_correction(
            self.mixer_name, int(self.if_frq), int(self.lo_frq),
            tuple(mat))
        return self.measure_upper_sb()

    def sb_imbalance_scan(self, center: np.ndarray = np.array([0., 0.]),
                          ranges: Tuple = (0.5, 0.5), steps: int = 11) -> np.ndarray:
        """Scan the relative amplitude and phase imbalance and measure the leakage at each point.

        if `MixerCalibration.do_plot` is `True` (default), then this generates a live plot of this measurement.

        Parameters
        ----------
        center
            center coordinate [amp imbalance, phase imbalance]
        ranges
            scan range on the two imbalances
        steps
            how many steps (will be used for both imbalances)

        Returns
        -------
        np.ndarray
            the imbalance coordinate at which the smallest leakage was found for the lower sideband
        """
        self.setup_analyzer(self.lo_frq - self.if_frq) # LO - IF
        res = self._scan2d(self.sb_imbalance,
                           center=center, ranges=ranges, steps=steps,
                           title='SB imbalance scan',
                           xtitle='g', ytitle='phi')
        return res

    def optimize_sb_imbalance(self, initial_guess: np.ndarray = np.array([0., 0.]),
                              ranges: Tuple[float, float] = (0.05, 0.05),
                              cb_options: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Optimize the mixer imbalances using COBYLA.

        The initial guess and ranges are used to specify the initial simplex
        for the COBYLA algorithm.
        initial guess is the starting point, and initial ranges are the distances
        along the two coordinates for the remaining two vertices of the initial simplex.

        Parameters
        ----------
        initial_guess
            x0 of the COBYLA algorithm, an array with two elements, for rel. amp and phase imbalance
        ranges
            distance for amp/phase imbalance vectors to complete the initial simplex
        cb_options
            Options to pass to the `scipy.optimize.minimize(method='COBYLA')`.
            Will be passed via the `options` dictionary.
        """
        self.setup_analyzer(self.lo_frq - self.if_frq) # LO - IF 
        res, nit = self._optimize2d(self.sb_imbalance,
                               initial_guess,
                               ranges=ranges,
                               cb_options=cb_options)
        return res, nit


@dataclass
class MixerConfig:
    #: Quantum machines config object
    qmconfig: QMConfig
    #: OPX address
    opx_address: str
    #: OPX port
    opx_port: str
    #: OPX cluster_name
    opx_cluster_name: str
    #: spectrum analyzer
    analyzer: Spike
    #: the LO for the mixer
    generator: SignalCore_SC5511A
    #: param that holds the IF
    if_param: Callable
    #: param holding the dc offsets
    offsets_param: Callable
    #: param holding the imbalances
    imbalances_param: Callable
    #: name of the mixer in the opx config
    mixer_name: str
    #: element we play a constant pulse on
    element_name: str
    #: name of the pulse we play
    pulse_name: str
    #: method for calibrating the mixer
    calibration_method: str = ' '
    #: power for the generator
    generator_power: Optional[float] = None
    #: options for scanning-based optimization, offsets
    offset_scan_ranges: Tuple[float, float] = (0.01, 0.01)
    offset_scan_steps: int = 11
    #: options for scanning-based optimization, imbalances
    imbalance_scan_ranges: Tuple[float, float] = (0.01, 0.01)
    imbalance_scan_steps: int = 11
    #: param that holds the LO frequency
    lo_param: Optional[Callable] = None
    #: parameter that holds the frequency
    frequency_param: Optional[Callable] = None
    # do you want to provide custom initial point?
    # (doesn't affect scan2D)
    opt2D_of_custom_init: bool = True
    opt2D_imb_custom_init: bool = True
    # do you want to do a larger scan (typically first calibration) or smaller scan (around already found point)?
    # (doesn't affect scan2D)
    opt2D_of_dia: str = 'large'
    opt2D_imb_dia: str = 'large'


def calibrate_mixer(config: MixerConfig,
                    offset_scan_ranges=None,
                    offset_scan_steps=None,
                    imbalance_scan_ranges=None,
                    imbalance_scan_steps=None,
                    calibrate_offsets=True,
                    calibrate_imbalance=True,
                    max_step_size=0.05):
    """
    Runs the entire mixer calibration for any mixer
    """
    print("Ensure that effective path lengths before I and Q of mixer are same.")
    print(f"Calibrating {config.mixer_name} by {config.calibration_method}...")

    # TODO: Should be configurable
    config.analyzer.zs_ref_level(-20)
    config.analyzer.zs_sweep_time(0.01)
    config.analyzer.zs_ifbw_auto(0)
    config.analyzer.zs_ifbw(1e5)

    # setup the generator frequency and its power
    if config.lo_param is not None:
        mixer_lo_freq = config.lo_param()
        config.generator.frequency(mixer_lo_freq)
    elif config.frequency_param is not None:
        mixer_lo_freq = config.frequency_param() - config.if_param() #RF - IF when measuring on the upper sideband
        config.generator.frequency(mixer_lo_freq)
    else:
        mixer_lo_freq = config.generator.frequency()

    # support for both SignalCore and R&S SGS
    if hasattr(config.generator, 'output_status'):
        config.generator.output_status(1)
    elif hasattr(config.generator, 'on'):
        config.generator.on()

    if config.generator_power is not None:
        config.generator.power(config.generator_power)

    if config.opx_cluster_name is not None:
        qmm = QuantumMachinesManager.QuantumMachinesManager(host=config.opx_address, 
                                                            port=None,
                                                            cluster_name=config.opx_cluster_name)
    else:
        qmm = QuantumMachinesManager.QuantumMachinesManager(host=config.opx_address, 
                                                            port=config.opx_port,)


    qm = qmm.open_qm(config.qmconfig(), close_other_machines=False)

    try:
        # initialize Mixer class object
        cal = MixerCalibration(mixer_lo_freq,
                               config.if_param(),
                               config.analyzer, qm,
                               mixer_name=config.mixer_name,
                               element_name=config.element_name,
                               pulse_name=config.pulse_name,
                               )

        # Call the appropriate calibration functions
        # calibrate voltage offsets for LO leakage
        if calibrate_offsets:
            offsets = config.offsets_param()
            cal.play_wf()
            if config.calibration_method == 'scanning':
                print("\nOffset calibration through: scanning \n")
                if offset_scan_steps is None:
                    offset_scan_steps = config.offset_scan_steps
                if offset_scan_ranges is None:
                    offset_scan_ranges = config.offset_scan_ranges

                print(f'Offsets: {offsets} Ranges: {offset_scan_ranges} \n')

                res_offsets = cal.lo_leakage_scan(
                    offsets,
                    ranges=offset_scan_ranges,
                    steps=offset_scan_steps,
                )
                offsets = res_offsets.tolist()
            else:
                print("\nOffset calibration through: COBYLA \n")

                if config.opt2D_of_custom_init is True:
                    pass
                else:
                    offsets = [0, 0]

                custom_of_range = config.offset_scan_ranges

                if config.opt2D_of_dia == 'large':
                    custom_of_range = [0.05, 0.05]
                elif config.opt2D_of_dia == 'small':
                    custom_of_range = [0.001, 0.001]
                elif config.opt2D_of_dia == 'custom':
                    pass

                print(f'Offsets: {offsets} Ranges: {custom_of_range} \n')

                res_offsets, nit = cal.optimize_lo_leakage(
                    offsets,
                    ranges=custom_of_range,
                    cb_options=dict(rhobeg=max_step_size, tol=1e-6, 
                                    disp=True, catol=1e-10)
                )

                if isinstance(res_offsets, np.ndarray):
                    offsets = res_offsets.tolist()
                elif res_offsets.success and nit < 200:
                    offsets = res_offsets.x.tolist()
                else:
                    print('Failed to converge. Use different initial values. \n')
                    return

            print(f'best values for offsets: {offsets} \n')
            config.offsets_param(offsets)
            print(f'verifying: {cal.lo_leakage(offsets)} \n')

        # calibrate imbalances for sideband suppression
        if calibrate_imbalance:
            imbalances = config.imbalances_param()
            cal.play_wf()
            if config.calibration_method == 'scanning':
                print("\nImbalance calibration through: scanning \n")
                if imbalance_scan_steps is None:
                    imbalance_scan_steps = config.imbalance_scan_steps
                if imbalance_scan_ranges is None:
                    imbalance_scan_ranges = config.imbalance_scan_ranges

                print(f'Imbalances: {imbalances} Ranges: {imbalance_scan_ranges} \n')

                res_imbalances = cal.sb_imbalance_scan(
                    imbalances,
                    ranges=imbalance_scan_ranges,
                    steps=imbalance_scan_steps,
                )
                imbalances = res_imbalances.tolist()
            else:
                print("\nOffset calibration through: COBYLA \n")

                if config.opt2D_imb_custom_init is True:
                    pass
                else:
                    if config.generator.IDN().get('vendor') == 'Rohde&Schwarz':
                        imbalances = [0, 1.57]
                    else:
                        imbalances = [0, 0]

                custom_imb_range = config.imbalance_scan_ranges

                if config.opt2D_imb_dia == 'large':
                    custom_imb_range = [0.05, 0.05]
                elif config.opt2D_imb_dia == 'small':
                    custom_imb_range = [0.001, 0.001]
                elif config.opt2D_imb_dia == 'custom':
                    pass

                print(f'Imbalances: {imbalances} Ranges: {custom_imb_range} \n')

                res_imbalances, nit = cal.optimize_sb_imbalance(
                    imbalances,
                    ranges=custom_imb_range,
                    cb_options=dict(rhobeg=max_step_size, tol=1e-6, 
                                    disp=True, catol=1e-10),
                )

                if isinstance(res_imbalances, np.ndarray):
                    imbalances = res_imbalances.tolist()
                elif res_imbalances.success and nit < 200:
                    imbalances = res_imbalances.x.tolist()

                else:
                    print('Failed to converge. Use different initial values. \n')
                    return

            print(f'best values for imbalance: {imbalances} \n')
            config.imbalances_param(imbalances)
            print(f'verifying: {cal.sb_imbalance(imbalances)} \n')

    finally:
        qm.close()

def mixer_of_step(config: MixerConfig, qm: QuantumMachine, di, dq):
    new_i = config.offsets_param()[0] + di
    new_q = config.offsets_param()[1] + dq
    qm.set_output_dc_offset_by_element(config.element_name, 'I', new_i)
    qm.set_output_dc_offset_by_element(config.element_name, 'Q', new_q)
    config.offsets_param([new_i, new_q])


def mixer_imb_step(config: MixerConfig, qm: QuantumMachine, dg, dp):
    new_g = config.imbalances_param()[0] + dg
    new_p = config.imbalances_param()[1] + dp
    if config.lo_param is not None:
        lof = config.lo_param()
    elif config.frequency_param is not None:
        lof = config.frequency_param() + config.if_param()
    else:
        lof = config.generator.frequency()

    qm.set_mixer_correction(config.mixer_name,
                            int(config.if_param()),
                            int(lof),
                            tuple(MixerCalibration.IQ_imbalance_correction(new_g, new_p)))
    config.imbalances_param([new_g, new_p])
