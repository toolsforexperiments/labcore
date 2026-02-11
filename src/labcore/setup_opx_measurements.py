"""general setup file for OPX measurements.

Use by importing and then configuring the options object.
"""

# this is to prevent the OPX logger to also create log messages (results in duplicate messages)
import os
os.environ['QM_DISABLE_STREAMOUTPUT'] = "1"

from typing import Optional, Callable
from dataclasses import dataclass
from functools import partial

from IPython.display import display
import ipywidgets as widgets

# FIXME: only until everyone uses the latest qm packages.
try: 
    from qm.QuantumMachinesManager import QuantumMachinesManager, QuantumMachine
except:
    from qm.quantum_machines_manager import QuantumMachinesManager
    from qm import QuantumMachine

from qm.qua import *

from instrumentserver.helpers import nestedAttributeFromString

from .instruments.opx.config import QMConfig
from .instruments.opx import sweep as qmsweep
from .instruments.opx.mixer import calibrate_mixer, MixerConfig, mixer_of_step, mixer_imb_step

from . import setup_measurements
from .setup_measurements import *

@dataclass
class Options(setup_measurements.Options):
    _qm_config: Optional[QMConfig] = None

    # this is implemented as a property so we automatically set the
    # options correctly everywhere else...
    @property
    def qm_config(self):
        return self._qm_config

    @qm_config.setter
    def qm_config(self, cfg):
        self._qm_config = cfg
        qmsweep.config = cfg

options = Options()
setup_measurements.options = options

@dataclass
class Mixer:
    config: MixerConfig
    qm: Optional[QuantumMachine] = None
    qmm: Optional[QuantumMachinesManager] = None

    def run_constant_waveform(self, **kwargs):
        """
        When using this with the octaves, the fields `cluster_name` and `octave` with their corresponding values need to be in the kwargs.
        """
        try:
            with program() as const_pulse:
                with infinite_loop_():
                    play('constant', self.config.element_name)
            if self.config.qmconfig.cluster_name is not None:
                qmm = QuantumMachinesManager(host=self.config.qmconfig.opx_address,
                                        port=None, 
                                        cluster_name=self.config.qmconfig.cluster_name,
                                        **kwargs)
            else:
                qmm = QuantumMachinesManager(host=self.config.qmconfig.opx_address,
                                        port=self.config.qmconfig.opx_port, **kwargs)
            self.qmm = qmm
            qm = qmm.open_qm(self.config.qmconfig(), close_other_machines=False)
            qm.execute(const_pulse)
            self.qm = qm
        except KeyError as e:
            message = None
            if len(kwargs) < 2:
                message = "Seems like no arguments were passed. " \
                "If you are using the octaves you need to pass the arguments 'cluster_name' and 'octave', try passing them as keyqord arguments and try again."
            raise AttributeError(message + f" Error raised was: {e}" )


    def step_of(self, di, dq):
        if self.qm is None:
            raise RuntimeError('No active QuantumMachine.')
        mixer_of_step(self.config, self.qm, di, dq)

    def step_imb(self, dg, dp):
        if self.qm is None:
            raise RuntimeError('No active QuantumMachine.')
        mixer_imb_step(self.config, self.qm, dg, dp)

def add_mixer_config(qubit_name, analyzer, generator, readout=False, element_to_param_map=None, **config_kwargs):
    """
    FIXME: add docu (@wpfff)
    TODO: make sure we document the meaning of `element_to_param_map`.

    contributor(s): Michael Mollenhaur

    arguments:
      qubit_name - string; name of the qubit listed in the parameter manager you are going to work with
      analyzer - instrument; instrument module for the spectrum analyzer used for the mixer
      generator - instrument; instrument module for the LO generator used for the mixer
      readout - boolean; whether you are calibrating the readout mixer for the specified qubit
      element_to_param_map - string; specifies whether to call the qubit or readout parameter manager values

    """
    element_name = qubit_name
    if readout is True:
        element_name += '_readout'

    if element_to_param_map is None:
        element_to_param_map = qubit_name

        if readout is True:
            element_to_param_map += '.readout'
    
    cfg = MixerConfig(
        qmconfig=options.qm_config,
        opx_address=options.qm_config.opx_address,
        opx_port=options.qm_config.opx_port,
        opx_cluster_name=options.qm_config.cluster_name,
        analyzer=analyzer,
        generator=generator,
        if_param=nestedAttributeFromString(options.parameters, f"{element_to_param_map}.IF"),
        offsets_param=nestedAttributeFromString(options.parameters, f"mixers.{element_to_param_map}.offsets"),
        imbalances_param=nestedAttributeFromString(options.parameters, f"mixers.{element_to_param_map}.imbalance"),
        mixer_name=f'{element_name}_IQ_mixer',
        element_name=element_name,
        pulse_name='constant',
        **config_kwargs
    )
    return Mixer(
        config=cfg,
    )


# A simple graphical mixer tuning tool
def mixer_tuning_tool(mixer):
    # widgets for dc offset tuning
    of_step = widgets.FloatText(description='dc of. step:', value=0.01, min=0, max=1, step=0.001)
    iup_btn = widgets.Button(description='I ^')
    idn_btn = widgets.Button(description='I v')
    qup_btn = widgets.Button(description='Q ^')
    qdn_btn = widgets.Button(description='Q v')

    def on_I_up(b):
        mixer.step_of(of_step.value, 0)

    def on_I_dn(b):
        mixer.step_of(-of_step.value, 0)

    def on_Q_up(b):
        mixer.step_of(0, of_step.value)

    def on_Q_dn(b):
        mixer.step_of(0, -of_step.value)

    iup_btn.on_click(on_I_up)
    idn_btn.on_click(on_I_dn)
    qup_btn.on_click(on_Q_up)
    qdn_btn.on_click(on_Q_dn)

    # widgets for imbalance tuning
    imb_step = widgets.FloatText(description='imb. step:', value=0.01, min=0, max=1, step=0.001)
    gup_btn = widgets.Button(description='g ^')
    gdn_btn = widgets.Button(description='g v')
    pup_btn = widgets.Button(description='phi ^')
    pdn_btn = widgets.Button(description='phi v')

    def on_g_up(b):
        mixer.step_imb(imb_step.value, 0)

    def on_g_dn(b):
        mixer.step_imb(-imb_step.value, 0)

    def on_p_up(b):
        mixer.step_imb(0, imb_step.value)

    def on_p_dn(b):
        mixer.step_imb(0, -imb_step.value)

    gup_btn.on_click(on_g_up)
    gdn_btn.on_click(on_g_dn)
    pup_btn.on_click(on_p_up)
    pdn_btn.on_click(on_p_dn)

    # assemble reasonably for display
    ofupbox = widgets.HBox([iup_btn, qup_btn])
    ofdnbox = widgets.HBox([idn_btn, qdn_btn])
    ofbox = widgets.VBox([of_step, ofupbox, ofdnbox])

    imbupbox = widgets.HBox([gup_btn, pup_btn])
    imbdnbox = widgets.HBox([gdn_btn, pdn_btn])
    imbbox = widgets.VBox([imb_step, imbupbox, imbdnbox])

    box = widgets.HBox([ofbox, imbbox])
    display(box)


run_measurement = partial(run_measurement, qmconfig=lambda: options.qm_config() if options.qm_config is not None else None)
