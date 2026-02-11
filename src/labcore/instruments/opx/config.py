import os
import logging
import json
from typing import Dict, Any, Optional

import numpy as np

from instrumentserver.helpers import nestedAttributeFromString

from .machines import close_my_qm

logger = logging.getLogger(__name__)

# FIXME: Docstring incomplete
class QMConfig:
    """
    Base class for a QMConfig class. The purpose of this class is to implement the real time changes of the
    parameter manager with the OPX. We do this to always have the most up-to-date parameters from the
    parameter manager and integration weights (which depend on the parameters in the parameter manager).

    By default, when a new config is generated this class will close any open QuantumMachines that are using the same
    controllers that this config uses. To not do this pass False to close_other_machines in config().

    The user should still manually write the config dictionary used for the specific physical setup that the
    measurement is going to be performed but a few helper methods are implemented in the base class: two helper
    methods to write integration weights and a method that creates and adds the integration weights to the config dict.

    If the constructor is overriden the new constructor should call the super constructor to pass the parameter manager
    instance used.

    If the constructor is overriden because the parameter manager is not being used, the method config(self) also needs
    to be overriden.

    To have the integration weights added automatically into the config dict, you have to implement config_(self).
    config_(self) should return the python dictionary without the integration weights in it.
    If this is the case, the already implemented config(self) method will add the integration weights when called and
    return the python dictionary with the integration weights added.

    The add_integration_weights will go through the items in the pulses dictionary and add weights to any
    pulse that has the following characteristics:
        * The key of the pulse starts with the str 'readout'. If the first 7 characters of the key are not 'readout'
          that pulse will be ignored.
        * It needs to have the string '_pulse', present in its key. anything that comes before '_pulse' is
          as the name of the pulse, anything afterwards gets ignored.
          e.g. If my pulse name is: 'readout_short_pulse', 'readout_short' is taken as a unique pulse that requires
          integration weights. If another pulse exists called: 'readout_short_pulse_something_else'
          add_integration_weights will add the same integration weights as it did to 'readout_short'.
          This is useful if you have 2 different pules of the same length that need the same integration weights.
        * For each unique pulse there needs to be a parameter in the parameter manager with the unique pulse name
          followed by len. This is where the length of the pulse will be taken to get the
          integration weights. An underscore ('_') in the pulse name will be interpreted as a dot ('.') in the
          parameter manager.
          e.g. If I have a pulse with name 'readout_short_pulse', the pulse name is 'readout_short' and there should
          be a parameter in the parameter manager called: 'readout.short.len' with the pulse length in it.
    Any pulse that does not fulfil any of the three requirements will be ignored and will not have
    integration weights.

    3 integration weights will be created for each unique pulse, a flat integration weights
    for a full demodulation, flat integration weights for a sliced demodulation and weighted integration weights.
    The weights for the weighted integration will be loaded from the calibration files folder.
    If no weights are found in the calibration folder for a pulse, flat ones will be used instead.
    If old integration weights are found in the file, they will be deleted.

    :param params: The instance of the parameter manager where the length of the pulses are stored.
    :param opx_address: The address of the OPX where the config is going to get used.
    :param opx_port: The port of the OPX where the config is going to get used.
    """

    def __init__(self, params, opx_address: Optional[str] = None, opx_port: Optional[int] = None, cluster_name: Optional[str] = None,
                 octave=None) -> None:
        self.params = params
        self.opx_address = opx_address
        self.opx_port = opx_port
        self.octave = octave
        self.cluster_name = cluster_name

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        return self.config()

    def config(self, close_other_machines: bool=False) -> Dict[str, Any]:
        """
        Creates the config dictionary.

        :param close_other_machines: If True, closes any currently open qm in the opx that uses the same controllers
            that this config is using.
        :returns: The config dictionary.
        """
        original = self.config_()
        conf_with_weights = self.add_integration_weights(original)
        with_random_wf = self.add_random_waveform(conf_with_weights)
        if close_other_machines:
            if self.opx_port is None or self.opx_address is None:
                logger.warning(f'opx_port or opx_adress are empty, cannot close qm.')
            else:
                close_my_qm(with_random_wf, self.opx_address, self.opx_port)
        return with_random_wf

    def config_(self) -> Dict[str, Any]:
        raise NotImplementedError

    def add_random_waveform(self, conf):
        """
        Adds a random waveform to the config dictionary. This is needed becaouse of a bug in the qm code that does not
        close open QuantumMachines if the configuration is exactly the same.
        """
        config_dict = conf.copy()
        config_dict['waveforms']['random_wf'] = {'type': 'constant',
                                                 'sample': np.random.rand()*0.1}
        return config_dict

    def add_integration_weights(self, conf):
        """
        Automatically add integration weights to the config dictionary. See module docstring for further explanation on
        the rules for this to work.
        """
        integration_weights_file = 'calibration_files/integration_weights.json'

        # Changes to True if old integration weights are found
        deleted_weights = False

        # Used to not repeat missing file warning
        no_file_warning = False
        config_dict = conf.copy()
        pulses = {}

        if os.path.exists(integration_weights_file):
            with open(integration_weights_file) as json_file:
                loaded_weights = json.load(json_file)
        else:
            loaded_weights = None

        # Go throguh pulses and check if they should have integration weights
        for key in config_dict['pulses'].keys():
            if 'readout_' in key and '_pulse' in key:
                path_to_param = key.split('_')
                readout_pulse_name = ''
                
                # find the parameter name as it should be in the parameter manager.
                # note: it may be nested under something else!
                for k in path_to_param:
                    if k[:5] == 'pulse':
                        break  # found it! continue...
                    else:
                        # add all the stuff it's nested in...
                        if len(readout_pulse_name) > 0:
                            readout_pulse_name += f'_{k}'
                        else:
                            readout_pulse_name += f"{k}"
                
                pulse = readout_pulse_name  # don't ask... too lazy to find all instances.
                readout_param_name = readout_pulse_name.replace('_', '.')
                len_param_name = readout_param_name + '.len'

                # str with the name of the length of the pulse in the param manager
                if self.params.has_param(len_param_name):
                    if readout_pulse_name not in pulses.keys():
                        pulse_len = nestedAttributeFromString(self.params, len_param_name)()

                        # Using the old integration weights style for the sliced weights because the OPX currently
                        # raises an exception when using the new ones.
                        flat = [(0.2, pulse_len)]
                        minus_flat = [(-0.2, pulse_len)]
                        flat_sliced = [0.2] * int(pulse_len//4)
                        mflat_sliced = [-0.2] * int(pulse_len//4)
                        empty = [(0.0, pulse_len)]
                        empty_sliced = [0.0] * int(pulse_len//4)

                        pulses[pulse] = {}
                        pulses[pulse][pulse + '_cos'] = {
                            'cosine': flat,
                            'sine': empty
                        }
                        pulses[pulse][pulse + '_sin'] = {
                            'cosine': empty,
                            'sine': flat
                        }
                        pulses[pulse][pulse + '_minus_sin'] = {
                            'cosine': empty,
                            'sine': minus_flat
                        }
                        pulses[pulse][pulse + '_sliced_cos'] = {
                            'cosine': flat_sliced,
                            'sine': empty_sliced
                        }
                        pulses[pulse][pulse + '_sliced_sin'] = {
                            'cosine': empty_sliced,
                            'sine': flat_sliced
                        }
                        pulses[pulse][pulse + '_sliced_minus_sin'] = {
                            'cosine': empty_sliced,
                            'sine': mflat_sliced
                        }

                        # Creating the variables for the weighted integration weights.
                        # If integrationg weights of the correct length are found on file, they get overwritten with
                        # the proper loaded weights.
                        pulse_weight_I = flat
                        pulse_weight_Q = flat
                        pulse_weight_m_I = flat
                        pulse_weight_m_Q = flat

                        pulse_weight_empty = empty

                        if loaded_weights is not None:
                            # Check if the current pulse has loaded integration weights
                            if any(pulse in weights for weights in loaded_weights):
                                pulse_weight_I_temp = loaded_weights[pulse + '_I']
                                pulse_weight_Q_temp = loaded_weights[pulse + '_Q']

                                I_length = sum(i[1] for i in pulse_weight_I_temp)
                                Q_length = sum(i[1] for i in pulse_weight_Q_temp)

                                # Check that they are the correct length.
                                if I_length == pulse_len and Q_length == pulse_len:
                                    pulse_weight_I = pulse_weight_I_temp
                                    pulse_weight_Q = pulse_weight_Q_temp

                                    pulse_weight_m_I = [(-1*pulseI[0], pulseI[1]) for pulseI in pulse_weight_I]
                                    pulse_weight_m_Q = [(-1*pulseQ[0], pulseQ[1]) for pulseQ in pulse_weight_Q]

                                    pulse_weight_empty = [(0.0, pulse_weight_I[0][1])] * len(pulse_weight_I)
                                    logging.info(f'Loaded weighted integration weights for {pulse}.')
                                else:
                                    logging.info(f'Found old integration weights for {pulse}, deleting them from file.')
                                    loaded_weights.pop(pulse + '_I')
                                    loaded_weights.pop(pulse + '_Q')
                                    deleted_weights = True

                            else:
                                logging.info(f'No integration weights found for {pulse}, using flat weights.')
                        else:
                            if not no_file_warning:
                                no_file_warning = True
                                logging.info('Integration weights file not found, using flat weights.')

                        pulses[pulse][pulse + '_weighted_cos'] = {
                            'cosine': pulse_weight_I,
                            'sine': pulse_weight_m_Q
                        }
                        pulses[pulse][pulse + '_weighted_sin'] = {
                            'cosine': pulse_weight_Q,
                            'sine': pulse_weight_I
                        }
                        pulses[pulse][pulse + '_weighted_minus_sin'] = {
                            'cosine': pulse_weight_m_Q,
                            'sine': pulse_weight_m_I
                        }

                    # Assembling the dictionary that the config dict needs in each pulse for integration weights.
                    possible_integration_weights_per_pulse = {}  # Dictionary with integration weights for this pulse.
                    for weights in pulses[pulse].keys():
                        possible_integration_weights_per_pulse[weights] = weights
                    config_dict["pulses"][key]['integration_weights'] = possible_integration_weights_per_pulse

        # Assembling the 'integration_weights' dictionary.
        integration_weights = {}
        for pul, val in pulses.items():
            for integration_name, int_weight in val.items():
                integration_weights[integration_name] = int_weight

        config_dict['integration_weights'] = integration_weights


        if loaded_weights is not None:
            # Check that there are not old integration weights for pulses that don't exists anymore.
            delete = []
            for weights in loaded_weights.keys():
                if weights[:-2] not in pulses.keys():
                    delete.append(weights)
                    deleted_weights = True

            for old_weight in delete:
                # Delete the weight from the file if these weights are not used.
                loaded_weights.pop(old_weight)

            # Delete weights that should be deleted and save the weights without the old ones present.
            if deleted_weights:
                os.remove(integration_weights_file)
                if len(loaded_weights) != 0:
                    with open(integration_weights_file, 'w') as file:
                        json.dump(loaded_weights, file)


        return config_dict

    # The following are helper methods written by Quantum Machines to create integration weights
    def _round_to_fixed_point_accuracy(self, x, base=2 ** -15):
        """
        Written by Quantum Machines.
        """

        return np.round(base * np.round(np.array(x) / base), 20)

    def convert_full_list_to_list_of_tuples(self, integration_weights, N=100, accuracy=2 ** -15):
        """
        Written by Quantum Machines.

        Converts a list of integration weights, in which each sample corresponds to a clock cycle (4ns), to a list
        of tuples with the format (weight, time_to_integrate_in_ns).
        Can be used to convert between the old format (up to QOP 1.10) to the new format introduced in QOP 1.20.

        :param integration_weights: A list of integration weights.
        :param N:   Maximum number of tuples to return. The algorithm will first create a list of tuples,
                    and then if it is
                    too long, it will run :func:`compress_integration_weights` on them.
        :param accuracy:    The accuracy at which to calculate the integration weights. Default is 2^-15, which is
                            the accuracy at which the OPX operates for the integration weights.
        :type integration_weights: list[float]
        :type N: int
        :type accuracy: float
        :return: List of tuples representing the integration weights
        """
        integration_weights = self._round_to_fixed_point_accuracy(integration_weights, accuracy)
        changes_indices = np.where(np.abs(np.diff(integration_weights)) > 0)[0].tolist()
        prev_index = -1
        new_integration_weights = []
        for curr_index in (changes_indices + [len(integration_weights) - 1]):
            constant_part = (integration_weights[curr_index].tolist(), round(4 * (curr_index - prev_index)))
            new_integration_weights.append(constant_part)
            prev_index = curr_index

        new_integration_weights = self.compress_integration_weights(new_integration_weights, N=N)
        return new_integration_weights

    def compress_integration_weights(self, integration_weights, N=100):
        """
        Written by Quantum Machines.

        Compresses the list of tuples with the format (weight, time_to_integrate_in_ns) to one with length < N.
        Works by iteratively finding the nearest integration weights and combining them with a weighted average.

        :param integration_weights: The integration_weights to be compressed.
        :param N: The maximum list length required.
        :return: The compressed list of tuples representing the integration weights.
        """
        while len(integration_weights) > N:
            diffs = np.abs(np.diff(integration_weights, axis=0)[:, 0])
            min_diff = np.min(diffs)
            min_diff_indices = np.where(diffs == min_diff)[0]
            integration_weights = np.array(integration_weights)
            times1 = integration_weights[min_diff_indices, 1]
            times2 = integration_weights[min_diff_indices + 1, 1]
            weights1 = integration_weights[min_diff_indices, 0]
            weights2 = integration_weights[min_diff_indices + 1, 0]
            integration_weights[min_diff_indices, 0] = (weights1 * times1 + weights2 * times2) / (times1 + times2)
            integration_weights[min_diff_indices, 1] = times1 + times2
            integration_weights = np.delete(integration_weights, min_diff_indices + 1, 0)
            integration_weights = list(zip(integration_weights.T[0].tolist(),
                                           integration_weights.T[1].astype(int).tolist()))

        return integration_weights

    def configure_octave(self, qmm, qm):
        raise NotImplementedError
