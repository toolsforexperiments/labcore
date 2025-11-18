# -*- coding: utf-8 -*-
"""
@author: Chao Zhou, Gaurav Agarwal

A simple driver for controlling multiple MiniCircuits RC-8SPDT-A18 switch matrices as one big matrix using QCoDes,
transferred from the one written by Xi Cao and Pinlei Lu.


"""

import logging
# from functools import partial
from typing import Union, List
from qcodes import Instrument
from urllib.request import urlopen

class MiniCircuits_SwitchMatrix_Multi(Instrument):
    def __init__(self, name:str, name_list:List[str], address_list:List[str], mode_dict:dict={}, reset=False, **kwargs):
        """
        :param name: name of all the switches as one instrument
        :param name_list: list of individual names of each switch matrix
        :param address_list: list of address
        :param mode_dict: dictionary that contains the preset modes
        :param reset:
        :param kwargs:
        """
        super().__init__(name, **kwargs)
        logging.info(__name__ + ' : Initializing MiniCircuits RC-8SPDT-A18')
        self._address_list = address_list
        self._mode_dict = mode_dict
        self.switch_dict: {str: MiniCircuits_SwitchMatrix} = {}
        for i, swt_name in enumerate(name_list):
            swt_ = MiniCircuits_SwitchMatrix(swt_name, address_list[i])
            setattr(self, swt_name, swt_)
            self.switch_dict[swt_name] = swt_

        self.add_parameter('portvalue_dict',
                           label='portvalue_dict',
                           get_cmd=self.do_get_portvalue_dict,
                           set_cmd=self.do_set_portvalue_dict)

        self.add_parameter('mode',
                           label='mode',
                           get_cmd=self.do_get_mode,
                           set_cmd=self.do_set_mode)

        self.add_parameter('available_modes',
                           label='available_modes',
                           get_cmd=self.get_mode_options,
                           set_cmd=self.update_mode_dict)

        # self.add_function('modify_add_mode',
        #                    label='modify_add_mode',
        #                    set_cmd=self.add_mode)

        # self.add_function('remove_mode',
        #                    label='remove_mode',
        #                    set_cmd=self.remove_mode)

        if reset:
            self.reset()

    def do_get_portvalue_dict(self):
        port_value_dict = {}
        for swt_name, swt_ in self.switch_dict.items():
            port_value_dict[swt_name] = swt_.portvalue()
        return port_value_dict

    def do_set_portvalue_dict(self, port_value_dict:dict):
        for swt_name, port_value in port_value_dict.items():
            self.switch_dict[swt_name].set_switch("P", port_value)


    def do_get_mode(self):
        current_states = self.portvalue_dict()
        matched_modes = []
        for mode_name, mode in self._mode_dict.items():
            mode_match = True
            for i, (swt_name, portvalue_) in enumerate(current_states.items()):
                for j, s in enumerate(mode[i]):
                    if (s in ["0", "1"]) and (s != portvalue_[j]):
                        mode_match = False
                        break
            if mode_match:
                matched_modes.append(mode_name)
        return matched_modes


    def do_set_mode(self, mode_name):
        current_states = self.portvalue_dict()
        if mode_name in self._mode_dict:
            for i, (swt_name, swt_) in enumerate(self.switch_dict.items()):
                swt_.set_switch(self._create_new_mode_string(current_states[swt_name], self._mode_dict[mode_name][i]))
        else:
         print( 'Confucius say there is no such mode. Nothing has been changed.')


    def _create_new_mode_string(self, current_state, new_state):
        if len(current_state)  != len(new_state):
            raise ValueError("current_state and new_state must be the same length.")
        output = ""
        for i in range(len(new_state)):
            if (new_state[i] not in ["0", "1"]):
                output += current_state[i]
            else:
                output += new_state[i]
        return output

    def get_mode_options(self):
        return self._mode_dict

    def update_mode_dict(self, mode_key_dict: dict):
        self._mode_dict.update(mode_key_dict)
        return


    def set_single_switch(self, switchName:str, chanel: str, state: Union[int, str] ):
        self.switch_dict[switchName].set_switch(chanel, state)

    def reset(self):
        for swt_ in self.switch_dict.values():
            swt_.reset()


class MiniCircuits_SwitchMatrix(Instrument):
    def __init__(self, name, address, reset=False, **kwargs):
        '''
        Initializes the Mini_Circuits switch, and communicates with the wrapper.

        Input:
          name (string)    : name of the instrument
          address (string) : http address
          reset (bool)     : resets to default values, default=False
        '''
        super().__init__(name, **kwargs)
        logging.info(__name__ + ' : Initializing MiniCircuits RC-8SPDT-A18')
        self._address = address
        self.add_parameter('portvalue',
                           label='portvalue',
                           get_cmd=self.do_get_portvalue,
                           set_cmd=self.set_switch)


        if reset:
            self.reset()

    def set_switch(self, state: Union[int, str], chanel:str = 'P'):
        '''
        :param chanel: switch 'A' through 'H' or 'P' if you want to control all the gates at same time
        :param state: 0 or 1 to choose output. 0=1 (green), 1=2 (red)
        '''
        state = str(state)
        logging.info(__name__ + ' : Set switch%s' % chanel +' to state %s' % state)
        if chanel != 'P':
            ret = urlopen(self._address + "/SET" + chanel + "=" + state)
        else:
            if (len(state)) != 8:
                print(len(state))
                raise Exception("Wrong input length!")
            newstate = 0
            for x in range(0,len(state)):
                if (int(state[x]) != 0) & (int(state[x]) != 1):
                    raise Exception("Wrong input value at %ith" % x + " switch!")
                else:
                    newstate += int(state[x])*(2**x)

            ret = urlopen(self._address + "/SETP" + "=" + str(newstate))

        self.get('portvalue')

    def do_get_portvalue(self):
        logging.debug(__name__+' : get portvalue')
        ret = urlopen(self._address + "/SWPORT?" )
        result = ret.readlines()[0]
        result = int(result)
        result = format(result,'08b')
        result = result[::-1]
        return result

    def reset(self):
        self.set_switch("P", "0" * 8)


if __name__ == "__main__":
    modes = {'2_IN': ['xxxxxxxx', 'xxxx00xx', 'xxxxxxxx'],
             '3_IN': ['xxxxxxxx', 'xxxx01xx', 'xxxxxxxx'],
             '12_IN': ['xxxxxxxx', 'xxxx1xx0', 'xxxxxxxx'],
             '18_IN': ['xxxxxxxx', 'xxxx1xx1', 'xxxxxxxx'],
             'A_Out': ['xxxxxxxx', '00xxxxxx', 'xxxxxxxx'],
             'B_Out': ['xxxxxxxx', '01xxxxxx', 'xxxxxxxx'],
             'H_Out': ['xxxxxxxx', '1x1xxxxx', 'xxxxxxxx'],
             'E_Out': ['xxxxxxxx', '1x0xxxxx', 'xxxxxxxx'],
             'VNAInOut': ['xxxxxxxx', 'xxx0xx0x', 'xxxxxxxx'],
             'PXIInOut': ['xxx1xxxx', 'xxx1xx1x', 'xxxxxxxx'],
             'Cav1In': ['01xxxxxx', 'xxxxxxxx', 'xxxxxxxx'],
             'Cav4In': ['11xxxxxx', 'xxxxxxxx', 'xxxxxxxx'],
             'Cav6In': ['x00xxxxx', 'xxxxxxxx', 'xxxxxxxx'],
             'SAQuCaIn': ['x010xxxx', 'xxxxxxxx', 'xxxxxxxx']
             }
    SWT = MiniCircuits_SwitchMatrix_Multi('SWT',name_list=["SWT1", "SWT2", "SWT3"],
                address_list=['http://169.254.254.251', 'http://169.254.254.249', 'http://169.254.254.252'],
                mode_dict= modes)
