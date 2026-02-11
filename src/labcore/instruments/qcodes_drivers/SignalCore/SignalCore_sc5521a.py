"Made by jenshnielse. Edited to add more functions by Randy Owen"

import ctypes #foreign function library. Provide C-compatible data types
import ctypes.wintypes #widnow specific data types
import os #for communicating with the operating system. Managing files+
import sys #for accessing what the python intepreter is seeing
from typing import Dict, Optional # for talking to earlier versions of python and defines standard notations
from qcodes import Instrument #Intrument class of the qcodes package
from qcodes.utils.validators import Enum,Numbers,Ints,Multiples,PermissiveMultiples
#validators check if an arguement is of a certain type. 
#Enum requires that ones of a provided set of values match the arguement


MAXDEVICES = 50 # the number of signal cores it looks for, I think
MAXDESCRIPTORSIZE = 9
COMMINTERFACE = ctypes.c_uint8(1)

#the next blocks of code are defining Classes of ctype.Structure that are like a dictonary
#   for telling C what type of object each keyword is and how to store it
#for example the ctypes.c_int8 is for passing C an unsigned interger byte
#  
class ManDate(ctypes.Structure): #defining how to represent the manufacturing date
    _fields_ = [('year', ctypes.c_uint8),
                ('month', ctypes.c_uint8),
                ('day', ctypes.c_uint8),
                ('hour', ctypes.c_uint8)]


class DeviceInfoT(ctypes.Structure): #defining how to represent the device information
    _fields_ = [('product_serial_number', ctypes.c_uint32), #uses 32-bit because serial number is long
                ('hardware_revision', ctypes.c_float), #uses float because it expects decimal
                ('firmware_revision', ctypes.c_float),
                ('device_interfaces', ctypes.c_uint8),
                ('man_date', ManDate)] #expects the variables defined in the ManData class
device_info_t = DeviceInfoT() #making an object instance of the DeviceInfoT Class so its attributes can be called later


class ListModeT(ctypes.Structure): #defining how to represent the type of sweeping mode
    _fields_ = [('sweep_mode', ctypes.c_uint8),
                ('sweep_dir', ctypes.c_uint8),
                ('tri_waveform', ctypes.c_uint8),
                ('hw_trigger', ctypes.c_uint8),
                ('step_on_hw_trig', ctypes.c_uint8),
                ('return_to_start', ctypes.c_uint8),
                ('trig_out_enable', ctypes.c_uint8),
                ('trig_out_on_cycle', ctypes.c_uint8)]
list_mode_t=ListModeT()

class PLLStatusT(ctypes.Structure): #defining how to represent the phase lock loop status
    _fields_ = [('sum_pll_ld', ctypes.c_uint8),
                ('crs_pll_ld', ctypes.c_uint8),
                ('fine_pll_ld', ctypes.c_uint8),
                ('crs_ref_pll_ld', ctypes.c_uint8),
                ('crs_aux_pll_ld', ctypes.c_uint8),
                ('ref_100_pll_ld', ctypes.c_uint8),
                ('ref_10_pll_ld', ctypes.c_uint8)]
pll_status_t=PLLStatusT()

class OperateStatusT(ctypes.Structure): #defining how to represent Operating statues 
    _fields_ = [('rf1_lock_mode', ctypes.c_uint8),
                ('rf1_loop_gain', ctypes.c_uint8),
                ('device_access', ctypes.c_uint8),
                ('device_standby', ctypes.c_uint8),
                ('auto_pwr_disable', ctypes.c_uint8),
                ('output_enable', ctypes.c_uint8),
                ('ext_ref_lock_enable', ctypes.c_uint8),
                ('ext_ref_detect', ctypes.c_uint8),
                ('ref_out_select', ctypes.c_uint8),
                ('list_mode_running', ctypes.c_uint8),
                ('rf_mode', ctypes.c_uint8),
                ('over_temp', ctypes.c_uint8),
                ('harmonic_ss', ctypes.c_uint8),
                ('pci_clk_enable', ctypes.c_uint8)]
operate_status_t=OperateStatusT()

class DeviceStatusT(ctypes.Structure): #seems to define a higher class that contains all operating mode details
    _fields_ = [('list_mode_t', ListModeT),
                ('operate_status_t', OperateStatusT),
                ('pll_status_t', PLLStatusT)]
device_status_t = DeviceStatusT()


class HWTriggerT(ctypes.Structure):
    _fields_ = [('edge', ctypes.c_uint8),
                ('pxi_enable', ctypes.c_uint8),
                ('pxi_line', ctypes.c_uint8)]
hw_trigger_t = HWTriggerT()


class DeviceRFParamsT(ctypes.Structure): #defining the RF parameters of the sweeps
    _fields_ = [('frequency', ctypes.c_double),
                ('sweep_start_freq', ctypes.c_double),
                ('sweep_stop_freq', ctypes.c_double),
                ('sweep_step_freq', ctypes.c_double),
                ('sweep_dwell_time', ctypes.c_uint32),
                ('sweep_cycles', ctypes.c_uint32),
                ('buffer_points', ctypes.c_uint32),
                ('rf_phase_offset', ctypes.c_float),
                ('power_level', ctypes.c_float),
                ('atten_value', ctypes.c_float),
                ('level_dac_value', ctypes.c_uint16)]
device_rf_params_t = DeviceRFParamsT()

#the dictonary of errors
error_dict = {'0':'SCI_SUCCESS',
              '0':'SCI_ERROR_NONE',
              '-1':'SCI_ERROR_INVALID_DEVICE_HANDLE',
              '-2':'SCI_ERROR_NO_DEVICE',
              '-3':'SCI_ERROR_INVALID_DEVICE',
              '-4':'SCI_ERROR_MEM_UNALLOCATE',
              '-5':'SCI_ERROR_MEM_EXCEEDED',
              '-6':'SCI_ERROR_INVALID_REG',
              '-7':'SCI_ERROR_INVALID_ARGUMENT',
              '-8':'SCI_ERROR_COMM_FAIL',
              '-9':'SCI_ERROR_OUT_OF_RANGE',
              '-10':'SCI_ERROR_PLL_LOCK',
              '-11':'SCI_ERROR_TIMED_OUT',
              '-12':'SCI_ERROR_COMM_INIT',
              '-13':'SCI_ERROR_TIMED_OUT_READ',
              '-14':'SCI_ERROR_INVALID_INTERFACE'}


#defining the qcodes insturment class for the device 
class SC5521A(Instrument):
    __doc__ = 'QCoDeS python driver for the Signal Core SC5521A.'

    def __init__(self, name: str, #name the intsrument
                        serial_number:str,
                       dll_path: str='SignalCore\\SC5520A\\api\\c\\scipci\\x64\\sc5520a_uhfs.dll', 
                       #a path to the DLL of the device, which is the C program that actually drives the device
                       #note, that this works for SC5521A despite the name
                       **kwargs):
        """
        QCoDeS driver for the Signal Core SC5521A.
        This driver has been tested when only one SignalCore is connected to the
        computer.

        Args:
        name (str): Name of the instrument.
        dll_path (str): Path towards the instrument DLL.
        """

        (super().__init__)(name, **kwargs)

        self._devices_number = ctypes.c_uint() #setting the d
        self._pxi10Enable = 0 #I don't know what this does
        self._lock_external = 0 #This might set the default to internal clock refrence 
        self._clock_frequency = 10 #sets the clock frequency to 10 MHz

        self._serial_number = ctypes.c_char_p(bytes(serial_number, 'utf-8'))

        buffers = [ctypes.create_string_buffer(MAXDESCRIPTORSIZE + 1) for bid in range(MAXDEVICES)]
        #this line creates a buffer (mutable memory) object for all the potential devices
        self.buffer_pointer_array = (ctypes.c_char_p * MAXDEVICES)()
        #c_char_p creates an array of C char types with null pointers of the size MAXDEVICES 
        for device in range(MAXDEVICES):
            self.buffer_pointer_array[device] = ctypes.cast(buffers[device], ctypes.c_char_p)
        #turning the elements of the buffer_pointer_array into char_p, which are pointers to strings
        self._buffer_pointer_array_p = ctypes.cast(self.buffer_pointer_array, ctypes.POINTER(ctypes.c_char_p))
        # This defines the pointers of the buffer_pointer_array. I think
        # Adapt the path to the computer language
        if sys.platform == 'win32': #checks the OS 
            dll_path = os.path.join(os.environ['PROGRAMFILES'], dll_path)#adding the c:\ProgramFiles to the DLL path
            self._dll = ctypes.WinDLL(dll_path)
            print(dll_path)
            print(self._dll)
        else:
            raise EnvironmentError(f"{self.__class__.__name__} is supported only on Windows platform")
        
        found = self._dll.sc5520a_uhfsSearchDevices(COMMINTERFACE, self._buffer_pointer_array_p, ctypes.byref(self._devices_number))
        
        #runs the SearchDevices command with the ouput being the pointer to the serial numbers located in _buffer_pointer_array_p
        if found:
            raise RuntimeError('Failed to find any device')
        self._open(serial_number)
        #setting up retrieving the device status, required for changing just one of the elements of the ListModeT
        self._list_mode = ListModeT()
        self._status = OperateStatusT()
        self._pll_status = PLLStatusT()
        self._device_status = DeviceStatusT(self._list_mode, self._status, self._pll_status)

        self.add_parameter(name='temperature',
                           docstring='Return the microwave source internal temperature.',
                           label='Device temperature',
                           unit='celsius',
                           get_cmd=self._get_temperature)

        self.add_parameter(name='output_status',
                           docstring='.',
                           vals=Enum(0, 1),
                           set_cmd=self._set_status,
                           get_cmd=self._get_status)

        self.add_parameter(name='power',
                           docstring='.',
                           label='Power',
                           unit='dbm',
                           set_cmd=self._set_power,
                           get_cmd=self._get_power)

        self.add_parameter(name='frequency',
                           docstring='.',
                           label='Frequency',
                           unit='Hz',
                           set_cmd=self._set_frequency,
                           get_cmd=self._get_frequency)

        self.add_parameter(name='rf1_mode',
                           docstring='0=single tone. 1=sweep',
                           vals=Enum(0,1),
                        #    initial_value=0,
                           set_cmd=self._set_rf_mode,
                           get_cmd=self._get_rf_mode)

        self.add_parameter(name='clock_frequency',
                           docstring='Select the internal clock frequency, 10 or 100MHz.',
                           unit='MHz',
                           vals=Enum(10, 100),
                        #    initial_value=10,
                           set_cmd=self._set_clock_frequency,
                           get_cmd=self._get_clock_frequency)

        self.add_parameter(name='clock_reference',
                           docstring='Select the clock reference, internal or external.',
                           vals=Enum('internal', 'external'),
                        #    initial_value='internal',
                           set_cmd=self._set_clock_reference,
                           get_cmd=self._get_clock_reference)

        ##Things Randy Wrote Start point
        
        self.add_parameter(name='sweep_start_frequency',
                           label='sweep_start_frequency',
                           docstring='Frequency at the start of sweep. Hz',
                           get_cmd=self._get_sweep_start_frequency,
                           set_cmd=self._set_sweep_start_frequency,
                           unit='Hz',
                           vals=Numbers(min_value=160E6,max_value=40E9)
                           )
        self.add_parameter(name='sweep_stop_frequency',
                           label='sweep_stop_frequency',
                           docstring='Frequency at the end of sweep.',
                           get_cmd=self._get_sweep_stop_frequency,
                           set_cmd=self._set_sweep_stop_frequency,
                           unit='Hz',
                           vals=Numbers(min_value=160E6,max_value=40E9)
                           )                                      
        self.add_parameter(name='sweep_step_frequency',
                           label='sweep_step_frequency',
                           docstring='Frequency at the end of sweep.',
                           get_cmd=self._get_sweep_step_frequency,
                           set_cmd=self._set_sweep_step_frequency,
                           unit='Hz',
                           vals=Numbers(min_value=0,max_value=40E9)
                           )
        self.add_parameter(name='sweep_dwell_time',
                           label='sweep_dwell_time',
                           docstring='time in between sweep points. Units of 500us',
                           get_cmd=self._get_sweep_dwell_time,
                           get_parser=int,
                           set_cmd=self._set_sweep_dwell_time,
                           set_parser=int,
                           unit='',
                           vals=Numbers(min_value=1)
                           )
        self.add_parameter(name='sweep_cycles',
                           label='sweep_cycles',
                           docstring='how many times sweep is repeated. 0 is infinite',
                           get_cmd=self._get_sweep_cycles,
                           get_parser=int,
                           set_cmd=self._set_sweep_cycles,
                           set_parser=int,
                           unit='',
                           vals=Ints(min_value=0),
                        #    initial_value=1,
                           )
        self.add_parameter(name='rf_phase_ouput',
                           label='rf_phase_output',
                           docstring='Ajust the phase of signal on the output. Must be multiples of 0.1 degree',
                           get_cmd=self._get_rf_phase_output,
                           set_cmd=self._set_rf_phase_output,
                           unit='degrees',
                           vals=PermissiveMultiples(0.1),
                           )
        self.add_parameter(name='sss_mode',
                           label='sss_mode',
                           docstring='0 = List mode. Device gets its frequency points from the list buffer uploaded via LIST_BUFFER_WRITE register. 1 = Sweep mode. The device computes the frequency points using the Start, Stop and Step frequencies',
                           get_cmd=self._get_sweep_mode,
                           set_cmd=self._set_sweep_mode,
                           unit='',
                           vals=Enum(0,1),
                        #    initial_value=1,
                           )
        self.add_parameter(name='sweep_dir',
                           label='sweep_dir',
                           docstring='0 = forwards sweep. 1 = Backwards sweep',
                           get_cmd=self._get_sweep_dir,
                           set_cmd=self._set_sweep_dir,
                           unit='',
                           vals=Enum(0,1),
                        #    initial_value=0,
                           )
        self.add_parameter(name='tri_waveform',
                           label='tri_waveform',
                           docstring='0 = Sawtooth waveform. 1 = Triangular waveform',
                           get_cmd=self._get_tri_waveform,
                           set_cmd=self._set_tri_waveform,
                           unit='',
                           vals=Enum(0,1),
                        #    initial_value=0,
                           )
        self.add_parameter(name='hw_trigger',
                           label='hw_trigger',
                           docstring='0 = software trigger. 1 = hardware trigger',
                           get_cmd=self._get_hw_trigger,
                           set_cmd=self._set_hw_trigger,
                           unit='',
                           vals=Enum(0,1),
                        #    initial_value=0,
                           )                                                  
        self.add_parameter(name='step_on_hw_trig',
                           label='step_on_hw_trig',
                           docstring='0 = start/stop. 1 =step to next freq. with hardware trigger',
                           get_cmd=self._get_step_on_hw_trig,
                           set_cmd=self._set_step_on_hw_trig,
                           unit='',
                           vals=Enum(0,1),
                        #    initial_value=0,
                           )
        self.add_parameter(name='return_to_start',
                           label='return_to_start',
                           docstring='0=stops at end of list. 1=return to start of list at end',
                           get_cmd=self._get_return_to_start,
                           set_cmd=self._set_return_to_start,
                           unit='',
                           vals=Enum(0,1),
                        #    initial_value=0,
                           )
        self.add_parameter(name='trig_out_enable',
                           label='trig_out_enable',
                           docstring='0=no trigger output. 1=trigger on TRIGOUT pin',
                           get_cmd=self._get_trig_out_enable,
                           set_cmd=self._set_trig_out_enable,
                           unit='',
                           vals=Enum(0,1),
                        #    initial_value=1,
                           )
        self.add_parameter(name='trig_out_on_cycle',
                           label='trig_out_on_cycle',
                           docstring='0=trigger on frequency change. 1=trigger on cycle end',
                           get_cmd=self._get_trig_out_on_cycle,
                           set_cmd=self._set_trig_out_on_cycle,
                           unit='',
                           vals=Enum(0,1),
                        #    initial_value=1,
                           )                    

        self.connect_message() #Sends out a message that things have been connected

    def _open(self, serial_number) -> None:
        if sys.platform == "win32":
            self._handle = ctypes.wintypes.HANDLE()
        else:
            raise EnvironmentError(f"{self.__class__.__name__} is supported only on Windows platform")

        msg=self._dll.sc5520a_uhfsOpenDevice(COMMINTERFACE, self.buffer_pointer_array[0], ctypes.c_uint8(1), ctypes.byref(self._handle))
        # msg=self._dll.sc5520a_uhfsOpenDevice(COMMINTERFACE, #which communication interface we are using
        # ctypes.c_char_p(bytes(serial_number, 'utf-8')), #serial number?
        # ctypes.c_uint8(1), 
        # ctypes.byref(self._handle))
        self._error_handler(msg)
        print(self._handle)

    def _close(self) -> None:
        msg=self._dll.sc5520a_uhfsCloseDevice(self._handle) #closes the device
        self._error_handler(msg)
    def _error_handler(self, msg: int) -> None:
        """Display error when setting the device fail.

        Args:
            msg (int): error key, see error_dict dict.

        Raises:
            BaseException
        """

        if msg!=0:
            raise BaseException("Couldn't set the devise due to {}.".format(error_dict[str(msg)]))
        else:
            pass
    
    def soft_trigger(self) -> None:
        """
        Send out a soft trigger, so that the we can start the sweep
        Generator need to be configured for list mode and soft trigger is selected as the trigger source
        """
        # logging.info(__name__ + ' : Send a soft trigger to the generator')
        self._dll.sc5520a_uhfsListSoftTrigger(self._handle)
        return None


    def _get_temperature(self) -> float:
        temperature = ctypes.c_float()
        self._dll.sc5520a_uhfsFetchTemperature(self._handle, ctypes.byref(temperature))
        return temperature.value

    def _set_status(self, status_: int) -> None:
        msg = self._dll.sc5520a_uhfsSetOutputEnable(self._handle, ctypes.c_int(status_))
        self._error_handler(msg)

    def _get_status(self) -> str:
        self._dll.sc5520a_uhfsFetchDeviceStatus(self._handle, ctypes.byref(device_status_t))
        return device_status_t.operate_status_t.output_enable

    def _set_power(self, power: float) -> None:
        msg = self._dll.sc5520a_uhfsSetPowerLevel(self._handle, ctypes.c_float(power))
        self._error_handler(msg)

    def _get_power(self) -> float:
        self._dll.sc5520a_uhfsFetchRfParameters(self._handle, ctypes.byref(device_rf_params_t))
        return device_rf_params_t.power_level

    def _set_frequency(self, frequency: float) -> None:
        msg = self._dll.sc5520a_uhfsSetFrequency(self._handle, ctypes.c_double(frequency))
        self._error_handler(msg)

    def _get_frequency(self) -> float:
        device_rf_params_t = DeviceRFParamsT()
        self._dll.sc5520a_uhfsFetchRfParameters(self._handle, ctypes.byref(device_rf_params_t))
        return float(device_rf_params_t.frequency)

    def _set_clock_frequency(self, clock_frequency: float) -> None:
        if clock_frequency == 10:
            self._select_high = 0
        else:
            self._select_high = 1
        msg = self._dll.sc5520a_uhfsSetReferenceMode(self._handle, ctypes.c_int(self._pxi10Enable), ctypes.c_int(self._select_high), ctypes.c_int(self._lock_external))
        self._error_handler(msg)

    def _get_clock_frequency(self) -> float:
        self._dll.sc5520a_uhfsFetchDeviceStatus(self._handle, ctypes.byref(device_status_t))
        ref_out_select = device_status_t.operate_status_t.ref_out_select
        if ref_out_select == 1:
            return 100
        return 10

    def _set_clock_reference(self, clock_reference: str) -> None:
        if clock_reference.lower() == 'internal':
            self._lock_external = 0
        else:
            self._lock_external = 1
        msg = self._dll.sc5520a_uhfsSetReferenceMode(self._handle, ctypes.c_int(self._pxi10Enable), ctypes.c_int(self._select_high), ctypes.c_int(self._lock_external))
        self._error_handler(msg)

    def _get_clock_reference(self) -> str:
        self._dll.sc5520a_uhfsFetchDeviceStatus(self._handle, ctypes.byref(device_status_t))
        ext_ref_detect = device_status_t.operate_status_t.ext_ref_detect
        if ext_ref_detect == 1:
            return 'external'
        return 'internal'

    def _set_rf_mode(self, rf_mode: int) -> None:
        c_rf_mode = ctypes.c_ubyte(int(rf_mode))
        msg = self._dll.sc5520a_uhfsSetRfMode(self._handle, c_rf_mode)
        self._error_handler(msg)

    def _get_rf_mode(self) -> str:
        self._dll.sc5520a_uhfsFetchDeviceStatus(self._handle, ctypes.byref(device_status_t))
        rf_mode = device_status_t.operate_status_t.rf_mode
        return int(rf_mode)

    #Methods I, peasant Randy, have defined, so probably don't work
    def _set_sweep_start_frequency(self, sweep_start_freq: float) -> None:
        """
        Set the start frequency of a sweep. Units of Hz.
        """
        c_sweep_start_freq = ctypes.c_double(int(sweep_start_freq))
        msg = self._dll.sc5520a_uhfsSweepStartFreq(self._handle, c_sweep_start_freq)
        self._error_handler(msg)
    def _get_sweep_start_frequency(self) -> str:
        """
        Set the start frequency of a sweep. Units of Hz.
        """
        device_rf_params_t = DeviceRFParamsT()
        self._dll.sc5520a_uhfsFetchRfParameters(self._handle, ctypes.byref(device_rf_params_t))
        return float(device_rf_params_t.sweep_start_freq)
    
    def _set_sweep_stop_frequency(self, sweep_stop_freq: float) -> None:
        """
        Set the stop frequency of a sweep. Units of Hz.
        """
        c_sweep_stop_freq = ctypes.c_double(int(sweep_stop_freq))
        msg = self._dll.sc5520a_uhfsSweepStopFreq(self._handle, c_sweep_stop_freq)
        self._error_handler(msg)
    def _get_sweep_stop_frequency(self) -> str:
        """
        Set the start frequency of a sweep. Units of Hz.
        """
        device_rf_params_t = DeviceRFParamsT()
        self._dll.sc5520a_uhfsFetchRfParameters(self._handle, ctypes.byref(device_rf_params_t))
        return float(device_rf_params_t.sweep_stop_freq)
    
    def _set_sweep_step_frequency(self, sweep_step_freq: float) -> None:
        """
        Set the frequency steps of a sweep. Units of Hz.
        """
        c_sweep_step_freq = ctypes.c_double(int(sweep_step_freq))
        msg = self._dll.sc5520a_uhfsSweepStepFreq(self._handle, c_sweep_step_freq)
        self._error_handler(msg)
    def _get_sweep_step_frequency(self) -> str:
        """
        Set the frequency steps of a sweep. Units of Hz.
        """
        device_rf_params_t = DeviceRFParamsT()
        self._dll.sc5520a_uhfsFetchRfParameters(self._handle, ctypes.byref(device_rf_params_t))
        return float(device_rf_params_t.sweep_step_freq)

    def _set_sweep_dwell_time(self, dwell_unit: int) -> None:
        """
        Set the sweep/list time at each frequency point.
        Note that the dwell time is set as multiple of 500 us.
        The input value is an unsigned int, it means how many multiple of 500 us.
        """
        c_dwell_unit = ctypes.c_uint(int(dwell_unit))
        msg = self._dll.sc5520a_uhfsSweepDwellTime(self._handle, c_dwell_unit)
        self._error_handler(msg)
    def _get_sweep_dwell_time(self) -> str:
        """
        Get the sweep/list time at each frequency point.
        Note that the dwell time is set as multiple of 500 us.
        """
        device_rf_params_t = DeviceRFParamsT()
        self._dll.sc5520a_uhfsFetchRfParameters(self._handle, ctypes.byref(device_rf_params_t))
        return float(device_rf_params_t.sweep_dwell_time)

    def _set_sweep_cycles(self, sweep_cycles: int) -> None:
        """
        Set the number of times the sweep will cycle before stopping.
        0 corresponds to a an infinite loop 
        """
        c_sweep_cycles = ctypes.c_uint(int(sweep_cycles)) #making this a python int should be redudant
        msg = self._dll.sc5520a_uhfsListCycleCount(self._handle, c_sweep_cycles)
        self._error_handler(msg)
    def _get_sweep_cycles(self) -> str:
        """
        Get the number of times the sweep will cycle before stopping.
        0 corresponds to a an infinite loop 
        """
        device_rf_params_t = DeviceRFParamsT()
        self._dll.sc5520a_uhfsFetchRfParameters(self._handle, ctypes.byref(device_rf_params_t))
        return int(device_rf_params_t.sweep_cycles)

    def _set_rf_phase_output(self, rf_phase_output:float) -> None:
        """
        Sets the phase of the output RF signal. 0.1 degree steps
        """
        c_rf_phase_output = ctypes.cfloat(rf_phase_output)
        msg = self._dll.sc5520a_uhfsSetSignalPhase(self._handle, c_rf_phase_output)
        self._error_handler(msg)
    def _get_rf_phase_output(self) -> str:
        """
        Sets the phase of the output RF signal.
        """
        device_rf_params_t = DeviceRFParamsT()
        self._dll.sc5520a_uhfsFetchRfParameters(self._handle, ctypes.byref(device_rf_params_t))
        return float(device_rf_params_t.rf_phase_offset)
    
    def _set_sweep_mode(self, sweep_mode:int) -> None:
        """
        Set the list/sweep mode of the generator
        0 = List mode. Device gets its frequency points from the list buffer uploaded via LIST_BUFFER_WRITE register
        1 = Sweep mode. The device computes the frequency points using the Start, Stop and Step frequencies
        """
        c_sweep_mode = ctypes.c_ubyte(int(sweep_mode)) #convert the Python Int into C byte 
        self._dll.sc5520a_uhfsFetchDeviceStatus(self._handle, ctypes.byref(self._device_status))
        #fetch the device status, which contains the list_mode_t values
        self._device_status.list_mode_t.sweep_mode = c_sweep_mode 
        # change the dictonary value of "sweep mode" in the internally stored list mode dictionary
        msg = self._dll.sc5520a_uhfsListModeConfig(self._handle, ctypes.byref(self._device_status.list_mode_t))
        
        
        self._error_handler(msg)
    def _get_sweep_mode(self) -> str:
        """
        Get the list/sweep mode of the generator
        0 = List mode. Device gets its frequency points from the list buffer uploaded via LIST_BUFFER_WRITE register
        1 = Sweep mode. The device computes the frequency points using the Start, Stop and Step frequencies
        """
        self._dll.sc5520a_uhfsFetchDeviceStatus(self._handle, ctypes.byref(self._device_status))
        #fetch the device status, which contains the list_mode_t values
        sweep_mode = self._device_status.list_mode_t.sweep_mode 
        return sweep_mode
    def _set_sweep_dir(self, sweep_dir:int) -> None:
        """
        Defines the sweep direction.
        0: Forewards sweep.
        1: Backwards sweep.
        """
        c_sweep_dir = ctypes.c_ubyte(int(sweep_dir)) #convert the Python Int into C byte 
        self._dll.sc5520a_uhfsFetchDeviceStatus(self._handle, ctypes.byref(self._device_status))
        #fetch the device status, which contains the list_mode_t values
        self._device_status.list_mode_t.sweep_dir = c_sweep_dir 
        # change the dictonary value of "sweep mode" in the internally stored list mode dictionary
        msg = self._dll.sc5520a_uhfsListModeConfig(self._handle, ctypes.byref(self._device_status.list_mode_t))
        self._error_handler(msg)
    def _get_sweep_dir(self) -> str:
        """
        Defines the sweep direction.
        0: Forewards sweep.
        1: Backwards sweep.
        """
        self._dll.sc5520a_uhfsFetchDeviceStatus(self._handle, ctypes.byref(self._device_status))
        #fetch the device status, which contains the list_mode_t values
        sweep_dir = self._device_status.list_mode_t.sweep_dir 
        return sweep_dir

    def _set_tri_waveform(self, tri_waveform:int) -> None:
        """
        Set the triangular waveform of the generator
        0 = Sawtooth waveform. Frequency returns to the beginning frequency upon reaching the end of a sweep cycle
        1 = Triangular waveform. Frequency reverses direction at the end of the list and steps back towards the
            beginning to complete a cycle
        """
        c_tri_waveform = ctypes.c_ubyte(int(tri_waveform)) #convert the Python Int into C byte 
        self._dll.sc5520a_uhfsFetchDeviceStatus(self._handle, ctypes.byref(self._device_status))
        #fetch the device status, which contains the list_mode_t values
        self._device_status.list_mode_t.tri_waveform = c_tri_waveform 
        # change the dictonary value of "sweep mode" in the internally stored list mode dictionary
        msg = self._dll.sc5520a_uhfsListModeConfig(self._handle, ctypes.byref(self._device_status.list_mode_t))
        self._error_handler(msg)
    def _get_tri_waveform(self) -> str:
        """
        Get the triangular waveform of the generator
        0 = Sawtooth waveform. Frequency returns to the beginning frequency upon reaching the end of a sweep cycle
        1 = Triangular waveform. Frequency reverses direction at the end of the list and steps back towards the
            beginning to complete a cycle
        """
        self._dll.sc5520a_uhfsFetchDeviceStatus(self._handle, ctypes.byref(self._device_status))
        #fetch the device status, which contains the list_mode_t values
        tri_waveform = self._device_status.list_mode_t.tri_waveform 
        return tri_waveform

    def _set_hw_trigger(self, hw_trigger:int) -> None:
        """
        Set the status of hardware trigger
        0 = Software trigger. Softtrigger can only be used to start and stop a sweep/list cycle. It does not work for
            step-on-trigger mode.
        1 = Hardware trigger. A high-to-low transition on the TRIGIN pin will trigger the device. It can be used for
            both start/stop or step-on-trigger functions.
        """
        c_hw_trigger = ctypes.c_ubyte(int(hw_trigger)) #convert the Python Int into C byte 
        self._dll.sc5520a_uhfsFetchDeviceStatus(self._handle, ctypes.byref(self._device_status))
        #fetch the device status, which contains the list_mode_t values
        self._device_status.list_mode_t.hw_trigger = c_hw_trigger 
        # change the dictonary value of "sweep mode" in the internally stored list mode dictionary
        msg = self._dll.sc5520a_uhfsListModeConfig(self._handle, ctypes.byref(self._device_status.list_mode_t))
        self._error_handler(msg)
    def _get_hw_trigger(self) -> str:
        """
        Get the status of hardware trigger
        0 = Software trigger. Softtrigger can only be used to start and stop a sweep/list cycle. It does not work for
            step-on-trigger mode.
        1 = Hardware trigger. A high-to-low transition on the TRIGIN pin will trigger the device. It can be used for
            both start/stop or step-on-trigger functions.
        """
        self._dll.sc5520a_uhfsFetchDeviceStatus(self._handle, ctypes.byref(self._device_status))
        #fetch the device status, which contains the list_mode_t values
        hw_trigger = self._device_status.list_mode_t.hw_trigger 
        return hw_trigger
    
    def _set_step_on_hw_trig(self, step_on_hw_trig:int) -> None:
        """
        Set the behavior of the sweep/list mode when receiving a trigger.
        0 = Start/Stop behavior. The sweep starts and continues to step through the list for the number of cycles set,
            dwelling at each step frequency for a period set by the dwell time. The sweep/list will end on a consecutive
            trigger.
        1 = Step-on-trigger. This is only available if hardware triggering is selected. The device will step to the next
            frequency on a trigger.Upon completion of the number of cycles, the device will exit from the stepping state
            and stop.
        """
        c_step_on_hw_trig = ctypes.c_ubyte(int(step_on_hw_trig)) #convert the Python Int into C byte 
        self._dll.sc5520a_uhfsFetchDeviceStatus(self._handle, ctypes.byref(self._device_status))
        #fetch the device status, which contains the list_mode_t values
        self._device_status.list_mode_t.step_on_hw_trig = c_step_on_hw_trig 
        # change the dictonary value of "sweep mode" in the internally stored list mode dictionary
        msg = self._dll.sc5520a_uhfsListModeConfig(self._handle, ctypes.byref(self._device_status.list_mode_t))
        self._error_handler(msg)
    def _get_step_on_hw_trig(self) -> str:
        """
        Get the behavior of the sweep/list mode when receiving a trigger.
        0 = Start/Stop behavior
        1 = Step-on-trigger
        """
        self._dll.sc5520a_uhfsFetchDeviceStatus(self._handle, ctypes.byref(self._device_status))
        #fetch the device status, which contains the list_mode_t values
        step_on_hw_trig = self._device_status.list_mode_t.step_on_hw_trig 
        return step_on_hw_trig
    
    def _set_return_to_start(self, return_to_start:int) -> None:
        """
        Set how the frequency will change at the end of the list/sweep
        0 = Stop at end of sweep/list. The frequency will stop at the last point of the sweep/list
        1 = Return to start. The frequency will return and stop at the beginning point of the sweep or list after a
            cycle.
        """
        c_return_to_start = ctypes.c_ubyte(int(return_to_start)) #convert the Python Int into C byte 
        self._dll.sc5520a_uhfsFetchDeviceStatus(self._handle, ctypes.byref(self._device_status))
        #fetch the device status, which contains the list_mode_t values
        self._device_status.list_mode_t.return_to_start = c_return_to_start 
        # change the dictonary value of "sweep mode" in the internally stored list mode dictionary
        msg = self._dll.sc5520a_uhfsListModeConfig(self._handle, ctypes.byref(self._device_status.list_mode_t))
        self._error_handler(msg)
    def _get_return_to_start(self) -> str:
        """
        Get how the frequency will change at the end of the list/sweep
        0 = Stop at end of sweep/list. The frequency will stop at the last point of the sweep/list
        1 = Return to start. The frequency will return and stop at the beginning point of the sweep or list after a
            cycle.
        """
        self._dll.sc5520a_uhfsFetchDeviceStatus(self._handle, ctypes.byref(self._device_status))
        #fetch the device status, which contains the list_mode_t values
        return_to_start = self._device_status.list_mode_t.return_to_start 
        return return_to_start
    
    def _set_trig_out_enable(self, trig_out_enable:int) -> None:
        """
        Set the trigger output status.
        It does not send out the trigger, just enable the generator to send out the trigger
        0 = No trigger output
        1 = Puts a trigger pulse on the TRIGOUT pin
        """
        c_trig_out_enable = ctypes.c_ubyte(int(trig_out_enable)) #convert the Python Int into C byte 
        self._dll.sc5520a_uhfsFetchDeviceStatus(self._handle, ctypes.byref(self._device_status))
        #fetch the device status, which contains the list_mode_t values
        self._device_status.list_mode_t.trig_out_enable = c_trig_out_enable 
        # change the dictonary value of "sweep mode" in the internally stored list mode dictionary
        msg = self._dll.sc5520a_uhfsListModeConfig(self._handle, ctypes.byref(self._device_status.list_mode_t))
        self._error_handler(msg)
    def _get_trig_out_enable(self) -> str:
        """
        Get the trigger output status.
        It does not send out the trigger, just enable the generator to send out the trigger
        0 = No trigger output
        1 = Puts a trigger pulse on the TRIGOUT pin
        """
        self._dll.sc5520a_uhfsFetchDeviceStatus(self._handle, ctypes.byref(self._device_status))
        #fetch the device status, which contains the list_mode_t values
        trig_out_enable = self._device_status.list_mode_t.trig_out_enable 
        return trig_out_enable
    
    def _set_trig_out_on_cycle(self, trig_out_on_cycle:int) -> None:
        """
        Set the trigger output mode
        0 = Puts out a trigger pulse at each frequency change
        1 = Puts out a trigger pulse at the completion of each sweep/list cycle
        """
        c_trig_out_on_cycle = ctypes.c_ubyte(int(trig_out_on_cycle)) #convert the Python Int into C byte 
        self._dll.sc5520a_uhfsFetchDeviceStatus(self._handle, ctypes.byref(self._device_status))
        #fetch the device status, which contains the list_mode_t values
        self._device_status.list_mode_t.trig_out_on_cycle = c_trig_out_on_cycle 
        # change the dictonary value of "sweep mode" in the internally stored list mode dictionary
        msg = self._dll.sc5520a_uhfsListModeConfig(self._handle, ctypes.byref(self._device_status.list_mode_t))
        self._error_handler(msg)
    def _get_trig_out_on_cycle(self) -> str:
        """
        Get the trigger output mode
        0 = Puts out a trigger pulse at each frequency change
        1 = Puts out a trigger pulse at the completion of each sweep/list cycle
        """
        self._dll.sc5520a_uhfsFetchDeviceStatus(self._handle, ctypes.byref(self._device_status))
        #fetch the device status, which contains the list_mode_t values
        trig_out_on_cycle = self._device_status.list_mode_t.trig_out_on_cycle 
        return trig_out_on_cycle

    def get_idn(self) -> Dict[str, Optional[str]]:
        self._dll.sc5520a_uhfsFetchDeviceInfo(self._handle, ctypes.byref(device_info_t))

        return {'vendor':'SignalCore',
                'model':'SC5521A',
                'serial':device_info_t.product_serial_number,
                'firmware':device_info_t.firmware_revision,
                'hardware':device_info_t.hardware_revision,
                'manufacture_date':'20{}-{}-{} at {}h'.format(device_info_t.man_date.year, device_info_t.man_date.month, device_info_t.man_date.day, device_info_t.man_date.hour)}