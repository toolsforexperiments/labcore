"""
@author: Gaurav Agarwal

QCoDes driver for QICK SoC with RF Board support.
Automatically detects available DAC/ADC channels and creates parameters for:
- Attenuators (RF signal chains)
- Filters (ADMV8818)
- DC biases
- LO synthesizers
- Gain controls

Note: This driver is designed to work with both local and remote (Pyro4) QICK SoC instances.
Since Pyro4 does not allow direct attribute access to object properties, this driver:
- Uses configuration dictionary (_soc_cfg) for channel detection instead of object attributes
- Caches attenuator values locally since they cannot be read back through Pyro4
- Uses method calls (_soc.rfb_set_*) for all hardware control operations
- Does not attempt to read back filter or gain states (returns defaults/cached values)

"""

import logging
from typing import Optional, Dict
from qcodes import Instrument, InstrumentChannel, ChannelList
from qcodes.utils.validators import Numbers, Enum, Bool

# logger = logging.getLogger(__name__)


class DACChannel(InstrumentChannel):
    """
    Channel class for DAC (generator) channels.
    Handles RF attenuators, filters, and DC bias configuration.
    """

    def __init__(self, parent: Instrument, name: str, channel: int, gen_config: dict, **kwargs):
        super().__init__(parent, name, **kwargs)
        self._channel = channel
        self._gen_config = gen_config

        # Cache for attenuator values (cannot read back through Pyro4)
        self._att1_cache = 10.0  # Default attenuation
        self._att2_cache = 10.0

        #TODO Confirm capabilities from config
        self._has_rf_chain = True
        self._has_dc_chain = False
        self._has_filter = True
        self._has_attenuator = True

        # Add basic channel info parameters
        # self.add_parameter(
        #     'channel_index',
        #     get_cmd=lambda: self._channel,
        #     snapshot_value=True,
        #     label='Channel Index',
        #     docstring='Index in the gens list'
        # )

        # self.add_parameter(
        #     'dac_tile',
        #     get_cmd=lambda: int(gen_config['dac'][0]),
        #     snapshot_value=True,
        #     label='DAC Tile',
        #     docstring='DAC tile number'
        # )

        # self.add_parameter(
        #     'dac_block',
        #     get_cmd=lambda: int(gen_config['dac'][1]),
        #     snapshot_value=True,
        #     label='DAC Block',
        #     docstring='DAC block number'
        # )

        # self.add_parameter(
        #     'sampling_freq',
        #     get_cmd=lambda: self.root_instrument._soc_cfg['rf']['dacs'][gen_config['dac']]['fs'],
        #     unit='MHz',
        #     snapshot_value=True,
        #     label='Sampling Frequency',
        #     docstring='DAC sampling frequency in MHz'
        # )

        # RF Chain parameters (attenuators)
        if self._has_rf_chain and self._has_attenuator:
            self.add_parameter(
                'rf_enabled',
                get_cmd=self._get_rf_enabled,
                set_cmd=self._set_rf_enabled,
                vals=Bool(),
                label='RF Output Enabled',
                docstring='Enable/disable RF output'
            )

            self.add_parameter(
                'att1',
                get_cmd=lambda: self._att1_cache,
                set_cmd=lambda val: self._set_attenuator(val, None),
                vals=Numbers(0, 31.75),
                unit='dB',
                label='Attenuator 1',
                docstring='First stage attenuation (0-31.75 dB)'
            )

            self.add_parameter(
                'att2',
                get_cmd=lambda: self._att2_cache,
                set_cmd=lambda val: self._set_attenuator(None, val),
                vals=Numbers(0, 31.75),
                unit='dB',
                label='Attenuator 2',
                docstring='Second stage attenuation (0-31.75 dB)'
            )

        # Filter parameters
        if self._has_filter:
            self.add_parameter(
                'filter_enabled',
                get_cmd=lambda: self._get_filter_state() != 'bypass',
                set_cmd=self._set_filter_enabled,
                vals=Bool(),
                label='Filter Enabled',
                docstring='Enable/disable filter'
            )

            self.add_parameter(
                'filter_center_freq',
                get_cmd=self._get_filter_fc,
                set_cmd=lambda val: self._set_filter(val, None, None),
                vals=Numbers(2.0, 18.0),
                unit='GHz',
                label='Filter Center Frequency',
                docstring='Filter center frequency in GHz'
            )

            self.add_parameter(
                'filter_bandwidth',
                get_cmd=self._get_filter_bw,
                set_cmd=lambda val: self._set_filter(None, val, None),
                vals=Numbers(0.5, 4.0),
                unit='GHz',
                label='Filter Bandwidth',
                docstring='Filter bandwidth in GHz'
            )

            self.add_parameter(
                'filter_type',
                get_cmd=self._get_filter_type,
                set_cmd=lambda val: self._set_filter(None, None, val),
                vals=Enum('bandpass', 'highpass', 'lowpass', 'bypass'),
                label='Filter Type',
                docstring='Filter type'
            )

        # DC Chain parameters
        if self._has_dc_chain:
            self.add_parameter(
                'dc_enabled',
                get_cmd=self._get_dc_enabled,
                set_cmd=self._set_dc_enabled,
                vals=Bool(),
                label='DC Output Enabled',
                docstring='Enable/disable DC output'
            )

    def _get_rf_enabled(self) -> bool:
        """Check if RF output is enabled."""
        # TODO: RF is considered enabled
        return True

    def _set_rf_enabled(self, enable: bool):
        """Enable or disable RF output."""
        if enable:
            # Enable with default attenuation values (10 dB each)
            # Cannot read back current values through Pyro4
            self.root_instrument._soc.rfb_set_gen_rf(self._channel, 10, 10)
        else:
            # Disable by setting max attenuation
            self.root_instrument._soc.rfb_set_gen_rf(self._channel, 31.75, 31.75)

    def _set_attenuator(self, att1: Optional[float], att2: Optional[float]):
        """Set attenuator values."""
        # Use cached values as defaults
        new_att1 = att1 if att1 is not None else self._att1_cache
        new_att2 = att2 if att2 is not None else self._att2_cache

        # Update cache
        if att1 is not None:
            self._att1_cache = att1
        if att2 is not None:
            self._att2_cache = att2

        self.root_instrument._soc.rfb_set_gen_rf(self._channel, new_att1, new_att2)

    def _get_filter_state(self) -> str:
        """Get current filter state."""
        try:
            if hasattr(self._rfb_ch, 'filt') and hasattr(self._rfb_ch.filt, 'band'):
                return self._rfb_ch.filt.band
            return 'bypass'
        except:
            return 'bypass'

    def _get_filter_fc(self) -> float:
        """Get filter center frequency."""
        try:
            if hasattr(self._rfb_ch, 'filt') and hasattr(self._rfb_ch.filt, 'fc'):
                return self._rfb_ch.filt.fc
            return 0.0
        except:
            return 0.0

    def _get_filter_bw(self) -> float:
        """Get filter bandwidth."""
        try:
            if hasattr(self._rfb_ch, 'filt') and hasattr(self._rfb_ch.filt, 'bw'):
                return self._rfb_ch.filt.bw
            return 1.0
        except:
            return 1.0

    def _get_filter_type(self) -> str:
        """Get filter type."""
        state = self._get_filter_state()
        if state == 'bypass':
            return 'bypass'
        elif 'LPF' in state:
            return 'lowpass'
        elif 'HPF' in state:
            return 'highpass'
        elif 'BPF' in state:
            return 'bandpass'
        return 'bypass'

    def _set_filter_enabled(self, enable: bool):
        """Enable or disable filter."""
        if not enable:
            self.root_instrument._soc.rfb_set_gen_filter(self._channel, fc=0, ftype='bypass')
        else:
            # Enable with current or default settings
            fc = self._get_filter_fc() if self._get_filter_fc() > 0 else 6.0
            bw = self._get_filter_bw()
            self.root_instrument._soc.rfb_set_gen_filter(self._channel, fc=fc, ftype='bandpass', bw=bw)

    def _set_filter(self, fc: Optional[float], bw: Optional[float], ftype: Optional[str]):
        """Set filter parameters."""
        current_fc = self._get_filter_fc() if self._get_filter_fc() > 0 else 6.0
        current_bw = self._get_filter_bw()
        current_type = self._get_filter_type()

        new_fc = fc if fc is not None else current_fc
        new_bw = bw if bw is not None else current_bw
        new_type = ftype if ftype is not None else current_type

        self.root_instrument._soc.rfb_set_gen_filter(self._channel, fc=new_fc, ftype=new_type, bw=new_bw)

    def _get_dc_enabled(self) -> bool:
        """Check if DC output is enabled."""
        # DC state cannot be read back through Pyro4
        return False

    def _set_dc_enabled(self, enable: bool):
        """Enable or disable DC output."""
        if enable:
            self.root_instrument._soc.rfb_set_gen_dc(self._channel)


class ADCChannel(InstrumentChannel):
    """
    Channel class for ADC (readout) channels.
    Handles RF attenuators, filters, and gain configuration.
    """

    def __init__(self, parent: Instrument, name: str, channel: int, adc_config: dict, **kwargs):
        super().__init__(parent, name, **kwargs)
        self._channel = channel
        self._adc_config = adc_config


        self._att_cache = 10.0  # Default attenuation

        self._has_rf_chain = True
        self._has_dc_chain = False
        self._has_filter = True
        self._has_attenuator = True

        # # Add basic channel info parameters
        # self.add_parameter(
        #     'channel_index',
        #     get_cmd=lambda: self._channel,
        #     snapshot_value=True,
        #     label='Channel Index',
        #     docstring='Index in the avg_bufs list'
        # )

        # self.add_parameter(
        #     'adc_tile',
        #     get_cmd=lambda: int(adc_config['adc'][0]),
        #     snapshot_value=True,
        #     label='ADC Tile',
        #     docstring='ADC tile number'
        # )

        # self.add_parameter(
        #     'adc_block',
        #     get_cmd=lambda: int(adc_config['adc'][1]),
        #     snapshot_value=True,
        #     label='ADC Block',
        #     docstring='ADC block number'
        # )

        # self.add_parameter(
        #     'sampling_freq',
        #     get_cmd=lambda: self.root_instrument._soc_cfg['rf']['adcs'][adc_config['adc']]['fs'],
        #     unit='MHz',
        #     snapshot_value=True,
        #     label='Sampling Frequency',
        #     docstring='ADC sampling frequency in MHz'
        # )

        # RF Chain parameters (attenuator)
        if self._has_rf_chain and self._has_attenuator:
            self.add_parameter(
                'rf_enabled',
                get_cmd=self._get_rf_enabled,
                set_cmd=self._set_rf_enabled,
                vals=Bool(),
                label='RF Input Enabled',
                docstring='Enable/disable RF input'
            )

            self.add_parameter(
                'att',
                get_cmd=lambda: self._att_cache,
                set_cmd=self._set_attenuator,
                vals=Numbers(0, 31.75),
                unit='dB',
                label='Attenuator',
                docstring='Input attenuation (0-31.75 dB)'
            )

        # Filter parameters
        if self._has_filter:
            self.add_parameter(
                'filter_enabled',
                get_cmd=lambda: self._get_filter_state() != 'bypass',
                set_cmd=self._set_filter_enabled,
                vals=Bool(),
                label='Filter Enabled',
                docstring='Enable/disable filter'
            )

            self.add_parameter(
                'filter_center_freq',
                get_cmd=self._get_filter_fc,
                set_cmd=lambda val: self._set_filter(val, None, None),
                vals=Numbers(2.0, 18.0),
                unit='GHz',
                label='Filter Center Frequency',
                docstring='Filter center frequency in GHz'
            )

            self.add_parameter(
                'filter_bandwidth',
                get_cmd=self._get_filter_bw,
                set_cmd=lambda val: self._set_filter(None, val, None),
                vals=Numbers(0.5, 4.0),
                unit='GHz',
                label='Filter Bandwidth',
                docstring='Filter bandwidth in GHz'
            )

            self.add_parameter(
                'filter_type',
                get_cmd=self._get_filter_type,
                set_cmd=lambda val: self._set_filter(None, None, val),
                vals=Enum('bandpass', 'highpass', 'lowpass', 'bypass'),
                label='Filter Type',
                docstring='Filter type'
            )

        # DC Chain parameters (gain)
        if self._has_dc_chain:
            self.add_parameter(
                'dc_enabled',
                get_cmd=self._get_dc_enabled,
                set_cmd=self._set_dc_enabled,
                vals=Bool(),
                label='DC Input Enabled',
                docstring='Enable/disable DC input'
            )

            self.add_parameter(
                'dc_gain',
                get_cmd=self._get_dc_gain,
                set_cmd=lambda val: self.root_instrument._soc.rfb_set_ro_dc(self._channel, val),
                vals=Numbers(-6, 26),
                unit='dB',
                label='DC Gain',
                docstring='DC input gain (-6 to 26 dB)'
            )

    def _get_rf_enabled(self) -> bool:
        """Check if RF input is enabled."""
        #TODO
        return True

    def _set_rf_enabled(self, enable: bool):
        """Enable or disable RF input."""
        if enable:
            # Enable with default attenuation (10 dB)
            self.root_instrument._soc.rfb_set_ro_rf(self._channel, 10)
        else:
            self.root_instrument._soc.rfb_set_ro_rf(self._channel, 31.75)

    def _set_attenuator(self, val: float):
        """Set attenuator value and cache it."""
        self._att_cache = val
        self.root_instrument._soc.rfb_set_ro_rf(self._channel, val)

    def _get_filter_state(self) -> str:
        """Get current filter state."""
        try:
            if hasattr(self._rfb_ch, 'filt') and hasattr(self._rfb_ch.filt, 'band'):
                return self._rfb_ch.filt.band
            return 'bypass'
        except:
            return 'bypass'

    def _get_filter_fc(self) -> float:
        """Get filter center frequency."""
        try:
            if hasattr(self._rfb_ch, 'filt') and hasattr(self._rfb_ch.filt, 'fc'):
                return self._rfb_ch.filt.fc
            return 0.0
        except:
            return 0.0

    def _get_filter_bw(self) -> float:
        """Get filter bandwidth."""
        try:
            if hasattr(self._rfb_ch, 'filt') and hasattr(self._rfb_ch.filt, 'bw'):
                return self._rfb_ch.filt.bw
            return 1.0
        except:
            return 1.0

    def _get_filter_type(self) -> str:
        """Get filter type."""
        state = self._get_filter_state()
        if state == 'bypass':
            return 'bypass'
        elif 'LPF' in state:
            return 'lowpass'
        elif 'HPF' in state:
            return 'highpass'
        elif 'BPF' in state:
            return 'bandpass'
        return 'bypass'

    def _set_filter_enabled(self, enable: bool):
        """Enable or disable filter."""
        if not enable:
            self.root_instrument._soc.rfb_set_ro_filter(self._channel, fc=0, ftype='bypass')
        else:
            fc = self._get_filter_fc() if self._get_filter_fc() > 0 else 6.0
            bw = self._get_filter_bw()
            self.root_instrument._soc.rfb_set_ro_filter(self._channel, fc=fc, ftype='bandpass', bw=bw)

    def _set_filter(self, fc: Optional[float], bw: Optional[float], ftype: Optional[str]):
        """Set filter parameters."""
        current_fc = self._get_filter_fc() if self._get_filter_fc() > 0 else 6.0
        current_bw = self._get_filter_bw()
        current_type = self._get_filter_type()

        new_fc = fc if fc is not None else current_fc
        new_bw = bw if bw is not None else current_bw
        new_type = ftype if ftype is not None else current_type

        self.root_instrument._soc.rfb_set_ro_filter(self._channel, fc=new_fc, ftype=new_type, bw=new_bw)

    def _get_dc_enabled(self) -> bool:
        """Check if DC input is enabled."""
        # DC state cannot be read back through Pyro4
        return False

    def _set_dc_enabled(self, enable: bool):
        """Enable or disable DC input."""
        if enable:
            gain = self._get_dc_gain()
            self.root_instrument._soc.rfb_set_ro_dc(self._channel, gain)

    def _get_dc_gain(self) -> float:
        """Get DC gain."""
        # DC gain cannot be read back through Pyro4
        # Store locally if needed, or return default
        return 0.0


class BiasChannel(InstrumentChannel):
    """
    Channel class for DC bias outputs.
    """

    def __init__(self, parent: Instrument, name: str, channel: int, **kwargs):
        super().__init__(parent, name, **kwargs)
        self._channel = channel
        self._voltage_step = 0.0001  # 0.1 mV step size
        self._resistor = 500 #Ohms
        self.step_delay_sec = 0.01 #sec

        self.add_parameter(
            'channel_index',
            get_cmd=lambda: self._channel,
            snapshot_value=True,
            label='Channel Index',
            docstring='Bias channel index'
        )

        self.add_parameter(
            'voltage',
            get_cmd=lambda: self.root_instrument._soc.rfb_get_bias(self._channel),
            set_cmd=lambda val: self.ramp_voltage(val),
            vals=Numbers(-10, 10),
            unit='V',
            label='Bias Voltage',
            docstring='Bias voltage (-10 to 10 V)'
        )

        self.add_parameter(
            'current',
            get_cmd=lambda: self.root_instrument._soc.rfb_get_bias(self._channel)/self._resistor * 1e3,
            set_cmd=lambda val: self.ramp_voltage(val*self._resistor*1e-3),
            vals=Numbers(-100, 100),
            unit='mA',
            label='Bias Current',
            docstring='Bias Current'
        )


        # self.add_parameter(
        #     'current_step',
        #     get_cmd=lambda : self._voltage_step/self._resistor,
        #     set_cmd=lambda new_current_step: setattr(self, '_voltage_step', new_current_step*self._resistor),
        #     vals=Numbers(0, 1),
        #     unit='V',
        #     label='Voltage Step',
        #     docstring='Voltage step size (0 to 1 V)'
        # )

        self.add_parameter(
            'current_ramp_rate',
            get_cmd=lambda : self._voltage_step/self._resistor * 1/self.step_delay_sec * 1e6,
            set_cmd=lambda new_current_step: setattr(self, '_voltage_step', new_current_step*self._resistor* self.step_delay_sec * 1e-6),
            vals=Numbers(0, 500),
            unit='uA per Sec',
            label='Current Ramp Rate',
            docstring='Usually around 10uA per sec'
        )

        self.add_parameter(
            'Resistor',
            get_cmd = lambda : self._resistor,
            set_cmd = lambda value: setattr(self, '_resistor', value),
            unit = 'Ohms',
            label = 'Resistor Value',
            docstring='Resistor used to calculate the ramp rate'
        )

    def ramp_voltage(self, target_voltage: float, step_delay_sec: float = 0.01):
        """Ramp bias voltage to target value in steps."""
        import time
        from numpy import arange, round
        if self.step_delay_sec : step_delay_sec = self.step_delay_sec

        current_voltage = self.root_instrument._soc.rfb_get_bias(self._channel)
        step = self._voltage_step if target_voltage > current_voltage else -self._voltage_step
        logging.info(__name__ + f"Ramping bias channel {self._channel}: {current_voltage}V ({current_voltage/self._resistor * 1e3}mA) to {target_voltage}V({target_voltage/self._resistor * 1e3}mA) in steps of {step}V ({step/self._resistor * 1e3}mA) with {step_delay_sec} s delay")

        for voltage in arange(current_voltage, target_voltage, step):
            voltage = round(voltage, 7) # round off random floats to 7 decimals
            self.root_instrument._soc.rfb_set_bias(self._channel, voltage)
            time.sleep(step_delay_sec)

        # Ensure final voltage is set
        self.root_instrument._soc.rfb_set_bias(self._channel, target_voltage)


class QickSoC_RFBoard(Instrument):
    """
    QCoDes driver for QICK SoC with RF Board.

    Automatically detects available DAC and ADC channels and creates
    appropriate parameters for attenuators, filters, DC biases, etc.

    Parameters
    ----------
    name : str
        Name of the instrument
    soc : QickSoc or RFQickSoc
        The QICK SoC object (already initialized)
    **kwargs
        Additional keyword arguments passed to Instrument

    Examples
    --------
    >>> from qick import QickSoc
    >>> soc = QickSoc()
    >>> qick_driver = QickSoC_RFBoard('qick', soc=soc)
    >>> qick_driver.dac0.att1(10)  # Set attenuator 1 to 10 dB
    >>> qick_driver.adc0.filter_center_freq(6.5)  # Set filter center freq to 6.5 GHz
    """

    def __init__(self, name: str,nameserver_host,nameserver_port,nameserver_name, **kwargs):
        super().__init__(name, **kwargs)


        import Pyro4
        from qick import QickConfig
        Pyro4.config.SERIALIZER = 'pickle'
        Pyro4.config.PICKLE_PROTOCOL_VERSION = 4
        ns = Pyro4.locateNS(host=nameserver_host, port=nameserver_port)
        soc = Pyro4.Proxy(ns.lookup(nameserver_name))
        soccfg = QickConfig(soc.get_cfg())
        self._soc = soc
        self._soc_cfg = soccfg.get_cfg()  #get the dictionary version of the soc config

        # self._soc = soc
        # self._soc_cfg = soccfg.get_cfg()  #get the dictionary version of the soc config

        # Detect and add DAC channels
        self._detect_dac_channels()

        # Detect and add ADC channels
        self._detect_adc_channels()

        # Detect and add bias channels
        self._detect_bias_channels()

        # Add board-level parameters
        self._add_board_parameters()

        logging.info(__name__ + f"Initialized QickSoC_RFBoard driver '{name}' with "
                   f"{len(self.dac_channels)} DAC channels, "
                   f"{len(self.adc_channels)} ADC channels, and "
                   f"{len(self.bias_channels) if hasattr(self, 'bias_channels') else 0} bias channels")

    def _detect_dac_channels(self):
        """Detect and create DAC channel instances."""
        self.dac_channels = ChannelList(self, "dac_channels", DACChannel)

        if 'gens' in self._soc_cfg:
            for i, gen_config in enumerate(self._soc_cfg['gens']):
                channel = DACChannel(self, f'dac{i}', i, gen_config)
                self.dac_channels.append(channel)
                self.add_submodule(f'dac{i}', channel)

    def _detect_adc_channels(self):
        """Detect and create ADC channel instances."""
        self.adc_channels = ChannelList(self, "adc_channels", ADCChannel)

        if 'readouts' in self._soc_cfg:
            for i, adc_config in enumerate(self._soc_cfg['readouts']):
                channel = ADCChannel(self, f'adc{i}', i, adc_config)
                self.adc_channels.append(channel)
                self.add_submodule(f'adc{i}', channel)

    def _detect_bias_channels(self):
        """Detect and create bias channel instances."""
        self.bias_channels = ChannelList(self, "bias_channels", BiasChannel)

        # Typically there are 8 bias channels on RF boards
        try:
            # Try to access bias channel 0 to see if it's supported
            self._soc.rfb_get_bias(0)
            # If successful, assume 8 bias channels (standard)
            for i in range(8):
                channel = BiasChannel(self, f'bias{i}', i)
                self.bias_channels.append(channel)
                self.add_submodule(f'bias{i}', channel)
        except:
            # No bias channels available
            pass

    def _add_board_parameters(self):
        """Add board-level parameters."""
        self.add_parameter(
            'board_type',
            get_cmd=lambda: str(self._soc_cfg['board']),
            snapshot_value=True,
            label='Board Type',
            docstring='QICK board type (ZCU111, ZCU216, etc.)'
        )

        self.add_parameter(
            'firmware_version',
            get_cmd=lambda: str(self._soc_cfg['fw_timestamp']),
            snapshot_value=True,
            label='Firmware Version',
            docstring='Firmware build timestamp'
        )

        self.add_parameter(
            'software_version',
            get_cmd=lambda: str(self._soc_cfg['sw_version']),
            snapshot_value=True,
            label='Software Version',
            docstring='QICK software version'
        )
        try:
            self.add_parameter(
                'ref_clock_freq',
                get_cmd=lambda: float(self._soc_cfg['refclk_freq']),
                unit='MHz',
                snapshot_value=True,
                label='Reference Clock Frequency',
                docstring='RF reference clock frequency in MHz'
            )
        except:
            pass

    def get_idn(self) -> Dict[str, Optional[str]]:
        """Return instrument identification."""
        return {
            'vendor': 'QICK',
            'model': self._soc_cfg['board'],
            'serial': None,
            'firmware': self._soc_cfg['fw_timestamp']
        }

    def print_overview(self):
        """Print a comprehensive overview of all channels and their capabilities."""
        print("="*80)
        print(f"QICK SoC RF Board Driver: {self.name}")
        print("="*80)
        print(f"Board: {self.board_type()}")
        print(f"Firmware: {self.firmware_version()}")
        print(f"Software: {self.software_version()}")
        if 'refclk_freq' in self._soc_cfg:
            print(f"Reference Clock: {self.ref_clock_freq()} MHz")

        print("\n" + "="*80)
        print(f"DAC Channels ({len(self.dac_channels)}):")
        print("="*80)
        for dac in self.dac_channels:
            print(f"\n{dac.short_name}:")
            # print(f"  Tile: {dac.dac_tile()}, Block: {dac.dac_block()}")
            # print(f"  Sampling Freq: {dac.sampling_freq()} MHz")
            if hasattr(dac, 'att1'):
                print(f"  RF Chain: Yes (Att1={dac.att1()} dB, Att2={dac.att2()} dB)")
            if hasattr(dac, 'filter_enabled'):
                filter_status = "Enabled" if dac.filter_enabled() else "Disabled"
                print(f"  Filter: {filter_status}")
                if dac.filter_enabled():
                    print(f"    Type: {dac.filter_type()}, FC: {dac.filter_center_freq()} GHz, BW: {dac.filter_bandwidth()} GHz")
            if hasattr(dac, 'dc_enabled'):
                print(f"  DC Chain: Available")

        print("\n" + "="*80)
        print(f"ADC Channels ({len(self.adc_channels)}):")
        print("="*80)
        for adc in self.adc_channels:
            print(f"\n{adc.short_name}:")
            # print(f"  Tile: {adc.adc_tile()}, Block: {adc.adc_block()}")
            # print(f"  Sampling Freq: {adc.sampling_freq()} MHz")
            if hasattr(adc, 'att'):
                print(f"  RF Chain: Yes (Att={adc.att()} dB)")
            if hasattr(adc, 'filter_enabled'):
                filter_status = "Enabled" if adc.filter_enabled() else "Disabled"
                print(f"  Filter: {filter_status}")
                if adc.filter_enabled():
                    print(f"    Type: {adc.filter_type()}, FC: {adc.filter_center_freq()} GHz, BW: {adc.filter_bandwidth()} GHz")
            if hasattr(adc, 'dc_enabled'):
                print(f"  DC Chain: Available (Gain: {adc.dc_gain() if hasattr(adc, 'dc_gain') else 'N/A'} dB)")

        if hasattr(self, 'bias_channels') and len(self.bias_channels) > 0:
            print("\n" + "="*80)
            print(f"Bias Channels ({len(self.bias_channels)}):")
            print("="*80)
            for bias in self.bias_channels:
                print(f"{bias.short_name}: {bias.voltage()} V")

        print("\n" + "="*80)


if __name__ == "__main__":

    print("I want to quit")
