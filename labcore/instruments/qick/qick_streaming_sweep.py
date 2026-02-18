"""Streaming helper for using QICK programs with the Sweep framework.

This module provides `QickBoardStreamingSweep`, a decorator that uses a program's
`stream_acquire()` method (if present) to receive incremental IQ chunks and
yield them as dictionaries with independent axes.

Behavior:
- Precomputes sweep-value arrays for independent specs (PulseVariable, TimeVariable etc).
- Contracts round and per-round rep indices into a single 'rep' = (round + per-round rep) axis.
- For each streaming event/blocking acquire call, yields a single aggregated
  dictionary containing:
  - 'ro_channel_and_readout_trigger': identifier string for each data point
    (format: "<channel>_r<readout_idx>")
  - 'channel': list of channel indices corresponding to each data point
  - 'readout': list of readout trigger indices corresponding to each data point
  - 'data': list of complex IQ values flattened in arrival order
  - 'rep': list of global rep indices (collapsed from round and per-round indices)
  - One key per independent sweep spec, mapping to a list of sweep values

- Falls back to calling `acquire()` (blocking) if `stream_acquire()` is not
  available on the program, yielding the same dictionary structure at the end just with all of the experiment's data at once

All data is yielded as raw in dictionaries without any processing, only reshaping and tagging/labelling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from collections import OrderedDict
from itertools import cycle as it_cycle, product as it_product, islice as it_islice, tee as it_tee
import numpy as np

from labcore.measurement import DataSpec
from labcore.measurement.record import make_data_spec

from labcore.measurement.sweep import AsyncRecord
import logging

logger = logging.getLogger(__name__)


config = None



@dataclass
class ComplexQICKData(DataSpec):
    """Complex IQ readout data spec with full dimensionality tracking.

    Attributes:
        i_data_stream: label for I component (default 'I')
        q_data_stream: label for Q component (default 'Q')
    """
    i_data_stream: str = 'I'
    q_data_stream: str = 'Q'
    # ro_ch: Optional[int] = None
    # sweep_dims: Optional[tuple] = None
    # n_readouts: Optional[int] = None

    def set_name(self, name: str) -> 'ComplexQICKData':
        """Change the name of this dataspec and return self for chaining."""
        self.name = name
        return self


@dataclass
class PulseVariable(DataSpec):
    pulse_parameter: Optional[str] = None
    sweep_parameter: Optional[str] = None
    loop_idx: Optional[int] = None
    loop_name: Optional[str] = None


@dataclass
class TimeVariable(DataSpec):
    time_parameter: Optional[str] = None
    loop_idx: Optional[int] = None
    loop_name: Optional[str] = None

@dataclass
class RepVariable(DataSpec):
    """Rep number variable spec for QICK sweeps."""
    loop_idx: Optional[int] = None
    loop_name: Optional[str] = 'reps'




class QickBoardStreamingSweep(AsyncRecord):
    """Decorator to run QICK programs and save streaming data incrementally.
    The data specs must be defined in loop/measurement order. Reps are defined automatically.
    """

    def __init__(self, *specs, **kwargs):
        self.communicator = {}
        self.unordered_nonexhaustive_specs = list(specs)
        self.specs = []
        for s in specs:
            spec = make_data_spec(s)
            self.specs.append(spec)

    def setup(self, func, *args, **kwargs):

        if config is None:
            raise Exception("QickSweep: config is not set")

        self.config = config
        conf = config.config() #TODO change this to be passed from kwargs, not from global
        qick_program = func(soccfg=conf[0], reps=conf[1].get('reps'), final_delay=conf[1].get('final_delay'), cfg=conf[1])
        self.communicator['qick_program'] = qick_program
        prog = qick_program
        # at this point, the qick program has been compiled into asm. we can retreive loop info.

        ################################################
        ### ========= Get Independnent Sweep Axes ==============
        # get loops info
        loop_dict : OrderedDict = prog.loop_dict # { (reps, 10000), (gain_sweep, 100), (freq_sweep, 50) ... }
        loop_dims : tuple = prog.loop_dims

        # construct data specs in the order of loops in the program
        loop_idx = 0
        ordered_specs = []
        for name, dim in loop_dict.items():
            # check if spec loop name exists
            # if it exists, add it according to order
            if name == 'reps':
                ordered_specs.append(RepVariable(name='rep', loop_name='reps', loop_idx=loop_idx))
                loop_idx +=1
                continue

            elif any( (spec.loop_name == name) for spec in self.unordered_nonexhaustive_specs):
                for spec in self.unordered_nonexhaustive_specs:
                    if spec.loop_name == name:
                        spec.loop_idx = loop_idx
                        loop_idx +=1
                        new_spec = make_data_spec(spec)
                        ordered_specs.append(new_spec)
                        break
                    else : pass

            else:
                raise Exception(f"QickStreamingSweep: Spec for loop '{name}' not provided in specs.")
        self.communicator['ordered_specs'] = ordered_specs

        # ========== Get sweep arrays ============
        # prepare sweep value iterators for independent specs, that will render sweep values
        self.sweep_arrays: OrderedDict[str, Optional[np.ndarray]] = OrderedDict()

        for ds in ordered_specs:
            if isinstance(ds, PulseVariable):
                arr = prog.get_pulse_param(ds.pulse_parameter, ds.sweep_parameter, as_array=True)
                self.sweep_arrays[ds.name] = np.asarray(arr).flatten()
            elif isinstance(ds, TimeVariable):
                arr = prog.get_time_param(ds.time_parameter, 't', as_array=True)
                self.sweep_arrays[ds.name] = np.asarray(arr).flatten()
                # coalesce rounds and reps into single rep axis
            elif isinstance(ds, RepVariable):
                self.sweep_arrays[ds.name] = range(1, prog.reps * conf[1].get('rounds') + 1,  1 )

        # get number of triggers etc
        readout_spec_number = sum([1 for ds in self.unordered_nonexhaustive_specs if isinstance(ds, ComplexQICKData)])

        # Extract ComplexQICKData spec names in order for use in collect()
        complex_data_specs = [ds for ds in self.unordered_nonexhaustive_specs if isinstance(ds, ComplexQICKData)]
        self.communicator['complex_data_spec_names'] = [ds.name for ds in complex_data_specs]

        readout_dict = prog.ro_chs #outputs ordered dict, 'ro_ch_number : int' : { '#trigs': int, 'length': int, 'length_us': float, ...}
        reads_per_shot = [ro['trigs'] for ro in prog.ro_chs.values()] # list of [# readouts for ro channel , # readouts for ro channel, ... ]

        # sanity check
        assert sum(reads_per_shot) == readout_spec_number, "Mismatch in defined readout specs and program readout triggers. Check defined data specs."

        # since data_dict requires each data point with it's axes value, we create an iterator to produce those points
        # We use product since the sweep values should be nested in the order of executed loops in Qick
        self.final_iterable = it_product(*self.sweep_arrays.values())


    def collect(self, len_normalize: bool = True, stream: bool = True, **kwargs):
        """Collect streaming IQ data from QICK program and yield aggregated dictionaries.
        Falls back to blocking acquire() if streaming not available.

        Parameters:
        - `rounds`: number of averaging rounds forwarded to `stream_acquire`.
        - `len_normalize`: if True, normalizes IQ points by their readout window values.
        - `stream`: if True, uses streaming API; otherwise falls back to blocking acquire.

        Yields:
        - dict with keys: { 'ro_channel_and_readout_idx' : np.array(IQ Values),
                            'reps' : np.array(rep number corresponding to data above),
                            'loop_name' : np.array(sweep loop value corresponding to data above),}
        """
        prog = self.communicator['qick_program']
        complex_data_spec_names = self.communicator.get('complex_data_spec_names', [])

        # ========== Collect streaming data ================
        stream_fn = getattr(prog, 'stream_acquire', None)
         # Use streaming API if available; otherwise fall back to a blocking acquire
        if callable(stream_fn):
            gen = prog.stream_acquire(
                self.config.soc,
                rounds=kwargs.get('rounds', 1),
                progress=True,
                len_normalize=len_normalize,
                remove_offset=False,
                include_full=False,
                return_end_of_exp_raw=False,
            )
            # iterate events and yield a single aggregated dictionary per data event
            for ev in gen:
                yield_dict: Dict[str, Any] = {}
                if ev['event'] == 'data':
                    partial = ev['partial']
                    round_idx = ev['round']
                    count_start_flat, count_stop_flat = ev.get('rep_slice', (0, 0))  #  in per-round flattened space

                    # Get ComplexQICKData spec names (in definition order) to use as keys
                    spec_idx = 0
                    warning_logged = False

                    for ch in partial.keys():
                        comp_data = partial[ch].dot([1,1j]) #data in shape of (new_points, nreads)
                        for ro_no in range(comp_data.shape[1]):
                            # Use spec name if available, otherwise fall back to generated key
                            # using extra logic in case some dataspec in decorator was mislabelled/undefined. Result : still collect data but warn user.
                            if spec_idx < len(complex_data_spec_names):
                                key = complex_data_spec_names[spec_idx]
                            else:
                                if not warning_logged:
                                    logger.warning(
                                        f"ComplexQICKData specs mismatch: Expected more specs than defined, received more data from QICK. All collected data is probably mislabeled. ")
                                    warning_logged = True
                                key = f"roch{ch}_read{ro_no}"
                            yield_dict[key] = comp_data[:, ro_no]
                            spec_idx += 1
                            # add sweep values
                    tp = [self.final_iterable.__next__() for i in range(count_stop_flat-count_start_flat)]
                    sweep_value_array = np.array(tp) # array of sweep values for this channel

                    for spec_name in self.sweep_arrays.keys():
                        yield_dict[spec_name] = sweep_value_array[:, list(self.sweep_arrays.keys()).index(spec_name)]

                    yield yield_dict
                    # for i in range(partial[0].shape[0]):
                    #     r = {}
                    #     for k, v in yield_dict.items():
                    #         r[k] = v[i]
                    #     yield r

                elif ev.get('event') == 'round-complete':
                    break
                    # return # end of experiment


        else: #TODO : acquire() fallback
            logger.critical("Streaming not callable, check qick libraries. Falling back to blocking acquire().")




    # def collect_streaming(self, rounds: int = 1, include_full: bool = False, remove_offset: bool = False):
    #     """Stream or acquire IQ data and yield aggregated dictionaries.

    #     Yields one dictionary per streaming event (or one for blocking acquire)
    #     containing complex IQ data with corresponding metadata: channel, readout
    #     trigger indices, rep numbers, and sweep parameter values.

    #     Parameters:
    #     - `rounds`: number of averaging rounds forwarded to `stream_acquire`.
    #     - `include_full`: passed to `stream_acquire` if available.
    #     - `remove_offset`: passed to `stream_acquire` if available.

    #     Yields:
    #     - dict with keys: 'ro_channel_and_readout_trigger', 'channel', 'readout',
    #       'data', 'rep', and one key per sweep spec.
    #     """
    #     prog = self.communicator['qick_program']
    #     cfg = self.config.config()[1]

    #     # ------- Figure out independent axes and non-IQ dependent specs------------
    #     # Extract loop dimensions and readout configuration from program,
    #     # this contains the loops as qick will execute them ; including reps
    #     # We try to follow QICK's terminology of shot, rep, round, loops.
    #     loop_dims = getattr(prog, 'loop_dims', []) # contains rep number, could be any one of them depending on execution
    #     loop_dims = tuple(loop_dims) # because order matters, immutable
    #     # number of per-round reps (flattened loop length)
    #     total_reps_per_round = int(np.prod(loop_dims)) if len(loop_dims) > 0 else 1

    #     # Prepare sweep-value arrays for independent specs so we can map flat indices -> values
    #     sweep_arrays: Dict[str, np.ndarray] = {}
    #     for ds in self.specs:
    #         #get independent specs only
    #         if ds.depends_on is None and not isinstance(ds, ComplexQICKData):
    #             spec_name = ds.name
    #             try:
    #                 if isinstance(ds, PulseVariable):
    #                     arr = prog.get_pulse_param(ds.pulse_parameter, ds.sweep_parameter, as_array=True)
    #                 elif isinstance(ds, TimeVariable):
    #                     arr = prog.get_time_param(ds.time_parameter, 't', as_array=True) * (cfg['n_echoes'] + 1)
    #                 else:
    #                     arr = np.asarray(ds.default if hasattr(ds, 'default') else [])
    #             except Exception:
    #                 logger.error(f"Could not get sweep array for spec {spec_name}, will fill with None")
    #                 arr = np.array([])

    #             arr = np.asarray(arr) # ensure numpy array, qick may return lists
    #             if arr.size == 0:
    #                 # fill with None so indexing is safe
    #                 sweep_arrays[spec_name] = np.array([None] * total_reps_per_round, dtype=object)
    #             else:
    #                 # try to flatten arr into per-round flattened ordering
    #                 if arr.size == total_reps_per_round:
    #                     sweep_arrays[spec_name] = arr.flatten()
    #                 else:
    #                     # if arr has same shape as loop_dims, flatten in C-order
    #                     if arr.shape == loop_dims:
    #                         sweep_arrays[spec_name] = np.asarray(arr).flatten()
    #                     else:
    #                         # best-effort: resize/repeat to match length
    #                         sweep_arrays[spec_name] = np.resize(np.asarray(arr).flatten(), total_reps_per_round)

    #     aggregated_sweep_iterable =
    #     # Use streaming API if available; otherwise fall back to a blocking acquire
    #     stream_fn = getattr(prog, 'stream_acquire', None)
    #     if callable(stream_fn):
    #         gen = prog.stream_acquire(
    #             self.config.soc,
    #             rounds=rounds,
    #             include_full=include_full,
    #             remove_offset=remove_offset,
    #             progress=True,
    #             return_end_of_exp_raw=False,
    #         )

    #         # iterate events and yield a single aggregated dictionary per data event
    #         for ev in gen:
    #             if ev.get('event') == 'data':
    #                 partial = ev.get('partial', {})
    #                 round_idx = int(ev.get('round', 0))
    #                 rep_slice = ev.get('rep_slice', (0, 0))  # (start_rep, stop_rep) in per-round flattened space

    #                 rep_start_flat, rep_stop_flat = rep_slice

    #                 # aggregated lists across channels and readouts for this event
    #                 data_list: List[complex] = []
    #                 rep_list: List[int] = []
    #                 ro_ch_readout_list: List[str] = []
    #                 channel_list: List[Any] = []
    #                 readout_list: List[int] = []
    #                 sweep_values_for_specs: Dict[str, List[Any]] = {k: [] for k in sweep_arrays.keys()}

    #                 for ch, arr in partial.items():
    #                     arr = np.asarray(arr)
    #                     if arr.size == 0:
    #                         continue
    #                     new_points, nreads, _ = arr.shape

    #                     for i in range(new_points):
    #                         per_round_flat = rep_start_flat + i
    #                         # global rep index collapses round and per-round index
    #                         global_rep = round_idx * total_reps_per_round + per_round_flat

    #                         for readout_idx in range(nreads):
    #                             val = arr[i, readout_idx, 0] + 1j * arr[i, readout_idx, 1]
    #                             data_list.append(val)
    #                             rep_list.append(global_rep)
    #                             ro_ch_readout_list.append(f"{ch}_r{readout_idx}")
    #                             channel_list.append(ch)
    #                             readout_list.append(readout_idx)

    #                             for spec_name, vals in sweep_arrays.items():
    #                                 try:
    #                                     sweep_values_for_specs[spec_name].append(vals[per_round_flat])
    #                                 except Exception:
    #                                     sweep_values_for_specs[spec_name].append(None)

    #                 out: Dict[str, Any] = {
    #                     'ro_channel_and_readout_trigger': ro_ch_readout_list,
    #                     'channel': channel_list,
    #                     'readout': readout_list,
    #                     'data': data_list,
    #                     'rep': rep_list,
    #                 }
    #                 out.update(sweep_values_for_specs)

    #                 yield out

    #             # ignore other event types for now (e.g., 'round-complete')
    #         return

    #     else:
    #         # blocking fallback: acquire full buffers then yield same structured dicts
    #         try:
    #             data = prog.acquire(self.config.soc, progress=True)
    #         except Exception:
    #             raise

    #         channels = data[0] if isinstance(data, tuple) else data

    #         # aggregate across channels for blocking acquire
    #         data_list: List[complex] = []
    #         rep_list: List[int] = []
    #         ro_ch_readout_list: List[str] = []
    #         channel_list: List[Any] = []
    #         readout_list: List[int] = []
    #         sweep_values_for_specs: Dict[str, List[Any]] = {k: [] for k in sweep_arrays.keys()}

    #         for ch_idx, arr in enumerate(channels):
    #             arr = np.asarray(arr)
    #             # arr expected shape: (nreps, nreads, 2)
    #             if arr.ndim < 3:
    #                 continue
    #             nreps, nreads, _ = arr.shape

    #             for rep in range(nreps):
    #                 for readout_idx in range(nreads):
    #                     val = arr[rep, readout_idx, 0] + 1j * arr[rep, readout_idx, 1]
    #                     data_list.append(val)
    #                     rep_list.append(rep)
    #                     ro_ch_readout_list.append(f"{ch_idx}_r{readout_idx}")
    #                     channel_list.append(ch_idx)
    #                     readout_list.append(readout_idx)

    #                     for spec_name, vals in sweep_arrays.items():
    #                         try:
    #                             sweep_values_for_specs[spec_name].append(vals[rep])
    #                         except Exception:
    #                             sweep_values_for_specs[spec_name].append(None)

    #         out: Dict[str, Any] = {
    #             'ro_channel_and_readout_trigger': ro_ch_readout_list,
    #             'channel': channel_list,
    #             'readout': readout_list,
    #             'data': data_list,
    #             'rep': rep_list,
    #         }
    #         out.update(sweep_values_for_specs)

    #         yield out
    #         return

__all__ = ["QickBoardStreamingSweep", "ComplexQICKData", "PulseVariable", "TimeVariable"]
