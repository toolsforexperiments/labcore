"""Streaming helper for using QICK programs with the Sweep framework.

This module provides `QickBoardStreamingSweep`, a variant of the existing
`QickBoardSweep` that uses a program's `stream_acquire()` method (if present)
to receive incremental IQ chunks and write them to disk as they arrive.

Behavior:
- Saves static/metadata fields (pulse/time/independent variables) once at
  the start of the measurement.
- For complex IQ dependent specs, appends incoming complex samples to an
  HDF5 dataset in the file so data is persisted incrementally.
- Falls back to calling `acquire()` if `stream_acquire()` is not available on
  the program.

The implementation intentionally keeps the HDF5 layout simple: a dataset is
created per complex dependent spec (or per RO channel, if no matching spec is
provided). Each such dataset is created with `maxshape=(None,)` and is
appended to as data arrives.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from labcore.measurement import DataSpec
from labcore.measurement.record import make_data_spec

from labcore.measurement.sweep import AsyncRecord

from labcore.measurement.storage import TIMESTRFORMAT


@dataclass
class ComplexQICKData(DataSpec):
    """Complex IQ readout data spec with full dimensionality tracking.

    The stored data has shape:
        [RO_channel, rep_number, *sweep_loop_dims, readout_number, IQ_component]

    Attributes:
        i_data_stream: label for I component (default 'I')
        q_data_stream: label for Q component (default 'Q')
        ro_ch: readout channel index (set during collection)
        sweep_dims: tuple of sweep loop dimensions (set from program config)
        n_readouts: number of readout triggers per shot (set during collection)
    """
    i_data_stream: str = 'I'
    q_data_stream: str = 'Q'
    ro_ch: Optional[int] = None
    sweep_dims: Optional[tuple] = None
    n_readouts: Optional[int] = None


@dataclass
class PulseVariable(DataSpec):
    pulse_parameter: Optional[str] = None
    sweep_parameter: Optional[str] = None


@dataclass
class TimeVariable(DataSpec):
    time_parameter: Optional[str] = None


class QickBoardStreamingSweep(AsyncRecord):
    """Decorator to run QICK programs and save streaming data incrementally.

    Usage mirrors `QickBoardSweep` but the `collect_streaming` method accepts
    a `data_dir` and `name` for immediate saving while the program runs.
    """

    def __init__(self, *specs, **kwargs):
        self.communicator = {}
        self.specs = []
        for s in specs:
            spec = make_data_spec(s)
            self.specs.append(spec)

    def setup(self, func, *args, **kwargs):
        if 'config' not in globals() and 'config' in kwargs:
            # keep parity with other sweep helpers if config passed as kw
            self.config = kwargs['config']

        # Expect external configuration to exist in `config` global like
        # the older helper; caller must ensure it's set before setup.
        if not hasattr(self, 'config') or self.config is None:
            raise Exception("QickStreamingSweep: config is not set")

        conf = self.config.config()
        # create program using same convention as qick_sweep_v2
        qick_program = func(soccfg=conf[0], reps=conf[1].get('reps'), final_delay=conf[1].get('final_delay'), cfg=conf[1])
        self.communicator['qick_program'] = qick_program

    def collect_streaming(self, data_dir: str, name: str, rounds: int = 1, add_timestamps: bool = False, include_full: bool = False, remove_offset: bool = False):
        """Run `stream_acquire()` (if available) and save streaming data into an HDF5 file.

        Data is reshaped to [RO_ch, rep, *sweep_dims, readout_no, IQ_component].

        Parameters:
        - `data_dir`, `name`: where to create the HDF5 file (extension `.h5` is added if missing).
        - `rounds`: forwarded to `stream_acquire` when used (soft-averaging rounds).
        - `add_timestamps`: prefix file with a timestamp.
        """
        prog = self.communicator['qick_program']
        cfg = self.config.config()[1]

        # file path
        if not name.endswith('.h5'):
            name = name + '.h5'
        if add_timestamps:
            t = time.localtime()
            time_stamp = time.strftime(TIMESTRFORMAT, t) + '_'
            name = time_stamp + name
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, name)

        # Extract loop dimensions and readout configuration from program
        loop_dims = getattr(prog, 'loop_dims', None) or cfg.get('loop_dims', [1])
        if not isinstance(loop_dims, (list, tuple)):
            loop_dims = [loop_dims]
        loop_dims = tuple(loop_dims)

        # Get RO channels and their readout counts
        ro_chs = getattr(prog, 'ro_chs', {})
        ro_ch_list = list(ro_chs.keys())
        reads_per_shot = [ro_chs[ch].get('trigs', 1) for ch in ro_ch_list]

        # open file and prepare datasets
        with h5py.File(filepath, 'w') as h5f:
            # Save static specs once
            self._save_static_specs(h5f, prog, cfg)

            # Create dataset structure for complex IQ data
            # Shape: [RO_ch, rep, *sweep_dims, readout_no, IQ(2)]
            ds_map = {}
            for ro_idx, ch in enumerate(ro_ch_list):
                nreads = reads_per_shot[ro_idx]

                # Find corresponding ComplexQICKData spec
                complex_specs = [s for s in self.specs if isinstance(s, ComplexQICKData)]
                if ro_idx < len(complex_specs):
                    spec = complex_specs[ro_idx]
                    ds_name = spec.name
                    # Annotate spec with metadata
                    spec.ro_ch = ch
                    spec.sweep_dims = loop_dims
                    spec.n_readouts = nreads
                else:
                    ds_name = f'RO_ch_{ch}'

                # Full target shape: [n_reps, *loop_dims, nreads, 2]
                # We'll collect into this shape across all polls
                target_shape = (rounds,) + loop_dims + (nreads, 2)

                # Create dataset: we'll store complex values reshaped appropriately
                # For now: store with maxshape=(None,) along rep axis, freeze others
                # Full shape would be: (rounds, *loop_dims, nreads, 2) -> we'll store pre-complex
                ds = h5f.create_dataset(
                    ds_name,
                    shape=(rounds,) + loop_dims + (nreads, 2),
                    maxshape=(None,) + loop_dims + (nreads, 2),
                    dtype=np.float64,
                    chunks=True
                )
                ds_map[ds_name] = {
                    'dataset': ds,
                    'ro_ch': ch,
                    'nreads': nreads,
                    'loop_dims': loop_dims,
                }

            # choose streaming if available
            stream_fn = getattr(prog, 'stream_acquire', None)
            if callable(stream_fn):
                gen = prog.stream_acquire(
                    self.config.soc,
                    rounds=rounds,
                    include_full=include_full,
                    remove_offset=remove_offset,
                    progress=True,
                    return_end_of_exp_raw=False
                )

                # Track current position in each round for proper storage
                round_rep_counts = {}  # round -> rep count so far

                # iterate events
                for ev in gen:
                    if ev.get('event') == 'data':
                        partial = ev.get('partial', {})
                        round_idx = ev.get('round', 0)
                        rep_slice = ev.get('rep_slice', (0, 0))  # (start_rep, stop_rep) in flattened space

                        rep_start_flat, rep_stop_flat = rep_slice
                        n_new_reps = rep_stop_flat - rep_start_flat

                        # Convert flattened rep indices to multi-dimensional indices
                        # For now, assume linear progression through cartesian product
                        # Each "rep" corresponds to one shot in the loop

                        for ch, arr in partial.items():
                            # arr shape: (new_points, nreads, 2)
                            # new_points = number of new shots/reps received
                            # nreads = readouts per shot
                            # 2 = IQ components

                            new_points, nreads, _ = arr.shape

                            # Find dataset for this channel
                            complex_specs = [s for s in self.specs if isinstance(s, ComplexQICKData)]
                            try:
                                ro_keys = list(prog.ro_chs.keys())
                                ch_idx = ro_keys.index(ch)
                            except (ValueError, AttributeError):
                                ch_idx = None

                            if ch_idx is not None and ch_idx < len(complex_specs):
                                ds_name = complex_specs[ch_idx].name
                            else:
                                ds_name = f'RO_ch_{ch}'

                            if ds_name not in ds_map:
                                continue

                            ds_info = ds_map[ds_name]
                            ds = ds_info['dataset']

                            # Reshape incoming data to multi-dimensional structure
                            # Input arr: (new_points, nreads, 2)
                            # Target: new_points samples placed into reps [rep_start_flat:rep_stop_flat]
                            # mapped to multi-dimensional indices via loop_dims

                            for i in range(new_points):
                                flat_idx = rep_start_flat + i
                                # Convert flat index to multi-dim indices for loop_dims
                                multi_idx = np.unravel_index(flat_idx, loop_dims)

                                # Store at [round, *multi_idx, :, :]
                                ds_idx = (round_idx,) + tuple(multi_idx) + (slice(None), slice(None))
                                ds[ds_idx] = arr[i]  # shape (nreads, 2)

                    elif ev.get('event') == 'round-complete':
                        # optional: store snapshot of full buffers as groups
                        round_idx = ev.get('round', 0)
                        round_raw = ev.get('round_raw', [])
                        # create a group for this round and store per-channel arrays
                        rgrp = h5f.create_group(f'round_{round_idx}_raw')
                        for idx, buf in enumerate(round_raw):
                            rgrp.create_dataset(f'chan_{idx}', data=buf)

                return filepath

            else:
                # fallback: call blocking acquire() and save result once
                try:
                    data = prog.acquire(self.config.soc, progress=True)
                except Exception as e:
                    raise

                # data is usually a list of measurement arrays per channel
                # reshape and store with proper dimensions
                for measIdx, arr in enumerate(data[0] if isinstance(data, tuple) else data):
                    arr = np.asarray(arr)

                    # Find spec name
                    complex_specs = [s for s in self.specs if isinstance(s, ComplexQICKData)]
                    if measIdx < len(complex_specs):
                        ds_name = complex_specs[measIdx].name
                    else:
                        ds_name = f'meas_{measIdx}'

                    # Store full-resolution data
                    h5f.create_dataset(ds_name, data=arr)

                return filepath

    def _save_static_specs(self, h5f: h5py.File, prog, cfg: Dict[str, Any]):
        """Save static (non-streaming) specs as HDF5 datasets/attributes."""
        for ds in self.specs:
            if ds.depends_on is None and not isinstance(ds, ComplexQICKData):
                # independent variable: save as dataset
                if isinstance(ds, PulseVariable):
                    try:
                        arr = prog.get_pulse_param(ds.pulse_parameter, ds.sweep_parameter, as_array=True)
                    except Exception:
                        arr = np.array([])
                elif isinstance(ds, TimeVariable):
                    try:
                        arr = prog.get_time_param(ds.time_parameter, 't', as_array=True) * (cfg.get('n_echoes', 0) + 1)
                    except Exception:
                        arr = np.array([])
                else:
                    # fallback: try to convert default value to array
                    arr = np.asarray(ds.default if hasattr(ds, 'default') else [])

                # create dataset
                name = ds.name
                if arr.size > 0:
                    h5f.create_dataset(name, data=arr)


__all__ = ["QickBoardStreamingSweep", "ComplexQICKData", "PulseVariable", "TimeVariable"]
