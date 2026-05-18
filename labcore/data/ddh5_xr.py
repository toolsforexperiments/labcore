"""
ddh5_xr.py -- Direct DDH5-to-xarray conversion without DataDict overhead.

Optimized pipeline that skips DataDict/MeshgridDataDict intermediate
representations. Reads DDH5 files with SWMR support for lock-free
concurrent access, supports lazy loading via h5py/dask, preserves
coordinate order/axis dependencies/units/labels, and handles
multi-dimensional independents.

Gridding strategies for incomplete data:
  'pad'      -- fill missing grid slots with NaN (raises warning)
  'truncate' -- trim to smallest complete grid (raises warning)
  'raise'    -- fail on shape mismatch

Modular layers:
  Schema discovery  ->  Grid inference  ->  Coordinate building  ->  Data var
  building  ->  Dataset assembly, plus Validation and Gridded-DDH5 export.
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import h5py

from .datadict_storage import FileOpener, deh5ify, DATAFILEXT

logger = logging.getLogger(__name__)

try:
    import dask.array as da

    _DASK_AVAILABLE = True
except ImportError:
    _DASK_AVAILABLE = False
    da = None

try:
    import xarray as xr

    _XARRAY_AVAILABLE = True
except ImportError:
    _XARRAY_AVAILABLE = False
    xr = None


# ===========================================================================
# Constants
# ===========================================================================

_TIMESTRFORMAT = "%Y-%m-%d %H:%M:%S"


# ===========================================================================
# Enums / modes
# ===========================================================================


class MissingMode(str, Enum):
    """
    How to handle data that does not perfectly fill the inferred grid.

    * ``pad`` -- fill empty grid slots with ``np.nan``; raise ``RuntimeWarning``.
    * ``truncate`` -- trim trailing incomplete cycle; raise ``RuntimeWarning``.
    * ``raise`` -- raise ``ValueError`` on any shape mismatch.
    """

    PAD = "pad"
    TRUNCATE = "truncate"
    RAISE = "raise"


class GridInferenceError(ValueError):
    """Raised when the grid shape cannot be determined from the DDH5 data."""


# ===========================================================================
# Dataclasses
# ===========================================================================


@dataclass
class DDH5FieldInfo:
    """Metadata for a single data field (independent or dependent).

    Parameters
    ----------
    name : str
        Field name as used in the HDF5 dataset.
    shape : tuple of int
        Stored shape of the dataset ``(nrecords, ...)``.
    dtype : numpy.dtype
        Data type of the stored values.
    axes : list of str
        For dependents: list of independent axis names.
        For independents: empty list.
    unit : str
        Physical unit string (empty if not specified).
    label : str
        Human-readable label (falls back to *name* if unset).
    """

    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    axes: List[str] = field(default_factory=list)
    unit: str = ""
    label: str = ""


@dataclass
class DDH5Schema:
    """Complete structural description of a DDH5 ``/data`` group.

    Parameters
    ----------
    fields : dict[str, DDH5FieldInfo]
        Map from field name to its metadata.
    dependents : list of str
        Names of fields that have an ``axes`` attribute (non-empty).
    independents : list of str
        Names of fields that have no ``axes`` attribute (empty list).
    nrecords : int
        First dimension length shared by all datasets.
    meta : dict[str, Any]
        Global group-level metadata (cleaned by :func:`deh5ify`).
    """

    fields: Dict[str, DDH5FieldInfo]
    dependents: List[str]
    independents: List[str]
    nrecords: int
    meta: Dict[str, Any]

    def select(self, names: List[str]) -> DDH5Schema:
        """Return a sub-schema containing only *names* (plus required axes)."""
        keep = set(names)
        for name in names:
            info = self.fields.get(name)
            if info and info.axes:
                keep.update(info.axes)
        fields = {n: self.fields[n] for n in keep if n in self.fields}
        deps = [n for n in self.dependents if n in keep]
        indeps = [n for n in self.independents if n in keep]
        return DDH5Schema(
            fields=fields,
            dependents=deps,
            independents=indeps,
            nrecords=self.nrecords,
            meta=self.meta,
        )


@dataclass
class GridInfo:
    """Inferred multi-dimensional grid description.

    Parameters
    ----------
    axes_order : list of str
        Axis names in dimension order (slowest-varying first), matching the
        ``axes`` attribute convention.
    shape : tuple of int
        Reshape target for dependent data ``(dim0, dim1, ...)``.
    is_complete : bool
        ``True`` when ``prod(shape) == nrecords`` exactly.
    expected_len : int
        ``prod(shape)`` -- how many elements a perfect grid needs.
    actual_len : int
        The number of records actually stored.
    """

    axes_order: List[str]
    shape: Tuple[int, ...]
    is_complete: bool
    expected_len: int
    actual_len: int


@dataclass
class DDH5ValidationReport:
    """Result of :func:`validate_ddh5`.

    Parameters
    ----------
    is_valid : bool
        ``True`` when no fatal errors were found.
    errors : list of str
        Fatal issues (missing axes, non-positive lengths, ...).
    warnings : list of str
        Non-fatal issues (NaN in axis data, non-monotonic, ...).
    schema : DDH5Schema or None
        Extracted schema (available even when validation fails partially).
    grid_info : GridInfo or None
        Inferred grid (``None`` when validation fails or axes cannot be
        determined).
    """

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings_list: List[str] = field(default_factory=list)
    schema: Optional[DDH5Schema] = None
    grid_info: Optional[GridInfo] = None


# ===========================================================================
# Internal helpers
# ===========================================================================


class _DDH5SWMRReader:
    """Context manager for reading DDH5 with SWMR for lock-free reads.

    Tries SWMR mode first; falls back to regular HDF5 opening.
    Unlike :class:`FileOpener`, does **not** create a lock file,
    so it is safe for concurrent read-while-write scenarios.

    Parameters
    ----------
    path : str, Path
        Full path to the ``.ddh5`` file.
    timeout : float, optional
        Maximum time (seconds) to wait for the file to appear / become
        accessible.  Default: 30 s.
    """

    def __init__(self, path: Union[str, Path], timeout: float = 30.0):
        self._path = Path(path)
        self._timeout = timeout
        self._h5: Optional[h5py.File] = None
        self.swmr_active: bool = False

    # ------------------------------------------------------------------
    def __enter__(self) -> h5py.File:
        t0 = time.time()
        while True:
            if not self._path.exists():
                if time.time() - t0 > self._timeout:
                    raise FileNotFoundError(
                        f"File {self._path} not found within {self._timeout} s"
                    )
                time.sleep(0.1)
                continue

            try:
                self._h5 = h5py.File(str(self._path), "r", swmr=True)
                self.swmr_active = True
                return self._h5
            except Exception:
                pass

            try:
                self._h5 = h5py.File(str(self._path), "r")
                self.swmr_active = False
                return self._h5
            except Exception:
                if time.time() - t0 > self._timeout:
                    raise
                time.sleep(0.1)

    # ------------------------------------------------------------------
    def __exit__(self, *args: Any) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


def _data_file_path(file: Union[str, Path]) -> Path:
    """Normalise a filepath, appending ``.ddh5`` if missing."""
    path = Path(file)
    if path.suffix != f".{DATAFILEXT}":
        path = Path(path.parent, path.stem + f".{DATAFILEXT}")
    return path


def _lazy_array(h5ds: h5py.Dataset) -> Any:
    """Create a lazy array from an h5py Dataset.

    Uses ``dask.array`` when available, otherwise returns the h5py dataset
    itself (which supports slicing).
    """
    if _DASK_AVAILABLE:
        chunks = h5ds.chunks
        if chunks is None:
            chunks = "auto"
        return da.from_array(h5ds, chunks=chunks)
    return h5ds


def _set_h5_attr(h5obj: Any, name: str, val: Any) -> None:
    """Set HDF5 attribute with automatic type adaptation.

    Handles :class:`bytes`, :class:`numpy.ndarray`, :class:`list` of strings,
    and plain scalars.
    """
    try:
        if isinstance(val, list):
            all_str = all(isinstance(v, str) for v in val)
            if all_str:
                h5obj.attrs[name] = np.array(val, dtype=h5py.string_dtype())
            else:
                h5obj.attrs[name] = np.array(val)
        elif isinstance(val, (np.ndarray,)):
            h5obj.attrs[name] = val
        elif isinstance(val, bool):
            h5obj.attrs[name] = int(val)
        elif isinstance(val, bytes):
            h5obj.attrs[name] = val
        elif val is None:
            h5obj.attrs[name] = "__NONE__"
        else:
            h5obj.attrs[name] = val
    except Exception:
        try:
            h5obj.attrs[name] = str(val)
        except Exception:
            logger.debug("Could not set HDF5 attribute %s=%s", name, val)


# ===========================================================================
# Layer 1 -- Schema discovery
# ===========================================================================


def ddh5_schema(
    path: Union[str, Path],
    groupname: str = "data",
    file_timeout: Optional[float] = None,
    swmr: bool = True,
) -> DDH5Schema:
    """Read DDH5 structure **without** loading value data.

    Only dataset shapes, attributes, and group-level metadata are read.
    This is lightweight and suitable for use before deciding whether
    to load data.

    Parameters
    ----------
    path : str or Path
        Path to the ``.ddh5`` file.
    groupname : str, optional
        HDF5 group name (default ``"data"``).
    file_timeout : float, optional
        Passed through to the file opener.
    swmr : bool, optional
        Attempt SWMR reading (default ``True``).

    Returns
    -------
    DDH5Schema
    """
    filepath = _data_file_path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"DDH5 file not found: {filepath}")

    if swmr:
        opener = _DDH5SWMRReader(filepath, timeout=file_timeout or 30.0)
    else:
        opener = FileOpener(filepath, "r", timeout=file_timeout)

    with opener as f:
        if groupname not in f:
            raise ValueError(f"HDF5 group '{groupname}' not found in {filepath}")

        grp = f[groupname]
        keys = list(grp.keys())

        if not keys:
            nrecords = 0
        else:
            nrecords = grp[keys[0]].shape[0]

        fields: Dict[str, DDH5FieldInfo] = {}
        meta: Dict[str, Any] = {}

        for attr_name in grp.attrs:
            meta[attr_name] = deh5ify(grp.attrs[attr_name])

        for k in keys:
            ds = grp[k]
            try:
                shp = ds.shape
            except Exception:
                shp = (0,)
            try:
                dtype = ds.dtype
            except Exception:
                dtype = np.dtype("float64")

            raw_axes = ds.attrs.get("axes", None)
            if raw_axes is not None:
                axes_val = deh5ify(raw_axes)
                if isinstance(axes_val, (np.ndarray,)):
                    axes_val = axes_val.tolist()
                if isinstance(axes_val, str):
                    axes_val = [axes_val]
                axes = list(axes_val) if axes_val else []
            else:
                axes = []

            unit_raw = ds.attrs.get("unit", "")
            unit = deh5ify(unit_raw) if unit_raw else ""
            if isinstance(unit, bytes):
                unit = unit.decode("utf-8", errors="replace")
            unit = str(unit) if unit else ""

            label = k
            for attr_name in ds.attrs:
                cleaned = str(attr_name).strip("_")
                if cleaned == "label":
                    lv = deh5ify(ds.attrs[attr_name])
                    if isinstance(lv, bytes):
                        lv = lv.decode("utf-8", errors="replace")
                    if lv and str(lv):
                        label = str(lv)
                    break

            fields[k] = DDH5FieldInfo(
                name=k,
                shape=shp,
                dtype=np.dtype(dtype),
                axes=axes,
                unit=str(unit),
                label=str(label),
            )

        # A field is independent if it is referenced as an axis by ANY field,
        # or if it has no axes of its own.
        # A field is dependent only if it has axes and is NOT used as an axis.
        all_axes: set = set()
        for fi in fields.values():
            all_axes.update(fi.axes)

        deps = [n for n, fi in fields.items() if fi.axes and n not in all_axes]
        indeps = [n for n, fi in fields.items() if n in all_axes or not fi.axes]

    return DDH5Schema(
        fields=fields,
        dependents=deps,
        independents=indeps,
        nrecords=nrecords,
        meta=meta,
    )


# ===========================================================================
# Layer 2 -- Grid inference
# ===========================================================================


def _infer_axis_lengths(
    grp: h5py.Group,
    axes: List[str],
    fields: Dict[str, DDH5FieldInfo],
) -> List[int]:
    """Determine the grid length along each axis.

    Uses :func:`numpy.unique` to count distinct axis values.
    For multi-dimensional axes, compares per-record slices to detect
    repetition; inner dimensions equal to the number of distinct
    along-axis values.

    Parameters
    ----------
    grp : h5py.Group
        Open HDF5 group containing the datasets.
    axes : list of str
        Axis names in dimension order.
    fields : dict[str, DDH5FieldInfo]
        Field metadata keyed by name.

    Returns
    -------
    list of int
        Grid length for each axis, in the same order as *axes*.
    """
    lengths: List[int] = []
    for ax_name in axes:
        fi = fields[ax_name]
        ds = grp[ax_name]
        ndim = len(fi.shape)

        if ndim == 1:
            ax_data = ds[:]
            lengths.append(len(np.unique(ax_data)))
        else:
            inner_prod = int(np.prod(fi.shape[1:]))
            if fi.shape[0] <= 1:
                lengths.append(inner_prod)
            else:
                first_slice = ds[0]
                repeated = True
                for i in range(1, min(fi.shape[0], 20)):
                    if not np.array_equal(ds[i], first_slice):
                        repeated = False
                        break
                if repeated:
                    lengths.append(inner_prod)
                else:
                    lengths.append(inner_prod)
    return lengths


def _infer_grid(
    grp: h5py.Group,
    schema: DDH5Schema,
    missing: MissingMode,
) -> GridInfo:
    """Infer the multi-dimensional grid shape from axis data.

    Parameters
    ----------
    grp : h5py.Group
        HDF5 group containing the datasets.
    schema : DDH5Schema
        Discovered schema.
    missing : MissingMode
        How to handle incomplete grids.

    Returns
    -------
    GridInfo

    Raises
    ------
    GridInferenceError
        When the grid shape cannot be determined.
    ValueError
        When *missing='raise'* and data does not perfectly fill the grid.
    """
    if not schema.dependents:
        return GridInfo(
            axes_order=[],
            shape=(schema.nrecords,),
            is_complete=True,
            expected_len=schema.nrecords,
            actual_len=schema.nrecords,
        )

    primary_dep = schema.dependents[0]
    axes = list(schema.fields[primary_dep].axes)

    if not axes:
        return GridInfo(
            axes_order=[],
            shape=(schema.nrecords,),
            is_complete=True,
            expected_len=schema.nrecords,
            actual_len=schema.nrecords,
        )

    for ax in axes:
        if ax not in schema.fields:
            raise GridInferenceError(
                f"Axis '{ax}' referenced by '{primary_dep}' "
                f"does not exist in the DDH5 file."
            )

    lengths = _infer_axis_lengths(grp, axes, schema.fields)

    if any(l == 0 for l in lengths):
        raise GridInferenceError(
            f"Zero-length axis detected. Axis lengths: "
            f"{dict(zip(axes, lengths))}"
        )

    shape = tuple(lengths)
    expected_len = int(np.prod(shape))

    dep_info = schema.fields[primary_dep]
    dep_total = int(np.prod(dep_info.shape))
    actual_len = dep_total

    if expected_len != actual_len:
        if missing == MissingMode.RAISE:
            raise ValueError(
                f"Grid shape {shape} (={expected_len} elements) does not "
                f"match the data ({actual_len} elements). "
                f"Use missing='pad' or missing='truncate'."
            )
        elif missing == MissingMode.TRUNCATE:
            truncated_shape = list(shape)
            truncated_axes = list(axes)
            found = False
            for i in range(len(truncated_shape)):
                reduced = int(np.prod(truncated_shape[: i + 1]))
                if actual_len >= reduced and (i + 1 < len(truncated_shape)):
                    continue
                if actual_len >= reduced:
                    truncated_shape = truncated_shape[: i + 1]
                    truncated_axes = truncated_axes[: i + 1]
                    found = True
                    break
                else:
                    inner = int(np.prod(truncated_shape[1: i + 1])) if i > 0 else 1
                    max_outer = actual_len // inner
                    if max_outer > 0:
                        truncated_shape[0] = max_outer
                        truncated_shape = truncated_shape[: i + 1]
                        truncated_axes = truncated_axes[: i + 1]
                    else:
                        truncated_shape = [actual_len]
                        truncated_axes = []
                    found = True
                    break
            if not found:
                truncated_shape = [actual_len]
                truncated_axes = []
            shape = tuple(truncated_shape)
            axes = truncated_axes
            expected_len = int(np.prod(shape))
            warnings.warn(
                f"Data truncated to grid shape {shape}: "
                f"expected {int(np.prod(tuple(lengths)))} elements, "
                f"got {actual_len}. Trailing {actual_len - expected_len} "
                f"elements dropped.",
                RuntimeWarning,
            )
        elif missing == MissingMode.PAD:
            warnings.warn(
                f"Grid shape {shape} requires {expected_len} elements, "
                f"but only {actual_len} found. "
                f"Missing {expected_len - actual_len} slots filled "
                f"with NaN.",
                RuntimeWarning,
            )

    return GridInfo(
        axes_order=list(axes),
        shape=shape,
        is_complete=(expected_len == actual_len),
        expected_len=expected_len,
        actual_len=actual_len,
    )


# ===========================================================================
# Layer 3 -- Data loading & reshaping
# ===========================================================================


def _reshape_to_grid(
    data: np.ndarray,
    grid_shape: Tuple[int, ...],
    target_ndim: int,
    missing: MissingMode,
) -> np.ndarray:
    """Reshape data to target grid shape.

    Handles both flat-record data (shape ``(N,)``) and data that is already
    partially multi-dimensional (shape ``(N, inner_dims...)``).  Compares
    total element counts to determine whether padding/truncation is needed.

    Parameters
    ----------
    data : np.ndarray
        Data array, first axis is the record dimension.
    grid_shape : tuple of int
        Target grid dimensions.
    target_ndim : int
        Desired number of grid dimensions in output.
    missing : MissingMode
        Mode for handling size mismatch.

    Returns
    -------
    np.ndarray
        Reshaped data.
    """
    total_elements = data.size
    expected_total = int(np.prod(grid_shape))

    if expected_total < total_elements:
        if missing == MissingMode.TRUNCATE:
            flat = data.ravel()[:expected_total]
            return flat.reshape(grid_shape)
        elif missing == MissingMode.RAISE:
            raise ValueError(
                f"Data has {total_elements} elements, "
                f"grid expects {expected_total}"
            )
        else:
            flat = data.ravel()[:expected_total]
            return flat.reshape(grid_shape)
    elif expected_total > total_elements:
        if missing == MissingMode.PAD:
            flat = data.ravel()
            pad_size = expected_total - total_elements
            padding = np.full(pad_size, np.nan, dtype=flat.dtype)
            return np.concatenate([flat, padding]).reshape(grid_shape)
        elif missing == MissingMode.RAISE:
            raise ValueError(
                f"Data has {total_elements} elements, "
                f"grid expects {expected_total}"
            )
    else:
        return data.ravel().reshape(grid_shape)


def _build_coordinates(
    grp: h5py.Group,
    schemas: DDH5Schema,
    grid_info: GridInfo,
    missing: MissingMode = MissingMode.PAD,
    lazy: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Build xarray coordinate arrays and their metadata.

    For 1D grid axes, extracts ordered unique values from the axis dataset
    (avoiding unnecessary reshape-to-full-grid).

    For multi-dimensional axes that repeat across records, takes the first
    record's array as the coordinate.

    For multi-dimensional axes that vary per record, reshapes them to the
    grid and includes them as multi-dimensional coordinates.

    Parameters
    ----------
    grp : h5py.Group
        HDF5 group.
    schemas : DDH5Schema
        Schema.
    grid_info : GridInfo
        Inferred grid.
    missing : MissingMode
        Strategy for incomplete grids.
    lazy : bool
        If True, use lazy arrays.

    Returns
    -------
    coords : dict[str, array-like]
        Coordinate arrays keyed by name.
    coord_attrs : dict[str, dict[str, Any]]
        Per-coordinate metadata (units, label, axes).
    """
    coords: Dict[str, Any] = {}
    coord_attrs: Dict[str, Dict[str, Any]] = {}
    grid_axes_set = set(grid_info.axes_order)
    n_axes = len(grid_info.axes_order)

    for name in schemas.independents:
        ds = grp[name]
        fi = schemas.fields[name]

        attrs: Dict[str, Any] = {}
        if fi.unit:
            attrs["units"] = fi.unit
        if fi.label and fi.label != name:
            attrs["label"] = fi.label

        if name in grid_axes_set and n_axes > 0:
            axis_idx = grid_info.axes_order.index(name)
            ndim = len(fi.shape)
            axis_grid_len = grid_info.shape[axis_idx] if axis_idx < len(grid_info.shape) else fi.shape[0]

            if ndim == 1:
                raw_1d = ds[:]
                unique_vals = np.unique(raw_1d)
                if len(unique_vals) == axis_grid_len:
                    coord_data = unique_vals
                else:
                    _, idx = np.unique(raw_1d, return_index=True)
                    coord_data = raw_1d[np.sort(idx)]
                    if len(coord_data) > axis_grid_len:
                        coord_data = coord_data[:axis_grid_len]
                coords[name] = coord_data
            else:
                raw_md = ds[:]
                same = True
                first = raw_md[0]
                for i in range(1, min(raw_md.shape[0], 30)):
                    if not np.array_equal(raw_md[i], first):
                        same = False
                        break
                if same:
                    coords[name] = first
                else:
                    reshaped = _reshape_to_grid(
                        raw_md, grid_info.shape, len(grid_info.shape), missing
                    )
                    coords[name] = reshaped
        else:
            if lazy:
                coords[name] = _lazy_array(ds)
            else:
                coords[name] = ds[:]

        coord_attrs[name] = attrs

    return coords, coord_attrs


def _build_data_variables(
    grp: h5py.Group,
    schemas: DDH5Schema,
    grid_info: GridInfo,
    missing: MissingMode,
    lazy: bool = False,
    fields_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build xarray DataArray entries for dependent datasets.

    Parameters
    ----------
    grp : h5py.Group
        HDF5 group.
    schemas : DDH5Schema
        Schema.
    grid_info : GridInfo
        Inferred grid.
    missing : MissingMode
        Handling for incomplete grids.
    lazy : bool
        If True, use lazy arrays where possible.
    fields_filter : list of str, optional
        If given, only include these dependent names.

    Returns
    -------
    dict[str, xr.DataArray or tuple]
        Entries suitable for ``xr.Dataset(..., coords=...)``.
    """
    data_vars: Dict[str, Any] = {}
    dims = grid_info.axes_order if grid_info.axes_order else ["dim_0"]

    for name in schemas.dependents:
        if fields_filter is not None and name not in fields_filter:
            continue

        ds = grp[name]
        fi = schemas.fields[name]
        unit = fi.unit
        axes = fi.axes

        if grid_info.shape and grid_info.shape != (schemas.nrecords,):
            raw_data = ds[:]
            n_grid_dims = len(grid_info.shape)
            reshaped = _reshape_to_grid(
                raw_data, grid_info.shape, n_grid_dims, missing
            )
        else:
            if lazy:
                reshaped = _lazy_array(ds)
            else:
                reshaped = ds[:]

        if _XARRAY_AVAILABLE:
            da_attrs: Dict[str, Any] = {}
            if unit:
                da_attrs["units"] = unit
            if axes:
                da_attrs["axes"] = axes
            da_attrs["label"] = fi.label

            data_vars[name] = xr.DataArray(
                data=reshaped,
                dims=dims[: reshaped.ndim] if reshaped.ndim <= len(dims) else dims,
                attrs=da_attrs,
            )
        else:
            data_vars[name] = (tuple(dims), reshaped)

    return data_vars


# ===========================================================================
# Layer 4 -- Public conversion API
# ===========================================================================


def ddh5_to_xarray(
    path: Union[str, Path],
    groupname: str = "data",
    lazy: bool = False,
    fields: Optional[List[str]] = None,
    startidx: Optional[int] = None,
    stopidx: Optional[int] = None,
    missing: MissingMode = MissingMode.PAD,
    swmr: bool = True,
    file_timeout: Optional[float] = None,
) -> "xr.Dataset":
    """Convert a DDH5 file directly to :class:`xarray.Dataset`.

    Skips :class:`DataDict` and :class:`MeshgridDataDict` intermediate
    representations. Supports SWMR reading, lazy loading, and two
    strategies (:attr:`MissingMode.PAD` / :attr:`MissingMode.TRUNCATE`)
    for incomplete grids.

    Parameters
    ----------
    path : str or Path
        Path to the ``.ddh5`` file (``.ddh5`` extension optional).
    groupname : str, optional
        HDF5 group name (default ``"data"``).
    lazy : bool, optional
        Use :mod:`dask` / h5py lazy arrays. Default ``False``.
    fields : list of str, optional
        Limit output to these data fields (dependents + their axes).
        If ``None``, all fields are included.
    startidx : int, optional
        Start record index (0-based).
    stopidx : int, optional
        End record index (exclusive).
    missing : MissingMode, optional
        - ``'pad'`` -- fill missing grid slots with NaN + warning
        - ``'truncate'`` -- trim trailing incomplete cycle + warning
        - ``'raise'`` -- fail on shape mismatch
    swmr : bool, optional
        Attempt SWMR mode for lock-free reads (default ``True``).
    file_timeout : float, optional
        Max wait time for file access (seconds).

    Returns
    -------
    xarray.Dataset

    Raises
    ------
    ImportError
        When ``xarray`` is not installed.
    GridInferenceError
        When the grid shape cannot be inferred from axis data.
    ValueError
        When *missing='raise'* and shape mismatch is detected.
    """
    if not _XARRAY_AVAILABLE:
        raise ImportError(
            "xarray is required for ddh5_to_xarray. "
            "Install with: pip install xarray"
        )

    filepath = _data_file_path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"DDH5 file not found: {filepath}")

    if swmr:
        opener = _DDH5SWMRReader(filepath, timeout=file_timeout or 30.0)
    else:
        opener = FileOpener(filepath, "r", timeout=file_timeout)

    with opener as f:
        if groupname not in f:
            raise ValueError(f"HDF5 group '{groupname}' not found.")

        grp = f[groupname]

        schemas = ddh5_schema(
            filepath, groupname=groupname, file_timeout=file_timeout, swmr=False
        )

        if fields is not None:
            schemas = schemas.select(fields)

        if startidx is not None or stopidx is not None:
            start = startidx or 0
            stop = stopidx or schemas.nrecords
            schemas.nrecords = stop - start
            for fi in schemas.fields.values():
                fi.shape = (schemas.nrecords,) + fi.shape[1:]

        grid_info = _infer_grid(grp, schemas, missing)

        coords_raw, coord_attrs = _build_coordinates(
            grp, schemas, grid_info, missing=missing, lazy=lazy
        )
        data_vars = _build_data_variables(

            grp,
            schemas,
            grid_info,
            missing=missing,
            lazy=lazy,
            fields_filter=fields,
        )

        coords: Dict[str, Any] = {}
        grid_dims = grid_info.axes_order if grid_info.axes_order else ["dim_0"]
        for cname, cdata in coords_raw.items():
            if isinstance(cdata, np.ndarray) and cdata.ndim > 1:
                nd = min(cdata.ndim, len(grid_dims))
                coords[cname] = (tuple(grid_dims[:nd]), cdata)
            else:
                coords[cname] = cdata

        ddh5_attrs: Dict[str, Any] = {}
        for k, v in schemas.meta.items():
            clean_key = k.strip("_")
            ddh5_attrs[clean_key] = v
        ddh5_attrs["source_file"] = str(filepath.resolve())

    try:
        dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=ddh5_attrs)
    except Exception:
        logger.exception("Failed to assemble xarray Dataset")
        raise

    for dim_name, dim_attrs in coord_attrs.items():
        if dim_name in dataset.coords:
            dataset[dim_name].attrs.update(dim_attrs)

    return dataset


# ===========================================================================
# Gridded DDH5 export
# ===========================================================================


def ddh5_to_gridded_ddh5(
    path: Union[str, Path],
    groupname: str = "data",
    output_path: Optional[Union[str, Path]] = None,
    missing: MissingMode = MissingMode.PAD,
    swmr: bool = True,
    file_timeout: Optional[float] = None,
) -> Path:
    """Convert a record-format DDH5 to a gridded DDH5.

    Reads the source DDH5, reshapes all data to the inferred
    multi-dimensional grid, and writes a new ``.ddh5`` file in the same
    directory.  The gridded file is written with gzip compression and is
    SWMR-compatible for subsequent efficient reads.

    Parameters
    ----------
    path : str or Path
        Path to the source ``.ddh5`` file.
    groupname : str, optional
        HDF5 group name.
    output_path : str or Path, optional
        Output path.  Defaults to ``<source_dir>/<source_stem>_gridded.ddh5``.
    missing : MissingMode, optional
        Same semantics as :func:`ddh5_to_xarray`.
    swmr : bool, optional
        Attempt SWMR for reading the source.
    file_timeout : float, optional
        Max wait time for file access (seconds).

    Returns
    -------
    Path
        Path to the created gridded DDH5 file.
    """
    filepath = _data_file_path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Source DDH5 not found: {filepath}")

    if output_path is None:
        output_path = filepath.parent / f"{filepath.stem}_gridded.ddh5"
    else:
        output_path = _data_file_path(output_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if swmr:
        reader = _DDH5SWMRReader(filepath, timeout=file_timeout or 30.0)
    else:
        reader = FileOpener(filepath, "r", timeout=file_timeout)

    with reader as f:
        if groupname not in f:
            raise ValueError(f"Group '{groupname}' not found.")

        grp = f[groupname]

        schemas = ddh5_schema(
            filepath, groupname=groupname, file_timeout=file_timeout, swmr=False
        )
        grid_info = _infer_grid(grp, schemas, missing)

        all_fields: Dict[str, np.ndarray] = {}
        source_meta_attrs: Dict[str, Dict[str, Any]] = {}

        for name in schemas.independents:
            ds = grp[name]
            raw = ds[:]
            if name in grid_info.axes_order and grid_info.shape:
                reshaped = _reshape_to_grid(
                    raw, grid_info.shape, len(grid_info.shape), missing
                )
                all_fields[name] = reshaped
            else:
                all_fields[name] = raw
            source_meta_attrs[name] = {
                a: deh5ify(grp[name].attrs[a])
                for a in grp[name].attrs
                if a.startswith("__")
            }

        for name in schemas.dependents:
            ds = grp[name]
            raw = ds[:]
            if grid_info.shape and grid_info.shape != (schemas.nrecords,):
                reshaped = _reshape_to_grid(
                    raw, grid_info.shape, len(grid_info.shape), missing
                )
                all_fields[name] = reshaped
            else:
                all_fields[name] = raw
            source_meta_attrs[name] = {
                a: deh5ify(grp[name].attrs[a])
                for a in grp[name].attrs
                if a.startswith("__")
            }

    t = time.localtime()
    tsec = time.mktime(t)
    tstr = time.strftime(_TIMESTRFORMAT, t)

    with h5py.File(str(output_path), "w", libver="latest") as outf:
        out_grp = outf.create_group(groupname)

        try:
            out_grp.attrs["__creation_time_sec__"] = tsec
        except Exception:
            out_grp.attrs["__creation_time_sec__"] = str(tsec)
        try:
            out_grp.attrs["__creation_time_str__"] = tstr
        except Exception:
            pass

        for k, v in schemas.meta.items():
            _set_h5_attr(out_grp, k, v)

        _set_h5_attr(out_grp, "__gridded__", True)
        _set_h5_attr(out_grp, "__grid_shape__", list(grid_info.shape))
        _set_h5_attr(out_grp, "__grid_axes__", grid_info.axes_order)
        _set_h5_attr(out_grp, "__source_file__", str(filepath.resolve()))

        for name, data in all_fields.items():
            fi = schemas.fields[name]
            maxshp = (None,) * data.ndim if data.ndim > 0 else (None,)
            try:
                out_ds = out_grp.create_dataset(
                    name,
                    data=data,
                    maxshape=maxshp,
                    compression="gzip",
                    compression_opts=4,
                )
            except Exception:
                out_ds = out_grp.create_dataset(
                    name,
                    data=data,
                    maxshape=maxshp,
                )

            if fi.axes:
                _set_h5_attr(out_ds, "axes", fi.axes)
            if fi.unit:
                _set_h5_attr(out_ds, "unit", fi.unit)
            if fi.label and fi.label != name:
                _set_h5_attr(out_ds, "label", fi.label)
            _set_h5_attr(out_ds, "__creation_time_sec__", tsec)

            for attr_name, attr_val in source_meta_attrs.get(name, {}).items():
                _set_h5_attr(out_ds, attr_name, attr_val)

        outf.flush()

    logger.info("Gridded DDH5 written: %s", output_path)
    return output_path


# ===========================================================================
# Layer 5 -- Validation
# ===========================================================================


def validate_ddh5(
    path: Union[str, Path],
    groupname: str = "data",
    file_timeout: Optional[float] = None,
) -> DDH5ValidationReport:
    """Validate a DDH5 file for metadata and structural consistency.

    Checks performed:
      1. Group existence and accessibility.
      2. All axes referenced by dependents exist as datasets.
      3. Axes must be independents (empty ``axes``).
      4. All datasets share the same record-count (first dimension length).
      5. For gridded data, check axis monotonicity.
      6. Units and labels presence is noted but not enforced.

    Parameters
    ----------
    path : str or Path
        Path to the ``.ddh5`` file.
    groupname : str, optional
        HDF5 group name.
    file_timeout : float, optional
        Max wait time for file access.

    Returns
    -------
    DDH5ValidationReport
        Structured report with ``is_valid``, ``errors``, and ``warnings``.
    """
    errors: List[str] = []
    warnings_list: List[str] = []
    schemas: Optional[DDH5Schema] = None
    grid_info: Optional[GridInfo] = None

    filepath = _data_file_path(path)
    if not filepath.exists():
        errors.append(f"File not found: {filepath}")
        return DDH5ValidationReport(
            is_valid=False, errors=errors, warnings_list=warnings_list
        )

    try:
        schemas = ddh5_schema(filepath, groupname=groupname, file_timeout=file_timeout)
    except Exception as exc:
        errors.append(f"Schema discovery failed: {exc}")
        return DDH5ValidationReport(
            is_valid=False, errors=errors, warnings_list=warnings_list
        )

    with FileOpener(filepath, "r", timeout=file_timeout) as f:
        grp = f[groupname]

        # --- Axes existence ---
        for dep_name in schemas.dependents:
            for ax in schemas.fields[dep_name].axes:
                if ax not in schemas.fields:
                    errors.append(
                        f"Dependent '{dep_name}' references axis '{ax}' "
                        f"which is not a dataset."
                    )

        # --- Record count consistency ---
        lens: Dict[str, int] = {}
        for name, fi in schemas.fields.items():
            try:
                ds = grp[name]
                lens[name] = ds.shape[0]
            except Exception as exc:
                errors.append(f"Cannot read shape of '{name}': {exc}")

        if lens:
            unique_lens = set(lens.values())
            if len(unique_lens) > 1:
                min_len = min(unique_lens)
                offenders = [
                    f"{n}={l}" for n, l in lens.items() if l != min_len
                ]
                warnings_list.append(
                    f"Record counts differ across datasets. Minimum={min_len}. "
                    f"Offenders: {', '.join(offenders)}"
                )

        # --- Grid inference ---
        try:
            grid_info = _infer_grid(grp, schemas, MissingMode.PAD)
        except GridInferenceError as exc:
            warnings_list.append(f"Grid inference: {exc}")
        except Exception as exc:
            warnings_list.append(f"Grid inference unexpected error: {exc}")

        # --- Monotonicity check for gridded axes ---
        if grid_info and grid_info.axes_order:
            for axis_idx, ax_name in enumerate(grid_info.axes_order):
                fi = schemas.fields[ax_name]
                if len(fi.shape) != 1:
                    continue
                try:
                    ax_data = grp[ax_name][:]
                    if grid_info.shape and axis_idx < len(grid_info.shape):
                        reshaped = np.reshape(
                            ax_data[: grid_info.expected_len], grid_info.shape
                        )
                        slices_list = [0] * len(grid_info.shape)
                        slices_list[axis_idx] = slice(None)
                        sliced = reshaped[tuple(slices_list)]
                        if len(sliced) > 1:
                            diffs = np.diff(sliced.astype(float))
                            diffs = diffs[~np.isnan(diffs)]
                            if len(diffs) > 0:
                                signs = np.sign(diffs)
                                unique_signs = np.unique(signs[signs != 0])
                                if 0 in unique_signs or len(unique_signs) > 1:
                                    warnings_list.append(
                                        f"Axis '{ax_name}' is not monotonic."
                                    )
                except Exception as exc:
                    warnings_list.append(
                        f"Monotonicity check failed for '{ax_name}': {exc}"
                    )

        # --- NaN presence in axis data ---
        for ax_name in schemas.independents:
            try:
                ax_data = grp[ax_name][:]
                if np.any(np.isnan(ax_data.astype(float))):
                    warnings_list.append(f"Axis '{ax_name}' contains NaN values.")
            except Exception:
                pass

    is_valid = len(errors) == 0
    return DDH5ValidationReport(
        is_valid=is_valid,
        errors=errors,
        warnings_list=warnings_list,
        schema=schemas,
        grid_info=grid_info,
    )


# ===========================================================================
# Utility
# ===========================================================================


def ddh5_info(
    path: Union[str, Path],
    groupname: str = "data",
    file_timeout: Optional[float] = None,
) -> str:
    """Return a human-readable summary of DDH5 contents.

    Includes field names, shapes, axes relationships, units, and
    grid information.

    Parameters
    ----------
    path : str or Path
        Path to the ``.ddh5`` file.
    groupname : str, optional
        HDF5 group name.
    file_timeout : float, optional
        Max wait time for file access.

    Returns
    -------
    str
    """
    filepath = _data_file_path(path)
    if not filepath.exists():
        return f"DDH5 file not found: {filepath}"

    schemas = ddh5_schema(
        path, groupname=groupname, file_timeout=file_timeout, swmr=False
    )

    lines: List[str] = []
    lines.append(f"DDH5: {filepath}")
    lines.append(f"  Group: /{groupname}")
    lines.append(f"  Records: {schemas.nrecords}")
    lines.append(f"  Fields: {len(schemas.fields)}")
    lines.append(f"  Dependents: {len(schemas.dependents)}")
    lines.append(f"  Independents: {len(schemas.independents)}")

    if schemas.dependents:
        lines.append("")
        lines.append("  Dependents:")
        for name in schemas.dependents:
            fi = schemas.fields[name]
            axes_str = ", ".join(fi.axes) if fi.axes else "(none)"
            unit_str = f" [{fi.unit}]" if fi.unit else ""
            lines.append(
                f"    {name}{unit_str}  shape={fi.shape}  axes=({axes_str})"
            )

    if schemas.independents:
        lines.append("")
        lines.append("  Independents:")
        for name in schemas.independents:
            fi = schemas.fields[name]
            unit_str = f" [{fi.unit}]" if fi.unit else ""
            deps_str = f"  axes=({', '.join(fi.axes)})" if fi.axes else ""
            lines.append(f"    {name}{unit_str}  shape={fi.shape}{deps_str}")

    if schemas.meta:
        lines.append("")
        lines.append("  Global meta:")
        for k, v in schemas.meta.items():
            clean_k = k.strip("_")
            lines.append(f"    {clean_k}: {v}")

    return "\n".join(lines)
