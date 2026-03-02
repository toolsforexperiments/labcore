"""Tests for labcore.data.datadict"""
import numpy as np
import pytest

from labcore.data.datadict import DataDict, dd2df, dd2xr, datadict_to_meshgrid, combine_datadicts, datasets_are_equal


def test_datadict_creation_and_structure():
    """DataDict correctly identifies axes vs dependents after validate()."""
    dd = DataDict(
        x=dict(unit='m'),
        z=dict(axes=['x'], unit='V'),
    )
    dd.validate()

    assert set(dd.axes()) == {'x'}
    assert set(dd.dependents()) == {'z'}
    # validate() fills in missing label
    assert dd['x'].get('label') == ''
    assert dd['z'].get('label') == ''


def test_add_data_individual_records():
    """add_data() called once per record appends rows and nrecords() tracks the count."""
    dd = DataDict(
        x=dict(unit='m'),
        z=dict(axes=['x']),
    )
    dd.add_data(x=1.0, z=10.0)
    dd.add_data(x=2.0, z=20.0)
    dd.add_data(x=3.0, z=30.0)

    assert dd.nrecords() == 3
    np.testing.assert_array_equal(dd.data_vals('x'), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(dd.data_vals('z'), [10.0, 20.0, 30.0])


def test_add_data_1d_arrays():
    """add_data() accepts 1D arrays, adding all records at once."""
    dd = DataDict(
        x=dict(unit='m'),
        z=dict(axes=['x']),
    )
    dd.add_data(x=np.array([1.0, 2.0, 3.0]), z=np.array([10.0, 20.0, 30.0]))

    assert dd.nrecords() == 3
    np.testing.assert_array_equal(dd.data_vals('x'), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(dd.data_vals('z'), [10.0, 20.0, 30.0])


def test_add_data_multidimensional_arrays():
    """add_data() accepts arrays where each record is itself an array (nested shape)."""
    dd = DataDict(
        x=dict(unit='m'),
        z=dict(axes=['x']),
    )
    # 3 records, each z value is a length-2 array
    dd.add_data(
        x=np.array([1.0, 2.0, 3.0]),
        z=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )

    assert dd.nrecords() == 3
    assert dd.data_vals('z').shape == (3, 2)
    np.testing.assert_array_equal(dd.data_vals('z')[1], [3.0, 4.0])


def test_dd2df_axes_as_multiindex():
    """dd2df() returns a DataFrame with axes as MultiIndex and dependents as columns."""
    dd = DataDict(
        x=dict(unit='m'),
        y=dict(unit='s'),
        z=dict(axes=['x', 'y']),
    )
    dd.add_data(x=np.array([1.0, 2.0, 3.0]), y=np.array([4.0, 5.0, 6.0]), z=np.array([7.0, 8.0, 9.0]))

    df = dd2df(dd)

    assert 'z' in df.columns
    assert df.index.names == ['x', 'y']
    np.testing.assert_array_equal(df['z'].values, [7.0, 8.0, 9.0])


def test_datadict_to_meshgrid_reshapes_flat_sweep():
    """datadict_to_meshgrid() infers grid shape from flat sweep data and reshapes values."""
    # flat sweep: x is slow axis (2 values), y is fast axis (3 values) — 6 rows total
    x_vals = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    y_vals = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    z_vals = x_vals * 10 + y_vals  # [0, 1, 2, 10, 11, 12]

    dd = DataDict(
        x=dict(unit='m', values=x_vals),
        y=dict(unit='s', values=y_vals),
        z=dict(axes=['x', 'y'], values=z_vals),
    )
    mgdd = datadict_to_meshgrid(dd)

    assert mgdd.shape() == (2, 3)
    np.testing.assert_array_equal(mgdd.data_vals('z'), [[0, 1, 2], [10, 11, 12]])


def test_dd2xr_produces_xarray_dataset():
    """dd2xr() converts a MeshgridDataDict to an xarray Dataset with correct coords and values."""
    # start from an already-gridded MeshgridDataDict
    x_vals = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    y_vals = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    z_vals = x_vals * 10 + y_vals

    dd = DataDict(
        x=dict(unit='m', values=x_vals),
        y=dict(unit='s', values=y_vals),
        z=dict(axes=['x', 'y'], values=z_vals),
    )
    mgdd = datadict_to_meshgrid(dd)
    ds = dd2xr(mgdd)

    assert 'z' in ds
    assert 'x' in ds.coords
    assert 'y' in ds.coords
    assert ds['z'].shape == (2, 3)
    np.testing.assert_array_equal(ds.coords['x'].values, [0.0, 1.0])
    np.testing.assert_array_equal(ds.coords['y'].values, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(ds['z'].values, [[0, 1, 2], [10, 11, 12]])


def test_combine_datadicts_merges_different_dependents():
    """combine_datadicts() merges DataDicts with different dependents sharing a common axis."""
    dd1 = DataDict(x=dict(unit='m'), z1=dict(axes=['x']))
    dd1.add_data(x=np.array([1.0, 2.0]), z1=np.array([10.0, 20.0]))

    dd2 = DataDict(x=dict(unit='m'), z2=dict(axes=['x']))
    dd2.add_data(x=np.array([1.0, 2.0]), z2=np.array([100.0, 200.0]))

    combined = combine_datadicts(dd1, dd2)

    # both dependents present under a single shared x axis
    assert set(combined.dependents()) == {'z1', 'z2'}
    assert set(combined.axes()) == {'x'}
    np.testing.assert_array_equal(combined.data_vals('z1'), [10.0, 20.0])
    np.testing.assert_array_equal(combined.data_vals('z2'), [100.0, 200.0])


def test_combine_datadicts_different_shapes_and_names():
    """combine_datadicts() with different record counts and variable names downgrades to DataDictBase."""
    dd1 = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    dd1.add_data(x=np.array([1.0, 2.0]), z=np.array([10.0, 20.0]))

    dd2 = DataDict(t=dict(unit='s'), v=dict(axes=['t']))
    dd2.add_data(t=np.array([0.1, 0.2, 0.3]), v=np.array([5.0, 6.0, 7.0]))

    from labcore.data.datadict import DataDictBase
    combined = combine_datadicts(dd1, dd2)

    assert isinstance(combined, DataDictBase)
    assert set(combined.dependents()) == {'z', 'v'}
    assert set(combined.axes()) == {'x', 't'}
    np.testing.assert_array_equal(combined.data_vals('x'), [1.0, 2.0])
    np.testing.assert_array_equal(combined.data_vals('z'), [10.0, 20.0])
    np.testing.assert_array_equal(combined.data_vals('t'), [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(combined.data_vals('v'), [5.0, 6.0, 7.0])


def test_datasets_are_equal():
    """datasets_are_equal() returns True for identical DataDicts and False when values or structure differ."""
    dd1 = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    dd1.add_data(x=np.array([1.0, 2.0]), z=np.array([10.0, 20.0]))

    # identical copy
    dd2 = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    dd2.add_data(x=np.array([1.0, 2.0]), z=np.array([10.0, 20.0]))

    assert datasets_are_equal(dd1, dd2)

    # different values
    dd3 = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    dd3.add_data(x=np.array([1.0, 2.0]), z=np.array([10.0, 99.0]))

    assert not datasets_are_equal(dd1, dd3)

    # different structure (extra field)
    dd4 = DataDict(x=dict(unit='m'), z=dict(axes=['x']), w=dict(axes=['x']))
    dd4.add_data(x=np.array([1.0, 2.0]), z=np.array([10.0, 20.0]), w=np.array([0.0, 0.0]))

    assert not datasets_are_equal(dd1, dd4)


def test_metadata_global_and_per_field():
    """add_meta() stores global and per-field metadata; meta_val() retrieves it."""
    dd = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    dd.add_data(x=np.array([1.0, 2.0]), z=np.array([10.0, 20.0]))

    dd.add_meta('sample', 'qubit_A')
    dd.add_meta('calibrated', True, data='z')

    assert dd.meta_val('sample') == 'qubit_A'
    assert dd.meta_val('calibrated', data='z') is True

    global_keys = [k for k, _ in dd.meta_items()]
    assert 'sample' in global_keys

    field_keys = [k for k, _ in dd.meta_items(data='z')]
    assert 'calibrated' in field_keys


def test_eq_identical_datadicts():
    """__eq__ returns True for DataDicts with identical structure and values."""
    dd1 = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    dd1.add_data(x=np.array([1.0, 2.0]), z=np.array([10.0, 20.0]))

    dd2 = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    dd2.add_data(x=np.array([1.0, 2.0]), z=np.array([10.0, 20.0]))

    assert dd1 == dd2


def test_eq_different_values():
    """__eq__ returns False when values differ, and False when compared to a non-DataDict."""
    dd1 = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    dd1.add_data(x=np.array([1.0, 2.0]), z=np.array([10.0, 20.0]))

    dd2 = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    dd2.add_data(x=np.array([1.0, 2.0]), z=np.array([99.0, 20.0]))

    assert not (dd1 == dd2)
    assert not (dd1 == "not a datadict")


def test_repr_contains_field_names_and_shape():
    """__repr__ includes dependent and axis names along with their shapes."""
    dd = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    dd.add_data(x=np.array([1.0, 2.0]), z=np.array([10.0, 20.0]))

    r = repr(dd)
    assert 'z' in r
    assert 'x' in r
    assert '(2,)' in r


def test_datadict_add_concatenates_records():
    """DataDict + DataDict returns a new DataDict with concatenated records."""
    dd1 = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    dd1.add_data(x=np.array([1.0, 2.0]), z=np.array([10.0, 20.0]))

    dd2 = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    dd2.add_data(x=np.array([3.0, 4.0]), z=np.array([30.0, 40.0]))

    combined = dd1 + dd2

    assert combined.nrecords() == 4
    np.testing.assert_array_equal(combined.data_vals('x'), [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(combined.data_vals('z'), [10.0, 20.0, 30.0, 40.0])


def test_datadict_append_concatenates_in_place():
    """DataDict.append() extends the DataDict in-place with records from another."""
    dd1 = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    dd1.add_data(x=np.array([1.0, 2.0]), z=np.array([10.0, 20.0]))

    dd2 = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    dd2.add_data(x=np.array([3.0, 4.0]), z=np.array([30.0, 40.0]))

    dd1.append(dd2)

    assert dd1.nrecords() == 4
    np.testing.assert_array_equal(dd1.data_vals('x'), [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(dd1.data_vals('z'), [10.0, 20.0, 30.0, 40.0])


def test_sanitize_removes_all_nan_rows():
    """sanitize() removes rows where all dependents are NaN, keeps partial rows."""
    dd = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    dd.add_data(
        x=np.array([1.0, 2.0, 3.0, 4.0]),
        z=np.array([10.0, np.nan, 30.0, np.nan]),
    )
    clean = dd.sanitize()

    # rows where z is NaN (rows 1 and 3) should be removed
    assert clean.nrecords() == 2
    np.testing.assert_array_equal(clean.data_vals('x'), [1.0, 3.0])
    np.testing.assert_array_equal(clean.data_vals('z'), [10.0, 30.0])


def test_sanitize_keeps_rows_with_at_least_one_valid_dependent():
    """sanitize() only removes a row if ALL dependents are NaN in that row."""
    dd = DataDict(x=dict(unit='m'), z1=dict(axes=['x']), z2=dict(axes=['x']))
    dd.add_data(
        x=np.array([1.0, 2.0, 3.0]),
        z1=np.array([10.0, np.nan, 30.0]),
        z2=np.array([100.0, 200.0, np.nan]),
    )
    clean = dd.sanitize()

    # no row has ALL dependents NaN, so nothing is removed
    assert clean.nrecords() == 3


def test_extract_pulls_dependent_with_its_axes():
    """extract() returns a new DataDict with only the requested dependent and its axes."""
    dd = DataDict(x=dict(unit='m'), y=dict(unit='s'), z1=dict(axes=['x']), z2=dict(axes=['y']))
    dd.add_data(
        x=np.array([1.0, 2.0]),
        y=np.array([3.0, 4.0]),
        z1=np.array([10.0, 20.0]),
        z2=np.array([30.0, 40.0]),
    )

    extracted = dd.extract(['z1'])

    assert set(extracted.dependents()) == {'z1'}
    assert set(extracted.axes()) == {'x'}
    assert 'z2' not in extracted
    assert 'y' not in extracted
    np.testing.assert_array_equal(extracted.data_vals('z1'), [10.0, 20.0])


def test_meshgrid_validate_mismatched_shapes_raises():
    """MeshgridDataDict.validate() raises ValueError when dependent shapes don't match."""
    from labcore.data.datadict import MeshgridDataDict
    dd = MeshgridDataDict(
        x=dict(values=np.array([[0.0, 1.0], [0.0, 1.0]])),
        y=dict(values=np.array([[0.0, 0.0], [1.0, 1.0]])),
        z1=dict(axes=['x', 'y'], values=np.array([[1.0, 2.0], [3.0, 4.0]])),
        z2=dict(axes=['x', 'y'], values=np.array([[1.0, 2.0, 3.0]])),  # wrong shape
    )
    with pytest.raises(ValueError):
        dd.validate()


def test_meshgrid_validate_non_monotonic_axis_raises():
    """MeshgridDataDict.validate() raises ValueError when an axis is not monotonic."""
    from labcore.data.datadict import MeshgridDataDict
    # x is not monotonic along axis 0 (goes 0, 0, 1, 1 but then back)
    dd = MeshgridDataDict(
        x=dict(values=np.array([[0.0, 0.0], [0.0, 0.0]])),  # no variation along axis 0
        y=dict(values=np.array([[0.0, 1.0], [0.0, 1.0]])),
        z=dict(axes=['x', 'y'], values=np.array([[1.0, 2.0], [3.0, 4.0]])),
    )
    with pytest.raises(ValueError):
        dd.validate()


def test_expand_flattens_nested_records():
    """expand() flattens nested (per-record) arrays into a 1D sequence."""
    dd = DataDict(
        x=dict(unit='m'),
        z=dict(axes=['x']),
    )
    # 3 records, each z value is a length-4 array
    dd.add_data(
        x=np.array([1.0, 2.0, 3.0]),
        z=np.array([[1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0]]),
    )

    assert not dd.is_expanded()
    assert dd.is_expandable()

    expanded = dd.expand()

    assert expanded.is_expanded()
    assert expanded.nrecords() == 12
    np.testing.assert_array_equal(expanded.data_vals('z'), [1,2,3,4,5,6,7,8,9,10,11,12])
    # x is repeated 4 times per original record
    np.testing.assert_array_equal(expanded.data_vals('x'), [1,1,1,1,2,2,2,2,3,3,3,3])


def test_datasets_are_equal_different_types():
    """datasets_are_equal() returns False when comparing different DataDict types."""
    from labcore.data.datadict import MeshgridDataDict
    x_vals = np.array([0.0, 0.0, 1.0, 1.0])
    y_vals = np.array([0.0, 1.0, 0.0, 1.0])
    z_vals = np.array([1.0, 2.0, 3.0, 4.0])

    dd = DataDict(x=dict(values=x_vals), y=dict(values=y_vals), z=dict(axes=['x', 'y'], values=z_vals))
    mgdd = datadict_to_meshgrid(dd)

    assert not datasets_are_equal(dd, mgdd)


def test_datasets_are_equal_ignores_meta():
    """datasets_are_equal() with ignore_meta=True returns True even when metadata differs."""
    dd1 = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    dd1.add_data(x=np.array([1.0, 2.0]), z=np.array([10.0, 20.0]))
    dd1.add_meta('sample', 'qubit_A')

    dd2 = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    dd2.add_data(x=np.array([1.0, 2.0]), z=np.array([10.0, 20.0]))
    dd2.add_meta('sample', 'qubit_B')

    assert not datasets_are_equal(dd1, dd2)
    assert datasets_are_equal(dd1, dd2, ignore_meta=True)


def test_validate_mismatched_lengths_raises():
    """validate() raises ValueError when fields have different numbers of records."""
    dd = DataDict(
        x=dict(unit='m', values=np.array([1.0, 2.0, 3.0])),
        z=dict(axes=['x'], values=np.array([10.0, 20.0])),
    )
    with pytest.raises(ValueError):
        dd.validate()