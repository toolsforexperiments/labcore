"""Tests for labcore.data.datadict_storage"""
import numpy as np
import pytest
from pathlib import Path

from labcore.data.datadict import DataDict, datasets_are_equal
from labcore.data.datadict_storage import datadict_to_hdf5, datadict_from_hdf5, h5ify, deh5ify, AppendMode, DDH5Writer, load_as_xr, load_as_df, find_data, all_datadicts_from_hdf5, most_recent_data_path


@pytest.fixture
def simple_dd():
    dd = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    dd.add_data(x=np.array([1.0, 2.0, 3.0]), z=np.array([10.0, 20.0, 30.0]))
    return dd


def test_deh5ify_bytes_to_str():
    """deh5ify decodes a raw bytes object to a plain Python string."""
    assert deh5ify(b'hello') == 'hello'


def test_h5ify_list_of_strings_passes_through():
    """h5ify leaves a list-of-strings unchanged (all_string=True path)."""
    result = h5ify(['a', 'b', 'c'])
    assert result == ['a', 'b', 'c']


def test_h5ify_non_string_list_converted_to_array():
    """h5ify converts a list containing non-strings to a numpy array."""
    result = h5ify([1, 2, 3])
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1, 2, 3])


def test_h5ify_deh5ify_string_roundtrip():
    """h5ify encodes a numpy Unicode array to bytes; deh5ify decodes it back."""
    original = np.array(['qubit_A', 'qubit_B', 'qubit_C'])  # dtype kind 'U'
    encoded = h5ify(original)
    decoded = deh5ify(encoded)

    assert encoded.dtype.kind == 'S'  # byte strings
    np.testing.assert_array_equal(decoded, original)


def test_append_mode_all_duplicates_rows(tmp_path, simple_dd):
    """AppendMode.all appends all rows on every write, including duplicates."""
    path = tmp_path / 'data.ddh5'

    datadict_to_hdf5(simple_dd, path)
    datadict_to_hdf5(simple_dd, path, append_mode=AppendMode.all)

    dd_loaded = datadict_from_hdf5(path)
    assert dd_loaded.nrecords() == 6
    np.testing.assert_array_equal(dd_loaded.data_vals('x'), [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])


def test_ddh5writer_creates_file_and_appends(tmp_path):
    """DDH5Writer creates data.ddh5 on enter and add_data() appends rows."""
    dd = DataDict(x=dict(unit='m'), z=dict(axes=['x']))

    with DDH5Writer(dd, basedir=tmp_path, name='test') as writer:
        writer.add_data(x=1.0, z=10.0)
        writer.add_data(x=2.0, z=20.0)
        writer.add_data(x=3.0, z=30.0)
        filepath = writer.filepath

    assert filepath.exists()
    assert filepath.name == 'data.ddh5'

    dd_loaded = datadict_from_hdf5(filepath)
    assert dd_loaded.nrecords() == 3
    np.testing.assert_array_equal(dd_loaded.data_vals('x'), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(dd_loaded.data_vals('z'), [10.0, 20.0, 30.0])


def test_ddh5writer_data_persisted_incrementally(tmp_path):
    """Each add_data() call is immediately readable from disk with correct values."""
    dd = DataDict(x=dict(unit='m'), z=dict(axes=['x']))

    with DDH5Writer(dd, basedir=tmp_path, name='test') as writer:
        writer.add_data(x=1.0, z=10.0)
        snap1 = datadict_from_hdf5(writer.filepath)
        assert snap1.nrecords() == 1
        np.testing.assert_array_equal(snap1.data_vals('x'), [1.0])
        np.testing.assert_array_equal(snap1.data_vals('z'), [10.0])

        writer.add_data(x=2.0, z=20.0)
        snap2 = datadict_from_hdf5(writer.filepath)
        assert snap2.nrecords() == 2
        np.testing.assert_array_equal(snap2.data_vals('x'), [1.0, 2.0])
        np.testing.assert_array_equal(snap2.data_vals('z'), [10.0, 20.0])

        writer.add_data(x=3.0, z=30.0)
        
    snap3 = datadict_from_hdf5(writer.filepath)
    assert snap3.nrecords() == 3
    np.testing.assert_array_equal(snap3.data_vals('x'), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(snap3.data_vals('z'), [10.0, 20.0, 30.0])


def test_ddh5writer_file_structure(tmp_path):
    """DDH5Writer creates <basedir>/YYYY-MM-DD/<timestamp>-<name>/data.ddh5 structure.
    Three writers with the same name each get a separate subdirectory with a correct timestamp."""
    import re
    import datetime
    dd = DataDict(x=dict(unit='m'), z=dict(axes=['x']))

    paths = []
    for i in range(1, 4):
        before = datetime.datetime.now()
        with DDH5Writer(dd, basedir=tmp_path, name='myexp') as w:
            w.add_data(x=float(i), z=float(i * 10))
            paths.append(w.filepath)
        after = datetime.datetime.now()

    for path in paths:
        # file is always named data.ddh5
        assert path.name == 'data.ddh5'

        # experiment name in run directory
        assert 'myexp' in path.parent.name

        # date folder matches YYYY-MM-DD
        date_folder = path.parent.parent
        assert re.match(r'\d{4}-\d{2}-\d{2}', date_folder.name)
        assert date_folder.parent == tmp_path

        # timestamp in run dir name matches today's date
        # format: 2026-03-02T141356_<uuid>-myexp
        run_dir = path.parent.name
        ts_str = run_dir.split('_')[0]  # e.g. 2026-03-02T141356
        ts = datetime.datetime.strptime(ts_str, '%Y-%m-%dT%H%M%S')
        assert before.date() == ts.date()

    # all three run directories are distinct
    assert len({p.parent for p in paths}) == 3


def test_load_as_xr(tmp_path):
    """load_as_xr() loads a 2D grid sweep from data.ddh5 as an xarray Dataset."""
    x_vals = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    y_vals = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    z_vals = x_vals * 10 + y_vals

    dd = DataDict(x=dict(unit='m'), y=dict(unit='s'), z=dict(axes=['x', 'y']))
    dd.add_data(x=x_vals, y=y_vals, z=z_vals)

    with DDH5Writer(dd, basedir=tmp_path, name='test') as writer:
        folder = writer.filepath.parent

    ds = load_as_xr(folder)

    assert 'z' in ds
    assert ds['z'].shape == (2, 3)
    np.testing.assert_array_equal(ds.coords['x'].values, [0.0, 1.0])
    np.testing.assert_array_equal(ds.coords['y'].values, [0.0, 1.0, 2.0])


def test_load_as_df(tmp_path):
    """load_as_df() loads a sweep from data.ddh5 as a pandas DataFrame."""
    dd = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    dd.add_data(x=np.array([1.0, 2.0, 3.0]), z=np.array([10.0, 20.0, 30.0]))

    with DDH5Writer(dd, basedir=tmp_path, name='test') as writer:
        folder = writer.filepath.parent

    df = load_as_df(folder)

    assert 'z' in df.columns
    assert df.index.names == ['x']
    np.testing.assert_array_equal(df['z'].values, [10.0, 20.0, 30.0])


def test_find_data_locates_all_runs(tmp_path):
    """find_data() returns all folders containing data.ddh5 under the root."""
    dd = DataDict(x=dict(unit='m'), z=dict(axes=['x']))

    folders = []
    for i in range(1, 4):
        with DDH5Writer(dd, basedir=tmp_path, name=f'run{i}') as w:
            w.add_data(x=float(i), z=float(i * 10))
            folders.append(w.filepath.parent)

    found = find_data(tmp_path)

    assert len(found) == 3
    assert set(found.keys()) == set(folders)


def test_datadict_from_hdf5_slicing(tmp_path, simple_dd):
    """datadict_from_hdf5 startidx/stopidx returns only the requested rows."""
    path = tmp_path / 'data.ddh5'
    datadict_to_hdf5(simple_dd, path)

    sliced = datadict_from_hdf5(path, startidx=1, stopidx=3)

    assert sliced.nrecords() == 2
    np.testing.assert_array_equal(sliced.data_vals('x'), [2.0, 3.0])
    np.testing.assert_array_equal(sliced.data_vals('z'), [20.0, 30.0])


def test_find_data_folder_filter(tmp_path):
    """find_data() with folder_filter returns only folders whose name matches the pattern."""
    dd = DataDict(x=dict(unit='m'), z=dict(axes=['x']))

    with DDH5Writer(dd, basedir=tmp_path, name='cavity') as w:
        w.add_data(x=1.0, z=10.0)
        cavity_folder = w.filepath.parent

    with DDH5Writer(dd, basedir=tmp_path, name='qubit') as w:
        w.add_data(x=2.0, z=20.0)

    found = find_data(tmp_path, folder_filter='cavity')

    assert len(found) == 1
    assert cavity_folder in found


def test_append_mode_none_overwrites(tmp_path, simple_dd):
    """AppendMode.none deletes existing data and writes fresh on each call."""
    path = tmp_path / 'data.ddh5'

    datadict_to_hdf5(simple_dd, path)

    dd_new = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    dd_new.add_data(x=np.array([9.0, 8.0]), z=np.array([90.0, 80.0]))
    datadict_to_hdf5(dd_new, path, append_mode=AppendMode.none)

    dd_loaded = datadict_from_hdf5(path)
    assert dd_loaded.nrecords() == 2
    np.testing.assert_array_equal(dd_loaded.data_vals('x'), [9.0, 8.0])
    np.testing.assert_array_equal(dd_loaded.data_vals('z'), [90.0, 80.0])


def test_datadict_from_hdf5_structure_only(tmp_path, simple_dd):
    """datadict_from_hdf5 with structure_only=True returns axes/dependents but no values."""
    path = tmp_path / 'data.ddh5'
    datadict_to_hdf5(simple_dd, path)

    struct = datadict_from_hdf5(path, structure_only=True)

    assert set(struct.axes()) == {'x'}
    assert set(struct.dependents()) == {'z'}
    assert struct.nrecords() == 0


def test_hdf5_roundtrip(tmp_path, simple_dd):
    """datadict_to_hdf5 + datadict_from_hdf5 preserves structure and values."""
    path = tmp_path / 'data.ddh5'

    datadict_to_hdf5(simple_dd, path)
    dd_loaded = datadict_from_hdf5(path)

    assert datasets_are_equal(simple_dd, dd_loaded, ignore_meta=True)
    np.testing.assert_array_equal(dd_loaded.data_vals('x'), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(dd_loaded.data_vals('z'), [10.0, 20.0, 30.0])


def test_all_datadicts_from_hdf5(tmp_path):
    """all_datadicts_from_hdf5() loads every group from a single HDF5 file."""
    path = tmp_path / 'multi.ddh5'

    dd1 = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    dd1.add_data(x=np.array([1.0, 2.0]), z=np.array([10.0, 20.0]))

    dd2 = DataDict(a=dict(unit='s'), b=dict(axes=['a']))
    dd2.add_data(a=np.array([3.0, 4.0]), b=np.array([30.0, 40.0]))

    datadict_to_hdf5(dd1, path, groupname='group1')
    datadict_to_hdf5(dd2, path, groupname='group2', append_mode=AppendMode.none)

    result = all_datadicts_from_hdf5(path)

    assert set(result.keys()) == {'group1', 'group2'}
    np.testing.assert_array_equal(result['group1'].data_vals('x'), [1.0, 2.0])
    np.testing.assert_array_equal(result['group2'].data_vals('a'), [3.0, 4.0])


def test_ddh5writer_add_tag(tmp_path):
    """add_tag() creates a <tagname>.tag file in the run directory."""
    dd = DataDict(x=dict(unit='m'), z=dict(axes=['x']))

    with DDH5Writer(dd, basedir=tmp_path, name='test') as writer:
        writer.add_data(x=1.0, z=10.0)
        writer.add_tag('mytag')
        run_dir = writer.filepath.parent

    assert (run_dir / 'mytag.tag').exists()


def test_load_as_xr_with_fields(tmp_path):
    """load_as_xr() with fields= returns only the requested dependents."""
    x_vals = np.array([0.0, 0.0, 1.0, 1.0])
    y_vals = np.array([0.0, 1.0, 0.0, 1.0])

    dd = DataDict(x=dict(unit='m'), y=dict(unit='s'),
                  z=dict(axes=['x', 'y']), w=dict(axes=['x', 'y']))
    dd.add_data(x=x_vals, y=y_vals, z=x_vals + y_vals, w=x_vals * y_vals)

    with DDH5Writer(dd, basedir=tmp_path, name='test') as writer:
        folder = writer.filepath.parent

    ds = load_as_xr(folder, fields=['z'])

    assert 'z' in ds
    assert 'w' not in ds


def test_most_recent_data_path(tmp_path):
    """most_recent_data_path() returns the folder with the latest timestamp."""
    dd = DataDict(x=dict(unit='m'), z=dict(axes=['x']))

    paths = []
    for i in range(1, 4):
        with DDH5Writer(dd, basedir=tmp_path, name=f'run{i}') as w:
            w.add_data(x=float(i), z=float(i * 10))
            paths.append(w.filepath.parent)

    result = most_recent_data_path(tmp_path)
    assert result == sorted(paths)[-1]


def test_find_data_newer_than_excludes_old_runs(tmp_path):
    """find_data() with newer_than=now returns nothing because all runs predate the cutoff."""
    import datetime
    dd = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    for i in range(1, 3):
        with DDH5Writer(dd, basedir=tmp_path, name=f'run{i}') as w:
            w.add_data(x=float(i), z=float(i * 10))

    cutoff = datetime.datetime.now()
    found = find_data(tmp_path, newer_than=cutoff)

    assert len(found) == 0


def test_find_data_older_than_excludes_future_runs(tmp_path):
    """find_data() with older_than far in the past returns nothing because all runs postdate it."""
    import datetime
    dd = DataDict(x=dict(unit='m'), z=dict(axes=['x']))
    for i in range(1, 3):
        with DDH5Writer(dd, basedir=tmp_path, name=f'run{i}') as w:
            w.add_data(x=float(i), z=float(i * 10))

    cutoff = datetime.datetime(2000, 1, 1)
    found = find_data(tmp_path, older_than=cutoff)

    assert len(found) == 0