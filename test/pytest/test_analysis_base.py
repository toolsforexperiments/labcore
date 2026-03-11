"""Tests for labcore.analysis.analysis_base"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from labcore.analysis.analysis_base import DatasetAnalysis, AnalysisExistsError


def test_init_savefolder_structure(tmp_path):
    """__init__ builds analysis and data savefolders with the correct structure.

    savefolders[0] = analysisfolder / name / datafolder.stem
    savefolders[1] = datafolder / name
    """
    datafolder = tmp_path / '2026-03-02T141356_abc-myexp'
    analysis = DatasetAnalysis(datafolder, name='myanalysis', analysisfolder=tmp_path / 'analysis')

    assert analysis.savefolders[0] == tmp_path / 'analysis' / 'myanalysis' / datafolder.stem
    assert analysis.savefolders[1] == datafolder / 'myanalysis'


def test_add_stores_entity_and_raises_on_duplicate(tmp_path):
    """add() stores an entity by name; raises ValueError on a duplicate key."""
    datafolder = tmp_path / '2026-03-02T141356_abc-myexp'
    datafolder.mkdir()

    with DatasetAnalysis(datafolder, name='myanalysis', analysisfolder=tmp_path / 'analysis') as a:
        a.add(my_value=42)
        assert a.entities['my_value'] == 42

        with pytest.raises(ValueError):
            a.add(my_value=99)


def test_load_analysis_data_roundtrip(tmp_path):
    """Dict saved via add() can be reloaded with load_analysis_data()."""
    datafolder = tmp_path / '2026-03-02T141356_abc-myexp'
    datafolder.mkdir()
    analysisfolder = tmp_path / 'analysis'

    with DatasetAnalysis(datafolder, name='myanalysis', analysisfolder=analysisfolder) as a:
        a.add(result={'freq': 5.0, 'amp': 1.2})

    loaded = a.load_analysis_data('result')
    assert loaded == {'freq': 5.0, 'amp': 1.2}


def test_has_analysis_data(tmp_path):
    """has_analysis_data() returns True after saving, False for unknown names."""
    datafolder = tmp_path / '2026-03-02T141356_abc-myexp'
    datafolder.mkdir()
    analysisfolder = tmp_path / 'analysis'

    with DatasetAnalysis(datafolder, name='myanalysis', analysisfolder=analysisfolder) as a:
        a.add(result={'x': 1})

    assert a.has_analysis_data('result') is True
    assert a.has_analysis_data('nonexistent') is False


def test_raise_on_earlier_analysis(tmp_path):
    """raise_on_earlier_analysis causes AnalysisExistsError on re-entry when all listed files exist."""
    datafolder = tmp_path / '2026-03-02T141356_abc-myexp'
    datafolder.mkdir()
    analysisfolder = tmp_path / 'analysis'

    # First run — saves a result file
    with DatasetAnalysis(datafolder, name='myanalysis', analysisfolder=analysisfolder) as a:
        a.add(result={'x': 1})

    # Second run — should raise because 'result' json already exists
    with pytest.raises(AnalysisExistsError):
        with DatasetAnalysis(
            datafolder, name='myanalysis', analysisfolder=analysisfolder,
            raise_on_earlier_analysis=[('result', ['json'])],
        ) as a:
            pass


def test_context_manager_saves_dict_to_disk(tmp_path):
    """On __exit__, save() writes dict entities as JSON files to both savefolders."""
    datafolder = tmp_path / '2026-03-02T141356_abc-myexp'
    datafolder.mkdir()
    analysisfolder = tmp_path / 'analysis'

    with DatasetAnalysis(datafolder, name='myanalysis', analysisfolder=analysisfolder) as a:
        a.add(result={'freq': 5.0, 'amp': 1.2})

    # Both savefolders should contain a JSON file for 'result'
    for folder in a.savefolders:
        json_files = list(folder.glob('*result*.json'))
        assert len(json_files) == 1


def test_save_numpy_scalar(tmp_path):
    """Numpy scalar and float entities are saved as JSON files."""
    datafolder = tmp_path / '2026-03-02T141356_abc-myexp'
    datafolder.mkdir()

    with DatasetAnalysis(datafolder, name='myanalysis', analysisfolder=tmp_path / 'analysis') as a:
        a.add(amplitude=np.float64(3.14))

    for folder in a.savefolders:
        assert len(list(folder.glob('*amplitude*.json'))) == 1


def test_save_string(tmp_path):
    """String entities are saved as .txt files."""
    datafolder = tmp_path / '2026-03-02T141356_abc-myexp'
    datafolder.mkdir()

    with DatasetAnalysis(datafolder, name='myanalysis', analysisfolder=tmp_path / 'analysis') as a:
        a.add(note='this is a note')

    for folder in a.savefolders:
        assert len(list(folder.glob('*note*.txt'))) == 1


def test_save_xr_dataset(tmp_path):
    """xr.Dataset entities are saved as .nc (netCDF) files."""
    datafolder = tmp_path / '2026-03-02T141356_abc-myexp'
    datafolder.mkdir()

    ds = xr.Dataset({'y': ('x', [1.0, 2.0, 3.0])}, coords={'x': [0, 1, 2]})
    with DatasetAnalysis(datafolder, name='myanalysis', analysisfolder=tmp_path / 'analysis') as a:
        a.add(data=ds)

    for folder in a.savefolders:
        assert len(list(folder.glob('*data*xrdataset*.nc'))) == 1


def test_save_dataframe(tmp_path):
    """pd.DataFrame entities are saved as .csv files."""
    datafolder = tmp_path / '2026-03-02T141356_abc-myexp'
    datafolder.mkdir()

    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    with DatasetAnalysis(datafolder, name='myanalysis', analysisfolder=tmp_path / 'analysis') as a:
        a.add(table=df)

    for folder in a.savefolders:
        assert len(list(folder.glob('*table*.csv'))) == 1


def test_save_xr_dataarray(tmp_path):
    """xr.DataArray entities are saved as .nc files."""
    datafolder = tmp_path / '2026-03-02T141356_abc-myexp'
    datafolder.mkdir()

    da = xr.DataArray([1.0, 2.0, 3.0], dims=['x'], coords={'x': [0, 1, 2]})
    with DatasetAnalysis(datafolder, name='myanalysis', analysisfolder=tmp_path / 'analysis') as a:
        a.add(trace=da)

    for folder in a.savefolders:
        assert len(list(folder.glob('*trace*xrdataarray*.nc'))) == 1


def test_init_string_datafolder(tmp_path):
    """datafolder passed as a string is converted to Path."""
    datafolder = str(tmp_path / '2026-03-02T141356_abc-myexp')
    a = DatasetAnalysis(datafolder, name='myanalysis', analysisfolder=tmp_path / 'analysis')
    assert isinstance(a.datafolder, Path)


def test_init_has_period_in_name(tmp_path):
    """has_period_in_name=True includes the suffix in savefolder[0]."""
    datafolder = tmp_path / 'myexp.h5'
    a = DatasetAnalysis(datafolder, name='myanalysis', analysisfolder=tmp_path / 'analysis',
                        has_period_in_name=True)
    assert a.savefolders[0] == tmp_path / 'analysis' / 'myanalysis' / 'myexp.h5'


def test_save_pickle_fallback(tmp_path):
    """Unsupported entity types are saved as pickle files."""
    datafolder = tmp_path / '2026-03-02T141356_abc-myexp'
    datafolder.mkdir()

    class CustomObj:
        value = 42

    with DatasetAnalysis(datafolder, name='myanalysis', analysisfolder=tmp_path / 'analysis') as a:
        a.add(obj=CustomObj())

    for folder in a.savefolders:
        assert len(list(folder.glob('*obj*.pickle'))) == 1


def test_add_figure(tmp_path):
    """add_figure() creates and stores a matplotlib Figure; raises on duplicate."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    datafolder = tmp_path / '2026-03-02T141356_abc-myexp'
    datafolder.mkdir()

    a = DatasetAnalysis(datafolder, name='myanalysis', analysisfolder=tmp_path / 'analysis')
    fig = a.add_figure('myfig')
    assert 'myfig' in a.entities
    assert fig is a.entities['myfig']

    with pytest.raises(ValueError):
        a.add_figure('myfig')

    plt.close('all')


def test_save_mpl_figure(tmp_path):
    """matplotlib Figure entities are saved as both .png and .pdf."""
    import matplotlib
    matplotlib.use('Agg')

    datafolder = tmp_path / '2026-03-02T141356_abc-myexp'
    datafolder.mkdir()

    with DatasetAnalysis(datafolder, name='myanalysis', analysisfolder=tmp_path / 'analysis') as a:
        a.add_figure('plot')

    for folder in a.savefolders:
        assert len(list(folder.glob('*plot*.png'))) == 1
        assert len(list(folder.glob('*plot*.pdf'))) == 1


def test_save_hv_plot(tmp_path):
    """holoviews Dimensioned entities are saved as .html files."""
    import holoviews as hv
    import numpy as np
    hv.extension('bokeh')

    datafolder = tmp_path / '2026-03-02T141356_abc-myexp'
    datafolder.mkdir()

    curve = hv.Curve(np.linspace(0, 1, 10))
    with DatasetAnalysis(datafolder, name='myanalysis', analysisfolder=tmp_path / 'analysis') as a:
        a.add(curve=curve)

    for folder in a.savefolders:
        assert len(list(folder.glob('*curve*hvplot*.html'))) == 1


def test_to_table_creates_and_appends(tmp_path):
    """to_table() creates a CSV on first call and appends a row on second call."""
    datafolder = tmp_path / '2026-03-02T141356_abc-myexp'
    analysisfolder = tmp_path / 'analysis'

    a = DatasetAnalysis(datafolder, name='myanalysis', analysisfolder=analysisfolder)
    a.to_table('results', {'freq': 5.0})

    csv_path = a.savefolders[0].parent / 'results.csv'
    assert csv_path.exists()
    df = pd.read_csv(csv_path, index_col=0)
    assert len(df) == 1
    assert df['freq'].iloc[0] == 5.0

    # Second call with different datafolder appends a new row
    datafolder2 = tmp_path / '2026-03-03T090000_abc-myexp2'
    a2 = DatasetAnalysis(datafolder2, name='myanalysis', analysisfolder=analysisfolder)
    a2.to_table('results', {'freq': 6.0})

    df2 = pd.read_csv(csv_path, index_col=0)
    assert len(df2) == 2


def test_analysis_name_groups_multiple_datasets(tmp_path):
    """Multiple datasets analyzed with the same name are grouped under analysisfolder/name/.

    analysisfolder/
      T1/
        2026-03-02T141356_abc-run1/   <- run1 results
        2026-03-02T141356_abc-run2/   <- run2 results
      power_rabi/
        2026-03-02T141356_abc-run3/   <- run3 results
        2026-03-02T141356_abc-run4/   <- run4 results
    """
    analysisfolder = tmp_path / 'analysis'

    runs = [
        ('2026-03-02T141356_abc-run1', 'T1'),
        ('2026-03-02T141357_abc-run2', 'T1'),
        ('2026-03-02T141358_abc-run3', 'power_rabi'),
        ('2026-03-02T141359_abc-run4', 'power_rabi'),
    ]

    for folder_name, analysis_name in runs:
        datafolder = tmp_path / folder_name
        datafolder.mkdir()
        with DatasetAnalysis(datafolder, name=analysis_name, analysisfolder=analysisfolder) as a:
            a.add(result={'value': 1.0})

    # Each analysis name produces its own subdirectory
    assert (analysisfolder / 'T1').is_dir()
    assert (analysisfolder / 'power_rabi').is_dir()

    # Each dataset gets its own subfolder inside the group
    t1_subfolders = list((analysisfolder / 'T1').iterdir())
    rabi_subfolders = list((analysisfolder / 'power_rabi').iterdir())
    assert len(t1_subfolders) == 2
    assert len(rabi_subfolders) == 2

    # Subfolder names match the datafolder stems
    t1_names = {f.name for f in t1_subfolders}
    assert t1_names == {'2026-03-02T141356_abc-run1', '2026-03-02T141357_abc-run2'}


def test_load_metadata_from_json(tmp_path):
    """load_metadata_from_json() reads a key from a JSON file in the datafolder."""
    datafolder = tmp_path / '2026-03-02T141356_abc-myexp'
    datafolder.mkdir()
    (datafolder / 'meta.json').write_text(json.dumps({'qubit_freq': 5.1, 'other': 99}))

    a = DatasetAnalysis(datafolder, name='myanalysis')
    assert a.load_metadata_from_json('meta.json', 'qubit_freq') == 5.1

    with pytest.raises(ValueError):
        a.load_metadata_from_json('meta.json', 'missing_key')


def test_load_saved_parameter(tmp_path):
    """load_saved_parameter() reads value from a QCoDeS-style parameters.json."""
    datafolder = tmp_path / '2026-03-02T141356_abc-myexp'
    datafolder.mkdir()
    params = {'parameter_manager.drive_freq': {'value': 4.8, 'unit': 'GHz'}}
    (datafolder / 'parameters.json').write_text(json.dumps(params))

    a = DatasetAnalysis(datafolder, name='myanalysis')
    assert a.load_saved_parameter('drive_freq') == 4.8

    with pytest.raises(ValueError):
        a.load_saved_parameter('nonexistent')