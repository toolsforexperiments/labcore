from pathlib import Path
from datetime import datetime

from labcore.data.datadict import DataDict, datasets_are_equal
from labcore.data.datadict_storage import DDH5Writer, datadict_to_hdf5, datadict_from_hdf5


def test_file_creation(tmp_path, n_files=500):
    datadict = DataDict(x=dict(unit='m'), y=dict(unit='m'), z=dict(axes=['x', 'y']))

    with DDH5Writer(datadict, str(tmp_path), safe_write_mode=True) as writer:
        now = datetime.now()
            
        for i in range(n_files):
            writer.add_data(x=i, y=i**2, z=i**3)

        today_date = now.strftime("%Y-%m-%d")
        current_hour = now.strftime("%H")
        current_minute = now.strftime("%M")

        data_tmp_path = tmp_path/writer.filepath.parent/".tmp"
        today_path = data_tmp_path/today_date
        hour_path = today_path/current_hour
        minute_path = hour_path/current_minute

        assert data_tmp_path.exists()
        assert today_path.exists()
        assert hour_path.exists()
        assert minute_path.exists()

        total_files = [file for file in data_tmp_path.rglob('*') if file.is_file()]
        assert len(total_files) == n_files
        

def test_number_of_files_per_folder(tmp_path):

    def check_file_limit(root_path, max_files):
        """
        Checks that the number of files does not exceed the passed max_files limit.
        """
        root = Path(root_path)
        for dirpath in root.rglob('*'):
            if dirpath.is_dir():
                items = list(dirpath.glob('*'))
                file_count = sum(1 for item in items if item.is_file())
                dir_count = sum(1 for item in items if item.is_dir())

                if file_count > 0 and dir_count > 0:
                    # Directory has files and subdirectories together
                    return False
                if file_count > max_files:
                    # Directory has more files than the limit
                    return False
        return True

    datadict = DataDict(x=dict(unit='m'), y=dict(unit='m'), z=dict(axes=['x', 'y']))

    default_n_files = DDH5Writer.n_files_per_dir

    n_files = 5000
    DDH5Writer.n_files_per_dir = 34

    with DDH5Writer(datadict, str(tmp_path), safe_write_mode=True) as writer:
        for i in range(n_files):
            writer.add_data(x=i, y=i ** 2, z=i ** 3)

        data_tmp_path = tmp_path / writer.filepath.parent / ".tmp"

        assert data_tmp_path.exists()

    assert check_file_limit(data_tmp_path, 34)

    # Testing again with a different set of numbers
    n_files = 4206
    DDH5Writer.n_files_per_dir = 12

    with DDH5Writer(datadict, str(tmp_path), safe_write_mode=True) as writer:
        for i in range(n_files):
            writer.add_data(x=i, y=i ** 2, z=i ** 3)

        data_tmp_path = tmp_path / writer.filepath.parent / ".tmp"

        assert data_tmp_path.exists()

    assert check_file_limit(data_tmp_path, 12)

    DDH5Writer.n_files_per_dir = default_n_files


def test_basic_unification(tmp_path, n_files=500):

    datadict = DataDict(x=dict(unit='m'), y=dict(unit='m'), z=dict(axes=['x', 'y']))

    with DDH5Writer(datadict, str(tmp_path), safe_write_mode=True) as writer:
        for i in range(n_files):
            writer.add_data(x=i, y=i**2, z=i**3)

        data_path = writer.filepath

    datadict_correct = DataDict(x=dict(unit='m'), y=dict(unit='m'), z=dict(axes=['x', 'y']))

    x = []
    y = []
    z = []
    for i in range(n_files):
        x.append(i)
        y.append(i ** 2)
        z.append(i ** 3)

    datadict_correct.add_data(x=x, y=y, z=z)

    correct_path = data_path.parent/"correct.ddh5"
    datadict_to_hdf5(datadict_correct, correct_path)

    created_data = datadict_from_hdf5(data_path)
    correct_data = datadict_from_hdf5(correct_path)

    assert datasets_are_equal(created_data, correct_data, ignore_meta=True)

