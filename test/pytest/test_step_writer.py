import os
import shutil
from pathlib import Path
from datetime import datetime

from labcore.data.datadict import DataDict, datasets_are_equal
from labcore.data.datadict_storage import DDH5Writer, datadict_to_hdf5, datadict_from_hdf5

# TODO: Add a test to see what would happen if the tmp folder gets removed mid way


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
                file_count = sum(1 for item in items if item.is_file() and item.name.endswith(".ddh5"))
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

    with DDH5Writer(datadict, str(tmp_path), name="first", safe_write_mode=True) as writer:
        for i in range(n_files):
            writer.add_data(x=i, y=i ** 2, z=i ** 3)

        data_tmp_path = tmp_path / writer.filepath.parent / ".tmp"

        assert data_tmp_path.exists()

    assert check_file_limit(data_tmp_path, 34)

    # Testing again with a different set of numbers
    n_files = 4206
    DDH5Writer.n_files_per_dir = 12

    datadict = DataDict(x=dict(unit='m'), y=dict(unit='m'), z=dict(axes=['x', 'y']))

    with DDH5Writer(datadict, str(tmp_path), name="second", safe_write_mode=True) as writer:
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


def test_live_unification(tmp_path):

    holding_path = tmp_path/"holding"
    holding_path.mkdir()

    datadict = DataDict(x=dict(unit='m'), y=dict(unit='m'), z=dict(axes=['x', 'y']))
    datadict_correct_mid = DataDict(x=dict(unit='m'), y=dict(unit='m'), z=dict(axes=['x', 'y']))

    x, y, z = [], [], []
    for i in range(500):
        x.append(i)
        y.append(i ** 2)
        z.append(i ** 3)
    datadict_correct_mid.add_data(x=x, y=y, z=z)

    default_n_files_per_reconstruction = DDH5Writer.n_files_per_reconstruction
    DDH5Writer.n_files_per_reconstruction = 100

    with DDH5Writer(datadict, str(tmp_path), safe_write_mode=True) as writer:
        for i in range(500):
            writer.add_data(x=i, y=i**2, z=i**3)

        data_path = writer.filepath

        mid_point_dd = datadict_from_hdf5(data_path)
        assert datasets_are_equal(mid_point_dd, datadict_correct_mid, ignore_meta=True)
        assert mid_point_dd.has_meta("last_reconstructed_file")

        datadict_correct_end = DataDict(x=dict(unit='m'), y=dict(unit='m'), z=dict(axes=['x', 'y']))
        for i in range(500, 1000):
            x.append(i)
            y.append(i ** 2)
            z.append(i ** 3)
        datadict_correct_end.add_data(x=x, y=y, z=z)

        # The .tmp inside of the data folder
        tmp = data_path.parent/".tmp"

        all_files = []
        for root, dirs, files in os.walk(tmp):
            for file in files:
                all_files.append(Path(root) / file)

        # If you move the last file, the whole system crashes
        all_files = sorted(all_files, key=lambda x: int(x.stem.split("#")[-1]))

        # Store where the files come
        file_index = {}

        for file in all_files[:len(all_files)-1]:
            file_index[file.name] = file
            shutil.move(file, holding_path)

        # End Generation
        for i in range(500, 1000):
            writer.add_data(x=i, y=i ** 2, z=i ** 3)

        end_point_dd = datadict_from_hdf5(data_path)
        assert datasets_are_equal(end_point_dd, datadict_correct_end, ignore_meta=True)
        assert end_point_dd.has_meta("last_reconstructed_file")

        for filename, original_path in file_index.items():
            shutil.move(holding_path/filename, original_path)

    DDH5Writer.n_files_per_reconstruction = default_n_files_per_reconstruction


def test_locking_main_file(tmp_path):

    datadict = DataDict(x=dict(unit='m'), y=dict(unit='m'), z=dict(axes=['x', 'y']))

    with DDH5Writer(datadict, str(tmp_path), safe_write_mode=True, file_timeout=5) as writer:
        for i in range(500):
            writer.add_data(x=i, y=i**2, z=i**3)

        data_path = writer.filepath

        assert data_path.exists()

        # Making the lock file
        lock_file = data_path.parent/f"~{data_path.stem}.lock"
        lock_file.touch()

        for i in range(500, 1000):
            writer.add_data(x=i, y=i ** 2, z=i ** 3)

        assert lock_file.exists()

        lock_file.unlink(missing_ok=False)


def test_deleting_files_when_done(tmp_path):

    correct_datadict = DataDict(x=dict(unit='m'), y=dict(unit='m'), z=dict(axes=['x', 'y']))
    for i in range(1000):
        correct_datadict.add_data(x=i, y=i ** 2, z=i ** 3)

    datadict = DataDict(x=dict(unit='m'), y=dict(unit='m'), z=dict(axes=['x', 'y']))

    with DDH5Writer(datadict, str(tmp_path), safe_write_mode=True, file_timeout=5) as writer:
        for i in range(1000):
            writer.add_data(x=i, y=i ** 2, z=i ** 3)

        data_path = writer.filepath
        tmp_path = data_path.parent / ".tmp"

        assert tmp_path.exists()

    assert not tmp_path.exists()

    loaded_datadict = datadict_from_hdf5(data_path)
    assert datasets_are_equal(loaded_datadict, correct_datadict, ignore_meta=True)


def test_deleting_files_when_done_with_lock_error(tmp_path):

    correct_datadict = DataDict(x=dict(unit='m'), y=dict(unit='m'), z=dict(axes=['x', 'y']))
    for i in range(1000):
        correct_datadict.add_data(x=i, y=i**2, z=i**3)

    datadict = DataDict(x=dict(unit='m'), y=dict(unit='m'), z=dict(axes=['x', 'y']))

    with DDH5Writer(datadict, str(tmp_path), safe_write_mode=True, file_timeout=5) as writer:
        for i in range(500):
            writer.add_data(x=i, y=i**2, z=i**3)

        data_path = writer.filepath
        tmp_path = data_path.parent/".tmp"

        lock_file = data_path.parent/f"~{data_path.stem}.lock"
        lock_file.touch()

        for i in range(500, 1000):
            writer.add_data(x=i, y=i ** 2, z=i ** 3)

        assert lock_file.exists()

        lock_file.unlink(missing_ok=False)

        assert tmp_path.exists()

    assert not tmp_path.exists()

    loaded_datadict = datadict_from_hdf5(data_path)
    assert datasets_are_equal(loaded_datadict, correct_datadict, ignore_meta=True)


def test_creation_of_not_reconstructed_error_due_to_error(tmp_path):

    datadict = DataDict(x=dict(unit='m'), y=dict(unit='m'), z=dict(axes=['x', 'y']))
    exception_was_raised = False

    try:
        with DDH5Writer(datadict, str(tmp_path), safe_write_mode=True, file_timeout=5) as writer:
            for i in range(500):
                writer.add_data(x=i, y=i**2, z=i**3)

            data_path = writer.filepath
            tmp_path = data_path.parent/".tmp"

            # Finds 10 files in tmp_path and deletes them
            files = list(tmp_path.rglob("*.ddh5"))[:10]
            for file in files:
                file.unlink()

    except AssertionError as e:
        exception_was_raised = True

    assert exception_was_raised
    not_reconstructed_tag = data_path.parent/"__not_reconstructed__.tag"
    assert not_reconstructed_tag.exists()
    assert tmp_path.exists()
