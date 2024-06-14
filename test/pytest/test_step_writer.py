import time
from datetime import datetime

from labcore.measurement.storage import _create_datadict_structure

from labcore.data.datadict import DataDict
from labcore.data.datadict_storage import DDH5Writer


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
        
        
        

                











