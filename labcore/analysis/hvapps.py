from datetime import datetime
from pathlib import Path

import param
import panel as pn
from panel.widgets import RadioButtonGroup as RBG, MultiSelect, Select

from ..data.datadict_storage import find_data, timestamp_from_path


class DataSelect(pn.viewable.Viewer):

    SYM = {
        'complete': '✅',
        'star': '😁',
        'bad': '😭',
        'trash': '❌',
    }
    DATAFILE = 'data.ddh5'

    selected_path = param.Parameter(None)

    @staticmethod
    def date2label(date_tuple):
        return "-".join((str(k) for k in date_tuple))

    @staticmethod
    def label2date(label):
        return tuple(int(l) for l in label[:10].split("-"))

    @staticmethod
    def group_data(data_list):
        ret = {}
        for path, info in data_list.items():
            ts = timestamp_from_path(path)
            date = (ts.year, ts.month, ts.day)
            if date not in ret:
                ret[date] = {}
            ret[date][path] = info
        return ret

    def __panel__(self):
        return self.layout

    def __init__(self, data_root, size=10):
        super().__init__()

        self.size = size

        # this contains a dict with the structure:
        # { date (as tuple): 
        #    { path of the dataset folder : (list of subdirs, list of files) }
        # }
        self.data_sets = self.group_data(find_data(root=data_root))

        self.layout = pn.Column()

        # two selectors for data selection
        self._group_select_widget = MultiSelect(
            name='Date', 
            size=self.size,
            width=150,
            stylesheets = [selector_stylesheet]
        )
        self._data_select_widget = Select(
            name='Data set', 
            size=self.size,
            width=500,
            stylesheets = [selector_stylesheet]
        )
        self.layout.append(pn.Row(self._group_select_widget, self.data_select))

        # a simple info panel about the selection
        self.lbl = pn.widgets.StaticText(
            stylesheets=[selector_stylesheet], 
            css_classes=['ttlabel'],
        )
        self.layout.append(self.info_panel)

        opts = {}
        for k in self.data_sets.keys():
            lbl = self.date2label(k) + f' [{len(self.data_sets[k])}]'
            opts[lbl] = k
        self._group_select_widget.options = opts

    @pn.depends("_group_select_widget.value")
    def data_select(self):
        opts = {}
        for d in self._group_select_widget.value:
            for dset, (dirs, files) in self.data_sets[d].items():
                ts = timestamp_from_path(dset)
                time = f"{ts.hour:02d}:{ts.minute:02d}:{ts.second:02d}"
                uuid = f"{dset.stem[18:26]}"
                name = f"{dset.stem[27:]}"
                lbl = f" {time} - {uuid} - {name} "
                for k in ['complete', 'star', 'trash']:
                    if f'__{k}__.tag' in files:
                        lbl += self.SYM[k]             
                opts[lbl] = dset

        self._data_select_widget.options = opts
        return self._data_select_widget

    @pn.depends("_data_select_widget.value")
    def info_panel(self):
        path = self._data_select_widget.value
        if isinstance(path, Path):
            path = path / self.DATAFILE
        self.lbl.value = f"Path: {path}"
        self.selected_path = path
        return self.lbl


selector_stylesheet = """
:host .bk-input {
    font-family: monospace;
}

:host(.ttlabel) .bk-clearfix {
    font-family: monospace;
}
"""