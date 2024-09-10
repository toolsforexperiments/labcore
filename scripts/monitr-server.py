from datetime import datetime
from pathlib import Path

import panel as pn
pn.extension()

from labcore.analysis.hvapps import DataSelect, DDH5LoaderNode

ds = DataSelect('.')
loader = DDH5LoaderNode()

def data_selected_cb(*events):
    loader.file_path = events[0].new

watch_data_selected = ds.param.watch(data_selected_cb, ['selected_path'])

pn.template.BootstrapTemplate(
    site="labcore",
    title="data explorer",
    sidebar=[],
    main=[ds, loader, loader.plot],
).servable()