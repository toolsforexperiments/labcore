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


def refilter_data_select(*events):
    ds.data_select()

search_data_typed = ds.param.watch(refilter_data_select, ['search_term'])

pn.template.BootstrapTemplate(
    site="labcore",
    title="data explorer",
    sidebar=[],
    main=[ds, loader, loader.plot],
).servable()


#--- Watchdog Test

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class MyEventHandler(FileSystemEventHandler):
    def on_created(self, event) -> None:
        ds.group_select()

event_handler = MyEventHandler()
observer = Observer()
observer.schedule(event_handler, path=".", recursive=True)
observer.start()

def watchdog_check():
    observer.stop()
    observer.join()

pn.state.add_periodic_callback(watchdog_check, 1000)