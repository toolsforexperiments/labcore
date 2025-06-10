import argparse
import logging
from pathlib import Path


import panel as pn
pn.extension()

from labcore.analysis.hvapps import DataSelect, DDH5LoaderNode

logger = logging.getLogger(__file__)

def make_template(data_root='.'):
    ds = DataSelect(data_root)
    loader = DDH5LoaderNode()

    def data_selected_cb(*events):
        loader.file_path = events[0].new

    watch_data_selected = ds.param.watch(data_selected_cb, ['selected_path'])

    def refilter_data_select(*events):
        ds.data_select()

    search_data_typed = ds.param.watch(refilter_data_select, ['search_term'])

    temp = pn.template.BootstrapTemplate(
        site="labcore",
        title="autoplot",
        sidebar=[],
        main=[ds, loader] 
    )

    return temp


def run_autoplot():
    parser = argparse.ArgumentParser(
        description="Data monitoring program made for Pfaff lab by Rocky Daehler, building"
                    " on Plottr made by Wolfgang Pfaff. Run command on it's own to start the"
                    " application, and pass an (optional) path to the data directory as a"
                    " second argument.")
    parser.add_argument('Datapath', nargs='?', default='.')

    args = parser.parse_args()

    data_root = Path(args.Datapath)
    if (not data_root.is_dir()):
        logger.error("Provided Path was invalid.\nPlease provide a path to an existing directory housing your data.")
        return

    logger.info(f"Running Labcore.Autoplot on data from {data_root}")

    template = make_template(data_root)
    template.show()

make_template(".").servable()
