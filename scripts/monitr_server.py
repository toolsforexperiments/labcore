from labcore.analysis.hvapps import DataSelect, DDH5LoaderNode
from datetime import datetime
from pathlib import Path

import argparse
import panel as pn
pn.extension()


def MakeTemplate(data_root='.'):
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
        title="data explorer",
        sidebar=[],
        main=[ds, loader, loader.plot],
    )

    return temp


def Run_Show():
    parser = argparse.ArgumentParser(
        description="Data monitored program made for Pfaff lab by Rocky Daehler, building"
                    " on Plottr made by Wolfgang Pfaff. Run command on it's own to start the"
                    " application, and pass an (optional) path to the data directory as a"
                    " second argument.")
    parser.add_argument('Datapath', nargs='?', default='.')

    args = parser.parse_args()

    data_root = Path(args.Datapath)
    if (not data_root.is_dir()):
        print("Provided Path was invalid.\nPlease provide a path to an existing directory housing your data.")
        return

    print(f"Running data monitoring application on data from {data_root}")

    template = MakeTemplate(data_root)
    template.show()


if __name__ == "__main__":
    template = MakeTemplate()
    template.show()
