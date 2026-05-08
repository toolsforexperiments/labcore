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
        # Auto-load data if toggle is enabled
        if loader.auto_load_toggle.value:
            import asyncio
            # Schedule async load in event loop
            asyncio.create_task(loader.load_and_preprocess())

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
    parser.add_argument('Datapath', nargs='?', default='.', help='Path to the data directory (default: current directory)')
    parser.add_argument('-p', '--port', type=int, default=19530, help='Port to run the server on (default: 19530)')
    parser.add_argument('-a', '--address', type=str, default='0.0.0.0', help='Address to bind to (default: 0.0.0.0)')
    parser.add_argument('-o', '--allow-origin', type=str, default=None,
                        help='Allowed websocket origins (comma-separated, e.g., "host1:19530,host2:19530")')

    args = parser.parse_args()

    data_root = Path(args.Datapath)
    if (not data_root.is_dir()):
        logger.error("Provided Path was invalid.\nPlease provide a path to an existing directory housing your data.")
        return

    logger.info(f"Running Labcore.Autoplot on data from {data_root}")
    logger.info(f"Server running on http://{args.address}:{args.port}")

    template = make_template(data_root)

    # Parse websocket origins
    websocket_origin = None
    if args.allow_origin:
        websocket_origin = [o.strip() for o in args.allow_origin.split(',')]
        logger.info(f"Allowed websocket origins: {websocket_origin}")

    template.show(port=args.port, address=args.address, websocket_origin=websocket_origin)

make_template(".").servable()
