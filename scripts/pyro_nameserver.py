import argparse
import logging
import os
import sys

# Configure logging to output to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pyro_nameserver():
    """
    Run the Pyro4 nameserver directly in the terminal.
    This allows stopping the server with Ctrl+C.
    """
    parser = argparse.ArgumentParser(
        description="Start Pyro4 nameserver for QICK integration. "
                    "The nameserver runs in the foreground and can be stopped with Ctrl+C."
    )
    parser.add_argument(
        '--host',
        '-n',
        default='localhost',
        help='Host address for the nameserver (default: localhost)'
    )
    parser.add_argument(
        '--port',
        '-p',
        type=int,
        default=8888,
        help='Port number for the nameserver (default: 8888)'
    )

    args = parser.parse_args()

    # Set required environment variables for Pyro4
    os.environ["PYRO_SERIALIZERS_ACCEPTED"] = "pickle"
    os.environ["PYRO_PICKLE_PROTOCOL_VERSION"] = "4"

    logger.info(f"Starting Pyro4 nameserver on {args.host}:{args.port}")
    logger.info("Press Ctrl+C to stop the nameserver")

    # Build the command to execute pyro4-ns
    cmd = [
        "pyro4-ns",
        "-n",
        args.host,
        "-p",
        str(args.port),
    ]

    try:
        # Replace the current process with pyro4-ns
        # This will run pyro4-ns directly in the terminal
        os.execvp("pyro4-ns", cmd)
    except FileNotFoundError:
        logger.error("pyro4-ns command not found. Please install Pyro4 with: pip install Pyro4")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start Pyro4 nameserver: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_pyro_nameserver()