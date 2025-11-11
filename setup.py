from setuptools import setup
from versioningit import get_cmdclasses

if __name__ == "__main__":
    setup(
        cmdclass=get_cmdclasses(),
        entry_points={
            'console_scripts': [
                'autoplot = scripts.monitr_server:run_autoplot',
                'pyro-ns = scripts.pyro_nameserver:run_pyro_nameserver',
            ],
        },
    )
