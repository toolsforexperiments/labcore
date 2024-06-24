import numpy as np
from collections.abc import Iterable
from pprint import pprint


from labcore.measurement.sweep import AsyncRecord
from labcore.measurement.record import make_data_spec

config = None

# TODO: Define a way of replacing the config dictionary in the part where you write the code for some defined options.
class QickBoardSweep(AsyncRecord):

    """
    Class is called QickBoard because QickSweep is already a thing
    """

    def __init__(self, *specs, **kwargs):
        
        print("I am in the init")

        self.communicator = {}
        self.specs = []
        for s in specs:
            spec = make_data_spec(s)
            self.specs.append(spec)


    def setup(self, fun, *args, **kwargs):

        # Checks that the config is not None
        if config is None:
            raise Exception("QickSweep: config is not set")
    
        conf = config.config()
        qick_program = fun(soccfg=conf[0], cfg=conf[1])
        self.communicator["qick_program"] = qick_program


    @classmethod
    def flatten(cls, iterable):
        for item in iterable:
            # Check if the item is an iterable but not a string or a numpy array
            if isinstance(item, Iterable) and not isinstance(item, (str, bytes, np.ndarray)):
                yield from cls.flatten(item)
            else:
                yield item
        

    def collect(self, *args, **kwargs):
        # Get the measurement data.
        data = self.communicator["qick_program"].acquire(config.soc)
        
        # This always comes in 3 parts, check the qick documentation for more info
        expt_pts, avg_di, avg_dq = data

        print(f'Inside of data:')
        pprint(data)
        yield data

        # for d in zip(*data):
        #     print(f'Inside the for loop')
        #     print(d)
        #     yield d
        





