
import numpy as np

import dataclasses

"""

Implementation would work like this given x (n-dimensional array):

sine() = SineDataGen()
model = sine.generate(x, noise_std = 0.5, A = 2, f = 3)

-------------
These two lines would create a synthesizer representing a sine wave and generate models 
for each set of data stored in x.

Std for Gaussian distribution passed to noise() is 0.5 --> maybe add way to have multiple 
                                                           distributions for different data sets?

A, f are passed to SineDataGen's model() as kwargs

DataGen's generate() then applies the noise to the model and returns an array with
the same dimension that was passed

"""

"""
generate should now work such that the following can be done:

x = [np array of coordinates]
sine = SineDataGen(A = 2, f = 3)

coords = sine.generate() --> uses A = 2, f = 3 
coords = sine.generate(A = 5) --> uses A = 5, f = 2

"""

@dataclasses.dataclass
class DataGen:
    noise_std : float = 1.0
    
    @staticmethod
    def model(coordinates, *args, **kwargs):
        pass

    def generate(self, coordinates, **kwargs):

        # updates previously set dataclass fields
        # coords = 
        params = dataclasses.asdict(self) 
        params.update(kwargs)
        noise_std = params.pop('noise_std')

        one_d = coordinates.ndim == 1
        coordinates = np.atleast_2d(coordinates)
        model_outputs = np.array([self.model(coords, **params) +
                                  self.noise(coords, noise_std)
                                  for coords in coordinates])
        if (one_d):
            return model_outputs.squeeze()

        return model_outputs
    
    @staticmethod
    def noise(coordinates, std):
        return np.random.normal(scale = std, size = len(coordinates))
    

@dataclasses.dataclass
class ExponentialDataGen(DataGen):

    base: float = np.e

    @staticmethod
    def model(coordinates, base = np.e):
        return base ** coordinates
    


@dataclasses.dataclass
class SineDataGen(DataGen):

    A : float = 1
    f : float = 1
    phi : float = 0
    of : float = 0

    @staticmethod
    def model(coordinates, A = 1, f = 1, phi = 0, of = 0):
        return A * np.sin(2 * np.pi * coordinates * f + phi) + of


@dataclasses.dataclass
class GaussianDataGen(DataGen):

    x0 : float = 0
    sigma : float = 1
    A : float = 1
    of : float = 0

    @staticmethod
    def model(coordinates, x0 = 0, sigma = 1, A = 1, of = 0):
        return A * np.exp(-((coordinates - x0) ** 2) / (2 * sigma ** 2)) + of