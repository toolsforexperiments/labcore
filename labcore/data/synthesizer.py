
import numpy as np

from abc import ABC, abstractmethod

"""

Implementation would work like this given x (n-dimensional array):

sine() = SineSynthesizer()
model = sine.generate(x, noise_std = 0.5, A = 2, f = 3)

-------------
These two lines would create a synthesizer representing a sine wave and generate models 
for each set of data stored in x.

Std for Gaussian distribution passed to noise() is 0.5 --> maybe add way to have multiple 
                                                           distributions for different data sets?

A, f are passed to SineSynthesizer's model() as kwargs

Synthesizer's generate() then applies the noise to the model and returns an array with
the same dimension that was passed

"""


class Synthesizer(ABC):
    @abstractmethod
    def model(self, coordinates, *args, **kwargs):
        pass

    def generate(self, coordinates, noise_std = 1.0, **model_kwargs):

        one_d = coordinates.ndim == 1
        coordinates = np.atleast_2d(coordinates)
        model_outputs = np.array([self.model(coords, **model_kwargs) + self.noise(noise_std)
                                  for coords in coordinates])
        if (one_d):
            return model_outputs.squeeze()

        return model_outputs
    
    def noise(self, std):
        return np.random.normal(scale = std)
    

class ExponentialSynthesizer(Synthesizer):

    def model(self, coordinates, base = np.e):
        return base ** coordinates
    


class SineSynthesizer(Synthesizer):

    def model(self, coordinates, A = 1, f = 1, phi = 0, of = 0):
        return A * np.sin(2 * np.pi * coordinates * f + phi) + of


class GaussianSynthesizer(Synthesizer):

    def model(self, coordinates, x0 = 0, sigma = 1, A = 1, of = 0):
        return A * np.exp(-((coordinates - x0) ** 2) / (2 * sigma ** 2)) + of