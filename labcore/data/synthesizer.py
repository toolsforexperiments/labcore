
import numpy as np

from abc import ABC, abstractmethod


class Synthesizer(ABC):
    @abstractmethod
    def model(self, coordinates, *args, **kwargs):
        pass

    def generate(self, coordinates):
        return self.model(coordinates) + self.noise()
    
    def noise(self, std = 1.0):
        return np.random.normal(scale = std)