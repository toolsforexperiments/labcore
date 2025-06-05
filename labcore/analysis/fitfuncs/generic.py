from typing import Tuple, Any, Optional, Union, Dict, List

import numpy as np
import lmfit

from ..fit import Fit


class Cosine(Fit):
    @staticmethod
    def model(
        coordinates: np.ndarray, A: float, f: float, phi: float, of: float
    ) -> np.ndarray:
        """$A \cos(2 \pi f x + \phi) + of$"""
        return A * np.cos(2 * np.pi * coordinates * f + phi) + of

    @staticmethod
    def guess(
        coordinates: Union[Tuple[np.ndarray, ...], np.ndarray], data: np.ndarray
    ) -> Dict[str, float]:
        of = np.mean(data)
        A = (np.max(data) - np.min(data)) / 2.0

        # Making sure that coordinates is ndarray.
        # Changing the type in the signature will create a different mypy error.
        assert isinstance(coordinates, np.ndarray)
        fft_val = np.fft.rfft(data)[1:]
        fft_frq = np.fft.rfftfreq(
            data.size, np.mean(coordinates[1:] - coordinates[:-1])
        )[1:]
        idx = np.argmax(np.abs(fft_val))
        f = fft_frq[idx]
        phi = np.angle(fft_val[idx])

        return dict(A=A, of=of, f=f, phi=phi)


class Exponential(Fit):
    @staticmethod
    def model(coordinates: np.ndarray, a: float, b: float) -> np.ndarray:
        """a * b ** x"""
        return a * b**coordinates

    @staticmethod
    def guess(
        coordinates: Union[Tuple[np.ndarray, ...], np.ndarray], data: np.ndarray
    ) -> Dict[str, float]:
        return dict(a=1, b=2)


class ExponentialDecay(Fit):
    @staticmethod
    def model(coordinates, A, of, tau) -> np.ndarray:
        """$A * \exp(-x/\tau) + of$"""
        return A * np.exp(-coordinates/tau) + of

    @staticmethod
    def guess(coordinates, data):

        # offset guess: The mean of the last 10 percent of the data
        of = np.mean(data[-data.size//10:])

        # amplitude guess: difference between max and min.
        A = np.abs(np.max(data) - np.min(data))
        if data[0] < data[-1]:
            A *= -1

        # tau guess: pick the point where we reach roughly 1/e
        one_over_e_val = of + A/3.
        one_over_e_idx = np.argmin(np.abs(data-one_over_e_val))
        tau = coordinates[one_over_e_idx]

        return dict(A=A, of=of, tau=tau)


class Linear(Fit):
    @staticmethod
    def model(coordinates, m, of) -> np.ndarray:
        """$A * \exp(-x/\tau) + of$"""
        return m * coordinates + of

    @staticmethod
    def guess(coordinates, data):

        # amplitude guess: difference between  max and min y over the max and min x.
        m = np.abs(np.max(data) - np.min(data))/np.abs(np.max(coordinates) - np.min(coordinates))

        # offset guess: how far shifted the linear function is along y
        of = data[0] - m * coordinates[0]

        return dict(m=m, of=of)


class ExponentiallyDecayingSine(Fit):
    @staticmethod
    def model(coordinates, A, of, f, phi, tau) -> np.ndarray:
        """$A \sin(2*\pi*(f*x + \phi/360)) \exp(-x/\tau) + of$"""
        return A * np.sin(2 * np.pi * (f * coordinates + phi/360)) * np.exp(-coordinates/tau) + of

    @staticmethod
    def guess(coordinates, data):
        """This guess will ignore the first value because since it usually is not relaiable."""

        # offset guess: The mean of the data
        of = np.mean(data)

        # amplitude guess: difference between max and min.
        A = np.abs(np.max(data) - np.min(data)) / 2.
        if data[0] < data[-1]:
            A *= -1

        # f guess: Maximum of the absolute value of the fourier transform.
        fft_data = np.fft.rfft(data)[1:]
        fft_coordinates = np.fft.rfftfreq(data.size, coordinates[1] - coordinates[0])[1:]

        # note to confirm, could there be multiple peaks? I am always taking the first one here.
        f_max_index = np.argmax(fft_data)
        f = fft_coordinates[f_max_index]

        # phi guess
        phi = -np.angle(fft_data[f_max_index], deg=True)

        # tau guess: pick the point where we reach roughly 1/e
        one_over_e_val = of + A/3.
        one_over_e_idx = np.argmin(np.abs(data-one_over_e_val))
        tau = coordinates[one_over_e_idx]

        return dict(A=A, of=of, phi=phi, f=f, tau=tau)


class Gaussian(Fit):
    @staticmethod
    def model(coordinates, x0, sigma, A, of):
        """$A * np.exp(-(x-x_0)^2/(2\sigma^2)) + of"""
        return A * np.exp(-(coordinates - x0) ** 2 / (2 * sigma ** 2)) + of

    @staticmethod
    def guess(coordinates, data):
        # TODO: very crude at the moment, not likely to work well with not-so-nice data.
        of = np.mean(data)
        dev = data - of
        i_max = np.argmax(np.abs(dev))
        x0 = coordinates[i_max]
        A = data[i_max] - of
        sigma = np.abs((coordinates[-1] - coordinates[0])) / 20
        return dict(x0=x0, sigma=sigma, A=A, of=of)

