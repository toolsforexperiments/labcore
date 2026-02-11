from typing import Union
import numpy as np
from labcore.data.datadict import str2dd
from pprint import pprint

# Define constants and parameters
amplitude = 2  # Amplitude of the resonator response
noise_level = 0.2  # Noise level

# Simulate the resonator response
def simulate_S21(center_frequency, Q_factor, frequency_range, num_points):
    frequencies = np.linspace(center_frequency - frequency_range / 2, center_frequency + frequency_range / 2, num_points)
    response = amplitude / (1 + 1j * (frequencies - center_frequency) / (center_frequency / Q_factor))
    response += np.random.normal(0, noise_level, len(frequencies))
    return response, frequencies
    

def resonator_dataset(center_frequency, Q_factor, frequency_range, reps = 10, num_points: int = 100):
    data = str2dd("signal(repetition, fs); fs[Hz]; testing[s];")
    response, frequencies = simulate_S21(center_frequency, Q_factor,frequency_range, num_points)
    for i in range(reps):
        for j in range(100):
            data.add_data(
                signal=response,
                fs = frequencies,
                testing=[j],
                repetition=np.arange(num_points, dtype=int)+1,
            )
    return data
# Plot the resonator response


def plot_resonator_response(frequencies, response):
    plt.figure()
    plt.plot(frequencies, np.abs(response), label='Amplitude')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Microwave Resonator Response')
    plt.grid(True)
    plt.show()
    
# Main function
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    center_frequency = 5e9  # Center frequency in Hz
    frequency_range = 1e9  # Frequency range in Hz
    Q_factor = 100
    num_points = 2000
    response, frequencies = simulate_S21(center_frequency, Q_factor, frequency_range, num_points)
    plot_resonator_response(frequencies, response)

