import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from labcore.analysis import DatasetAnalysis
from labcore.analysis.fitfuncs.generic import Gaussian
from labcore.data.datadict_storage import datadict_from_hdf5
from labcore.measurement.record import dependent, independent, recording
from labcore.measurement.storage import run_and_save_sweep
from labcore.measurement.sweep import Sweep
from labcore.protocols.base import OperationStatus, ParamImprovement, ProtocolOperation
from labcore.testing.protocol_dummy.parameters import (
    GaussianAmplitude,
    GaussianCenter,
    GaussianOffset,
    GaussianSigma,
)

plt.switch_backend("agg")

logger = logging.getLogger(__name__)


class GaussianOperation(ProtocolOperation):
    SNR_THRESHOLD = 2

    def __init__(self, params=None):
        super().__init__()

        self._register_inputs(
            center=GaussianCenter(None),
            sigma=GaussianSigma(None),
            offset=GaussianOffset(None),
        )
        self._register_outputs(amplitude=GaussianAmplitude(None))

        self.condition = f"Success if the SNR of the Gaussian fit is bigger than the current threshold of {self.SNR_THRESHOLD}"

        self.independents = {"x_values": []}
        self.dependents = {"y_values": []}

        self.fit_result = None
        self.snr = None

    def _measure_dummy(self) -> Path:
        """
        Creates fake data that looks like a Gaussian with noise using a sweep.
        Uses input parameters (center, sigma, offset) to generate the Gaussian.
        """
        logger.info("Starting Gaussian measurement (generating fake Gaussian data)")

        # Gaussian parameters: amplitude, center, width
        true_amplitude = 10.0
        true_center = 0.5
        true_sigma = 2.0

        # Create x values for the sweep
        x_values = np.linspace(-10, 10, 100)

        @recording(independent("x"), dependent("y"))
        def measure_gaussian(x_val):
            """Generate a single Gaussian data point with noise"""
            y_clean = true_amplitude * np.exp(
                -((x_val - true_center) ** 2) / (2 * true_sigma**2)
            )
            noise = np.random.normal(0, 0.5)
            return x_val, y_clean + noise

        sweep = Sweep(x_values, measure_gaussian)

        # Run and save the sweep
        logger.debug("Sweep created, running measurement")
        loc, data_array = run_and_save_sweep(sweep, "data", self.name)
        logger.info(f"Measurement complete, data saved to {loc}")

        return loc

    def _load_data_dummy(self):
        """Load the generated fake data"""
        path = self.data_loc / "data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)

        self.independents["x_values"] = data["x"]["values"]
        self.dependents["y_values"] = data["y"]["values"]

    def analyze(self):
        """Fit the data to a Gaussian"""
        with DatasetAnalysis(self.data_loc, self.name) as ds:
            x = self.independents["x_values"]
            y = self.dependents["y_values"]

            # Perform Gaussian fit
            fit = Gaussian(x, y)
            self.fit_result = fit.run()
            fit_curve = self.fit_result.eval()
            residuals = y - fit_curve

            # Calculate SNR
            amplitude = self.fit_result.params["A"].value
            noise = np.std(residuals)
            self.snr = np.abs(amplitude / (4 * noise))

            # Create plot
            fig, ax = plt.subplots()
            ax.set_title("Gaussian - Amplitude Fit")
            ax.set_xlabel("X Values (A.U)")
            ax.set_ylabel("Y Values (A.U)")
            ax.plot(x, y, "o", label="Data", markersize=4)
            ax.plot(x, fit_curve, "-", label="Gaussian Fit", linewidth=2)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Save results
            ds.add(fit_curve=fit_curve, fit_result=self.fit_result, snr=float(self.snr))
            ds.add_figure(self.name, fig=fig)

            image_path = ds._new_file_path(ds.savefolders[1], self.name, suffix="png")
            self.figure_paths.append(image_path)

    def evaluate(self) -> OperationStatus:
        """
        Evaluate if the fit was successful based on SNR threshold.
        If successful, update the amplitude output parameter with the fitted amplitude value.
        """
        header = (
            f"## Gaussian - Amplitude Fit\n"
            f"Generated fake Gaussian data and fitted it to extract amplitude.\n"
            f"Data Path: `{self.data_loc}`\n"
            f"Plot:\n"
        )
        plot_image = self.figure_paths[0].resolve()

        if self.snr >= self.SNR_THRESHOLD:
            logger.info(
                f"SNR of {self.snr} is bigger than threshold of {self.SNR_THRESHOLD}. Applying new values"
            )

            old_value = self.amplitude()
            new_value = self.fit_result.params["A"].value  # Gaussian amplitude

            logger.info(
                f"Updating {self.amplitude.name} from {old_value} to {new_value}"
            )
            self.amplitude(new_value)

            self.improvements = [ParamImprovement(old_value, new_value, self.amplitude)]

            msg_2 = (
                f"Fit was **SUCCESSFUL** with an SNR of {self.snr:.3f}.\n"
                f"{self.amplitude.name} updated: {old_value} -> {new_value:.3f}\n\n"
                f"**Fit Report:**\n```\n{str(self.fit_result.lmfit_result.fit_report())}\n```\n\n"
            )

            self.report_output.append(header)
            self.report_output.append(plot_image)
            self.report_output.append(msg_2)

            if self.total_attempts_made != 3:
                msg_3 = f"Protocol at  {self.total_attempts_made} repetitions, repeating for testing."
                self.report_output.append(msg_3)
                return OperationStatus.RETRY

            return OperationStatus.SUCCESS

        logger.info(
            f"SNR of {self.snr} is smaller than threshold of {self.SNR_THRESHOLD}. Evaluation failed"
        )

        msg_2 = (
            f"Fit was **UNSUCCESSFUL** with an SNR of {self.snr:.3f}.\n"
            f"NO value has been changed.\n"
            f"Fit Report:\n\n```\n{str(self.fit_result.lmfit_result.fit_report())}\n```\n"
        )
        self.report_output = [header, plot_image, msg_2]

        return OperationStatus.FAILURE
