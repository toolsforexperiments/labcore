import logging
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from labcore.analysis import DatasetAnalysis
from labcore.analysis.fit import FitResult
from labcore.analysis.fitfuncs.generic import Exponential
from labcore.data.datadict_storage import datadict_from_hdf5
from labcore.measurement.record import dependent, independent, recording
from labcore.measurement.storage import run_and_save_sweep
from labcore.measurement.sweep import Sweep
from labcore.protocols.base import OperationStatus, ParamImprovement, ProtocolOperation
from labcore.testing.protocol_dummy.parameters import ExponentialA, ExponentialB

plt.switch_backend("agg")

logger = logging.getLogger(__name__)


class ExponentialOperation(ProtocolOperation):
    SNR_THRESHOLD = 2

    def __init__(self, params: Any = None) -> None:
        super().__init__()

        self.b: ExponentialB
        self._register_inputs(b=ExponentialB(params))
        self.a: ExponentialA
        self._register_outputs(a=ExponentialA(params))

        self.condition = f"Success if the SNR of the Exponential fit is bigger than the current threshold of {self.SNR_THRESHOLD}"

        self.independents = {"x_values": []}
        self.dependents = {"y_values": []}

        self.fit_result: FitResult | None = None
        self.snr: float | None = None

    def _measure_dummy(self) -> Path:
        """
        Creates fake data that looks like an Exponential with noise using a sweep.
        Model: a * b^x
        """
        logger.info(
            "Starting Exponential measurement (generating fake Exponential data)"
        )

        # True Exponential parameters
        true_a = 1
        true_b = 2

        # Create x values for the sweep (shorter range to avoid overflow)
        x_values = np.linspace(0, 10, 50)

        # Define a measurement function that generates Exponential data with noise
        @recording(independent("x"), dependent("y"))
        def measure_exponential(x_val: float) -> tuple[float, float]:
            """Generate a single Exponential data point with noise"""
            y_clean = true_a * (true_b**x_val)
            # Noise proportional to signal magnitude (5% relative noise)
            noise = np.random.normal(0, 0.01 * y_clean)
            return x_val, y_clean + noise

        # Create the sweep using Sweep directly
        sweep = Sweep(x_values, measure_exponential)

        # Run and save the sweep
        logger.debug("Sweep created, running measurement")
        loc, data_array = run_and_save_sweep(sweep, "data", self.name)
        logger.info(f"Measurement complete, data saved to {loc}")

        return Path(loc)

    def _load_data_dummy(self) -> None:
        """Load the generated fake data"""
        assert self.data_loc is not None
        path = self.data_loc / "data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)

        self.independents["x_values"] = data["x"]["values"]
        self.dependents["y_values"] = data["y"]["values"]

    def analyze(self) -> None:
        """Fit the data to an Exponential"""
        assert self.data_loc is not None
        with DatasetAnalysis(self.data_loc, self.name) as ds:
            x = np.asarray(self.independents["x_values"])
            y = np.asarray(self.dependents["y_values"])

            # Perform Exponential fit
            fit = Exponential(x, y)
            self.fit_result = cast(FitResult, fit.run())
            fit_curve = self.fit_result.eval()
            residuals = y - fit_curve

            # Calculate SNR
            # For exponential, use relative noise (residuals/signal) to avoid bias from growth
            relative_residuals = residuals / fit_curve
            relative_noise = np.std(relative_residuals)
            snr = float(np.abs(1 / (4 * relative_noise)))
            self.snr = snr

            # Create plot
            fig, ax = plt.subplots()
            ax.set_title("Exponential - Coefficient Fit")
            ax.set_xlabel("X Values (A.U)")
            ax.set_ylabel("Y Values (A.U)")
            ax.plot(x, y, "o", label="Data", markersize=4)
            ax.plot(x, fit_curve, "-", label="Exponential Fit", linewidth=2)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Save results
            ds.add(fit_curve=fit_curve, fit_result=self.fit_result, snr=snr)
            ds.add_figure(self.name, fig=fig)

            image_path = ds._new_file_path(ds.savefolders[1], self.name, suffix="png")
            self.figure_paths.append(image_path)

    def evaluate(self) -> OperationStatus:
        """
        Evaluate if the fit was successful based on SNR threshold.
        If successful, update the 'a' output parameter with the fitted coefficient value.
        """
        header = (
            f"## Exponential - Coefficient Fit\n"
            f"Generated fake Exponential data and fitted it to extract coefficient.\n"
            f"Data Path: `{self.data_loc}`\n"
            f"Plot:\n"
        )
        plot_image = self.figure_paths[0].resolve()

        assert self.snr is not None
        assert self.fit_result is not None
        if self.snr >= self.SNR_THRESHOLD:
            logger.info(
                f"SNR of {self.snr} is bigger than threshold of {self.SNR_THRESHOLD}. Applying new values"
            )

            old_value = self.a()
            new_value = self.fit_result.params["a"].value

            logger.info(f"Updating {self.a.name} from {old_value} to {new_value}")
            self.a(new_value)

            self.improvements = [ParamImprovement(old_value, new_value, self.a)]

            msg_2 = (
                f"Fit was **SUCCESSFUL** with an SNR of {self.snr:.3f}.\n"
                f"{self.a.name} updated: {old_value} -> {new_value:.3f}\n\n"
                f"**Fit Report:**\n```\n{str(self.fit_result.lmfit_result.fit_report())}\n```\n\n"
            )

            self.report_output = [header, plot_image, msg_2]

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
