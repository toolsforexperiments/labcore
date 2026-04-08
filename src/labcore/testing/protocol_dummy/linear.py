import logging
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from labcore.analysis import DatasetAnalysis
from labcore.analysis.fit import FitResult
from labcore.analysis.fitfuncs.generic import Linear
from labcore.data.datadict_storage import datadict_from_hdf5
from labcore.measurement.record import dependent, independent, recording
from labcore.measurement.storage import run_and_save_sweep
from labcore.measurement.sweep import Sweep
from labcore.protocols.base import (
    EvaluateResult,
    OperationStatus,
    ParamImprovement,
    ProtocolOperation,
)
from labcore.testing.protocol_dummy.parameters import LinearOffset, LinearSlope

plt.switch_backend("agg")

logger = logging.getLogger(__name__)


class LinearOperation(ProtocolOperation):
    SNR_THRESHOLD = 2

    def __init__(self, params: Any = None) -> None:
        super().__init__()

        self.offset: LinearOffset
        self._register_inputs(offset=LinearOffset(params))
        self.slope: LinearSlope
        self._register_outputs(slope=LinearSlope(params))

        self.condition = f"Success if the SNR of the Linear fit is bigger than the current threshold of {self.SNR_THRESHOLD}"

        self.independents = {"x_values": []}
        self.dependents = {"y_values": []}

        self.fit_result: FitResult | None = None
        self.snr: float | None = None

    def _measure_dummy(self) -> Path:
        """
        Creates fake data that looks like a Linear function with noise using a sweep.
        Model: m * x + of
        """
        logger.info("Starting Linear measurement (generating fake Linear data)")

        # True Linear parameters
        true_slope = 2.5
        true_offset = 3.0

        # Create x values for the sweep
        x_values = np.linspace(-5, 5, 50)

        # Define a measurement function that generates Linear data with noise
        @recording(independent("x"), dependent("y"))
        def measure_linear(x_val: float) -> tuple[float, float]:
            """Generate a single Linear data point with noise"""
            y_clean = true_slope * x_val + true_offset
            noise = np.random.normal(0, 0.5)
            return x_val, y_clean + noise

        # Create the sweep using Sweep directly
        sweep = Sweep(x_values, measure_linear)

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
        """Fit the data to a Linear function"""
        assert self.data_loc is not None
        with DatasetAnalysis(self.data_loc, self.name) as ds:
            x = np.asarray(self.independents["x_values"])
            y = np.asarray(self.dependents["y_values"])

            # Perform Linear fit
            fit = Linear(x, y)
            self.fit_result = cast(FitResult, fit.run())
            fit_curve = self.fit_result.eval()
            residuals = y - fit_curve

            # Calculate SNR
            # Use relative noise to avoid bias from y-value range
            signal_range = np.max(np.abs(fit_curve)) - np.min(np.abs(fit_curve))
            noise = np.std(residuals)
            # SNR based on noise relative to signal range
            snr = float(np.abs(signal_range / (4 * noise)))
            self.snr = snr

            # Create plot
            fig, ax = plt.subplots()
            ax.set_title("Linear - Slope Fit")
            ax.set_xlabel("X Values (A.U)")
            ax.set_ylabel("Y Values (A.U)")
            ax.plot(x, y, "o", label="Data", markersize=4)
            ax.plot(x, fit_curve, "-", label="Linear Fit", linewidth=2)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Save results
            ds.add(fit_curve=fit_curve, fit_result=self.fit_result, snr=snr)
            ds.add_figure(self.name, fig=fig)

            image_path = ds._new_file_path(ds.savefolders[1], self.name, suffix="png")
            self.figure_paths.append(image_path)

    def evaluate(self) -> EvaluateResult:
        """
        Evaluate if the fit was successful based on SNR threshold.
        If successful, update the slope output parameter with the fitted slope value.
        """
        header = (
            f"## Linear - Slope Fit\n"
            f"Generated fake Linear data and fitted it to extract slope.\n"
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

            old_value = self.slope()
            new_value = self.fit_result.params["m"].value

            logger.info(f"Updating {self.slope.name} from {old_value} to {new_value}")
            self.slope(new_value)

            self.improvements = [ParamImprovement(old_value, new_value, self.slope)]

            msg_2 = (
                f"Fit was **SUCCESSFUL** with an SNR of {self.snr:.3f}.\n"
                f"{self.slope.name} updated: {old_value} -> {new_value:.3f}\n\n"
                f"**Fit Report:**\n```\n{str(self.fit_result.lmfit_result.fit_report())}\n```\n\n"
            )

            self.report_output = [header, plot_image, msg_2]

            return EvaluateResult(OperationStatus.SUCCESS)

        logger.info(
            f"SNR of {self.snr} is smaller than threshold of {self.SNR_THRESHOLD}. Evaluation failed"
        )

        msg_2 = (
            f"Fit was **UNSUCCESSFUL** with an SNR of {self.snr:.3f}.\n"
            f"NO value has been changed.\n"
            f"Fit Report:\n\n```\n{str(self.fit_result.lmfit_result.fit_report())}\n```\n"
        )
        self.report_output = [header, plot_image, msg_2]

        return EvaluateResult(OperationStatus.FAILURE)
