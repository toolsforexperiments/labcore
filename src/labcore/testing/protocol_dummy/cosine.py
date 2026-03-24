import logging
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from labcore.analysis import DatasetAnalysis
from labcore.analysis.fit import FitResult
from labcore.analysis.fitfuncs.generic import Cosine
from labcore.data.datadict_storage import datadict_from_hdf5
from labcore.measurement import Sweep
from labcore.measurement.record import dependent, independent, recording
from labcore.measurement.storage import run_and_save_sweep
from labcore.protocols.base import CheckResult, ProtocolOperation
from labcore.testing.protocol_dummy.parameters import (
    CosineAmplitude,
    CosineFrequency,
    CosineOffset,
    CosinePhase,
)

plt.switch_backend("agg")

logger = logging.getLogger(__name__)


class CosineOperation(ProtocolOperation):
    SNR_THRESHOLD = 2

    def __init__(self, params: Any = None) -> None:
        super().__init__()

        self.frequency: CosineFrequency
        self.phase: CosinePhase
        self.offset: CosineOffset
        self._register_inputs(
            frequency=CosineFrequency(params),
            phase=CosinePhase(params),
            offset=CosineOffset(params),
        )
        self.amplitude: CosineAmplitude
        self._register_outputs(amplitude=CosineAmplitude(params))

        self.condition = f"Success if the SNR of the Cosine fit is bigger than the current threshold of {self.SNR_THRESHOLD}"

        self._register_check("snr_check", self._check_snr)
        self._register_success_update(self.amplitude, lambda: self.fit_result.params["A"].value)

        self.independents = {"x_values": []}
        self.dependents = {"y_values": []}

        self.fit_result: FitResult | None = None
        self.snr: float | None = None

    def _measure_dummy(self) -> Path:
        """
        Creates fake data that looks like a Cosine with noise using a sweep.
        Model: A * cos(2*pi*f*x + phi) + of
        """
        logger.info("Starting Cosine measurement (generating fake Cosine data)")

        # True Cosine parameters
        true_amplitude = 5.0
        true_frequency = 0.2
        true_phase = np.pi / 4
        true_offset = 2.0

        # Create x values for the sweep
        x_values = np.linspace(0, 20, 100)

        @recording(independent("x"), dependent("y"))
        def measure_cosine(x_val: float) -> tuple[float, float]:
            """Generate a single Cosine data point with noise"""
            y_clean = (
                true_amplitude * np.cos(2 * np.pi * true_frequency * x_val + true_phase)
                + true_offset
            )
            noise = np.random.normal(0, 0.3)
            return x_val, y_clean + noise

        sweep = Sweep(x_values, measure_cosine)

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
        """Fit the data to a Cosine"""
        assert self.data_loc is not None
        with DatasetAnalysis(self.data_loc, self.name) as ds:
            x = np.asarray(self.independents["x_values"])
            y = np.asarray(self.dependents["y_values"])

            # Perform Cosine fit
            fit = Cosine(x, y)
            self.fit_result = cast(FitResult, fit.run())
            fit_curve = self.fit_result.eval()
            residuals = y - fit_curve

            # Calculate SNR
            amplitude = self.fit_result.params["A"].value
            noise = np.std(residuals)
            snr = float(np.abs(amplitude / (4 * noise)))
            self.snr = snr

            # Create plot
            fig, ax = plt.subplots()
            ax.set_title("Cosine - Amplitude Fit")
            ax.set_xlabel("X Values (A.U)")
            ax.set_ylabel("Y Values (A.U)")
            ax.plot(x, y, "o", label="Data", markersize=4)
            ax.plot(x, fit_curve, "-", label="Cosine Fit", linewidth=2)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Save results
            ds.add(fit_curve=fit_curve, fit_result=self.fit_result, snr=snr)
            ds.add_figure(self.name, fig=fig)

            image_path = ds._new_file_path(ds.savefolders[1], self.name, suffix="png")
            self.figure_paths.append(image_path)

            self.report_output.append(
                f"## Cosine - Amplitude Fit\n"
                f"Generated fake Cosine data and fitted it to extract amplitude.\n"
                f"Data Path: `{self.data_loc}`\n"
                f"Plot:\n"
            )
            self.report_output.append(image_path.resolve())
            self.report_output.append(
                f"**Fit Report:**\n```\n{self.fit_result.lmfit_result.fit_report()}\n```\n"
            )

    def _check_snr(self) -> CheckResult:
        snr = self.snr or 0.0
        passed = snr >= self.SNR_THRESHOLD
        if passed:
            self.report_output.append(f"Fit was **SUCCESSFUL** with an SNR of {snr:.3f}.\n")
        else:
            self.report_output.append(f"Fit was **UNSUCCESSFUL** with an SNR of {snr:.3f}. NO value has been changed.\n")
        return CheckResult("snr_check", passed, f"SNR={snr:.3f}, threshold={self.SNR_THRESHOLD}")
