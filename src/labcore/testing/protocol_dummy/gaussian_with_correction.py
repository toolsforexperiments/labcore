"""
GaussianWithCorrectionOperation — demonstrates the Correction mechanism.

When the SNR check fails, a _ReduceNoiseLevelCorrection is applied before the
next attempt. Each application divides the measurement noise std by
`noise_reduction_factor` (a CorrectionParameter, default 3.0):

    noise_std: 5.0 → 1.67 → 0.56

With amplitude ≈ 10 and SNR_THRESHOLD = 2:

    SNR ≈ amplitude / (4 * noise_std)
    5.0  →  ~0.5   FAIL
    1.67 →  ~1.5   FAIL
    0.56 →  ~4.5   PASS

If the correction is exhausted (can_apply() returns False) before SNR passes,
correct() escalates the status to FAILURE and the protocol stops.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from labcore.analysis import DatasetAnalysis
from labcore.analysis.fit import FitResult
from labcore.analysis.fitfuncs.generic import Gaussian
from labcore.data.datadict_storage import datadict_from_hdf5
from labcore.measurement.record import dependent, independent, recording
from labcore.measurement.storage import run_and_save_sweep
from labcore.measurement.sweep import Sweep
from labcore.protocols.base import (
    CheckResult,
    Correction,
    EvaluateResult,
    OperationStatus,
    ParamImprovement,
    ProtocolOperation,
)
from labcore.testing.protocol_dummy.parameters import (
    GaussianAmplitude,
    GaussianCenter,
    GaussianNoiseReductionFactor,
    GaussianOffset,
    GaussianSigma,
)

plt.switch_backend("agg")

logger = logging.getLogger(__name__)


class _ReduceNoiseLevelCorrection(Correction):
    """
    Divides the operation's noise std by noise_reduction_factor on each application.

    Demonstrates a stateful Correction: _applications persists across retries
    so the correction knows when it has been exhausted.
    """

    name = "reduce_noise_level"
    description = "Divide measurement noise std by the noise_reduction_factor parameter"
    triggered_by = "snr_check"

    def __init__(
        self, operation: "GaussianWithCorrectionOperation", max_applications: int = 3
    ) -> None:
        self.operation = operation
        self.max_applications = max_applications
        self._applications = 0

    def can_apply(self) -> bool:
        return self._applications < self.max_applications

    def apply(self) -> None:
        factor = self.operation.noise_reduction_factor()
        self.operation._noise_std /= factor
        self._applications += 1
        logger.info(
            f"[_ReduceNoiseLevelCorrection] noise_std → {self.operation._noise_std:.3f} "
            f"(application {self._applications}/{self.max_applications})"
        )


class GaussianWithCorrectionOperation(ProtocolOperation):
    """
    Gaussian fit operation that uses the registered-check + Correction system.

    Starts with high measurement noise (SNR guaranteed to fail). Each failed
    snr_check triggers _ReduceNoiseLevelCorrection, which divides the noise std
    by noise_reduction_factor. After enough corrections the SNR passes and the
    fitted amplitude is written to the output parameter.

    Args:
        params: Instrument params (None for DUMMY platform).
        max_corrections: Maximum number of noise-reduction steps before the
            correction is exhausted and the operation fails permanently.
    """

    SNR_THRESHOLD = 2

    # Type annotations for dynamically registered parameters
    amplitude: GaussianAmplitude
    noise_reduction_factor: GaussianNoiseReductionFactor

    def __init__(self, params: Any = None, max_corrections: int = 3) -> None:
        super().__init__()

        self._register_inputs(
            center=GaussianCenter(params),
            sigma=GaussianSigma(params),
            offset=GaussianOffset(params),
        )
        self._register_outputs(amplitude=GaussianAmplitude(params))

        # CorrectionParameter: how aggressively noise is reduced each step
        self._register_correction_params(
            noise_reduction_factor=GaussianNoiseReductionFactor(params)
        )
        self.noise_reduction_factor(3.0)  # set initial value

        # Internal noise level — starts high to guarantee first attempt fails
        self._noise_std: float = 5.0

        # The stateful correction strategy
        self._noise_reduction = _ReduceNoiseLevelCorrection(
            self, max_applications=max_corrections
        )

        # Register the check → correction mapping
        self._register_check(
            name="snr_check",
            check_func=self._check_snr,
            correction=self._noise_reduction,
        )

        self.independents = {"x_values": []}
        self.dependents = {"y_values": []}
        self.fit_result: FitResult | None = None
        self.snr: float | None = None

    # ------------------------------------------------------------------ checks

    def _check_snr(self) -> CheckResult:
        snr = self.snr if self.snr is not None else 0.0
        return CheckResult(
            name="snr_check",
            passed=snr >= self.SNR_THRESHOLD,
            description=f"SNR={snr:.3f}, threshold={self.SNR_THRESHOLD}",
        )

    # ------------------------------------------------------- platform-specific

    def _measure_dummy(self) -> Path:
        true_amplitude = 10.0
        true_center = 0.5
        true_sigma = 2.0
        noise_std = self._noise_std

        x_values = np.linspace(-10, 10, 100)

        @recording(independent("x"), dependent("y"))
        def measure_gaussian(x_val: float) -> tuple[float, float]:
            y_clean = true_amplitude * np.exp(
                -((x_val - true_center) ** 2) / (2 * true_sigma**2)
            )
            return x_val, y_clean + np.random.normal(0, noise_std)

        loc, _ = run_and_save_sweep(
            Sweep(x_values, measure_gaussian), "data", self.name
        )
        return Path(loc)

    def _load_data_dummy(self) -> None:
        assert self.data_loc is not None
        path = self.data_loc / "data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)
        self.independents["x_values"] = data["x"]["values"]
        self.dependents["y_values"] = data["y"]["values"]

    def analyze(self) -> None:
        assert self.data_loc is not None
        with DatasetAnalysis(self.data_loc, self.name) as ds:
            x = np.asarray(self.independents["x_values"])
            y = np.asarray(self.dependents["y_values"])

            fit = Gaussian(x, y)
            self.fit_result = cast(FitResult, fit.run())
            fit_curve = self.fit_result.eval()
            residuals = y - fit_curve

            amplitude = self.fit_result.params["A"].value
            noise = np.std(residuals)
            self.snr = float(np.abs(amplitude / (4 * noise)))

            fig, ax = plt.subplots()
            ax.set_title(f"Gaussian fit (noise_std={self._noise_std:.2f})")
            ax.plot(x, y, "o", markersize=3, label="data")
            ax.plot(x, fit_curve, "-", linewidth=2, label="fit")
            ax.legend()

            ds.add(fit_curve=fit_curve, fit_result=self.fit_result, snr=self.snr)
            ds.add_figure(self.name, fig=fig)
            self.figure_paths.append(
                ds._new_file_path(ds.savefolders[1], self.name, suffix="png")
            )

    # ----------------------------------------------------------------- correct

    def correct(self, result: EvaluateResult) -> EvaluateResult:
        """
        On SUCCESS: write the fitted amplitude to the output parameter.
        On RETRY: the base class routes to _ReduceNoiseLevelCorrection
            (which divides self._noise_std by noise_reduction_factor).
            If the correction is exhausted, the base class escalates to FAILURE.
        """
        # Base handles: check table in report, correction routing, exhaustion
        result = super().correct(result)

        if result.status == OperationStatus.SUCCESS:
            assert self.fit_result is not None
            old = self.amplitude()
            new = float(self.fit_result.params["A"].value)
            logger.info(f"Updating {self.amplitude.name}: {old} → {new:.3f}")
            self.amplitude(new)
            self.improvements = [ParamImprovement(old, new, self.amplitude)]
            self.report_output.append(
                f"Fit **SUCCESSFUL** (SNR={self.snr:.3f}). "
                f"{self.amplitude.name}: {old} → {new:.3f}\n"
            )
        else:
            snr_str = f"{self.snr:.3f}" if self.snr is not None else "N/A"
            self.report_output.append(
                f"Fit **UNSUCCESSFUL** (SNR={snr_str}). "
                f"noise_std={self._noise_std:.3f}\n"
            )

        return result
