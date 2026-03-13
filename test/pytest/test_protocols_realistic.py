"""
Realistic protocol tests using the dummy operations in labcore.protocols.dummy.

These tests exercise real curve fitting, SNR-based success/retry/failure logic,
and parameter flow between operations — as opposed to the structural unit tests in
test_protocols.py which use stub operations.

All tests that run _measure_dummy (which calls run_and_save_sweep with a relative
"data" path) use monkeypatch.chdir(tmp_path) so that HDF5 files land under the
pytest-supplied temporary directory.
"""

from __future__ import annotations

import numpy as np
import pytest

import labcore.protocols.base as proto_base
from labcore.protocols.base import (
    BranchBase,
    ProtocolBase,
    PlatformTypes,
)
from labcore.measurement.sweep import Sweep
from labcore.measurement.record import independent, dependent, recording
from labcore.measurement.storage import run_and_save_sweep

from labcore.testing.protocol_dummy.gaussian import GaussianOperation
from labcore.testing.protocol_dummy import ExponentialOperation
from labcore.testing.protocol_dummy import ExponentialDecayOperation
from labcore.testing.protocol_dummy import CosineOperation
from labcore.testing.protocol_dummy import LinearOperation
import labcore.testing.protocol_dummy.dummy_protocol as _dp_module
from labcore.testing.protocol_dummy import DummyProtocol, DummySuperOperation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def dummy_platform():
    proto_base.PLATFORMTYPE = PlatformTypes.DUMMY
    yield
    proto_base.PLATFORMTYPE = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_protocol(ops, report_path):
    """Minimal ProtocolBase with all ops in a single root branch."""

    class _Proto(ProtocolBase):
        def __init__(self):
            super().__init__(report_path=report_path)
            branch = BranchBase("root")
            for op in ops:
                branch.append(op)
            self.root_branch = branch

    return _Proto()


# ---------------------------------------------------------------------------
# 1. GaussianProtocol — SNR-based retry until total_attempts_made == 3
# ---------------------------------------------------------------------------


class TestGaussianFitWithRetry:
    def test_retries_3_times_and_succeeds(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        op = GaussianOperation()
        proto = make_protocol([op], report_path=tmp_path)
        proto.execute()

        assert proto.success is True
        assert op.total_attempts_made == 3

        # amplitude parameter should have been updated from its initial 0.0
        fitted_amplitude = op.amplitude()
        assert fitted_amplitude is not None
        assert fitted_amplitude != 0.0

        # improvements recorded
        assert len(op.improvements) == 1

        # HTML report exists
        report_dir = tmp_path / f"{proto.name}_report"
        html_file = report_dir / f"{proto.name}_report.html"
        assert html_file.exists()


# ---------------------------------------------------------------------------
# 2. ExponentialProtocol — single success, parameter updated
# ---------------------------------------------------------------------------


class TestExponentialFitSuccess:
    def test_succeeds_and_updates_parameter(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        op = ExponentialOperation()
        proto = make_protocol([op], report_path=tmp_path)
        proto.execute()

        assert proto.success is True

        # 'a' parameter was updated from initial 0.0
        fitted_a = op.a()
        assert fitted_a is not None
        assert fitted_a != 0.0

        assert len(op.improvements) == 1


# ---------------------------------------------------------------------------
# 3. ExponentialDecayProtocol — also exercises the fit.run() bug fix
# ---------------------------------------------------------------------------


class TestExponentialDecayFitSuccess:
    def test_succeeds_and_updates_amplitude(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        op = ExponentialDecayOperation()
        proto = make_protocol([op], report_path=tmp_path)
        proto.execute()

        assert proto.success is True

        fitted_amplitude = op.amplitude()
        assert fitted_amplitude is not None
        assert fitted_amplitude != 0.0

        assert len(op.improvements) == 1


# ---------------------------------------------------------------------------
# 4. LinearProtocol — single success, slope updated
# ---------------------------------------------------------------------------


class TestLinearFitSuccess:
    def test_succeeds_and_updates_slope(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        op = LinearOperation()
        proto = make_protocol([op], report_path=tmp_path)
        proto.execute()

        assert proto.success is True

        fitted_slope = op.slope()
        assert fitted_slope is not None
        assert fitted_slope != 0.0

        assert len(op.improvements) == 1


# ---------------------------------------------------------------------------
# 5. Sequential parameter flow: CosineProtocol then LinearProtocol
# ---------------------------------------------------------------------------


class TestCosineAndLinearParameterFlow:
    def test_both_succeed_and_params_updated(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        cosine_op = CosineOperation()
        linear_op = LinearOperation()
        proto = make_protocol([cosine_op, linear_op], report_path=tmp_path)
        proto.execute()

        assert proto.success is True

        # Both output parameters updated from initial 0.0
        assert cosine_op.amplitude() != 0.0
        assert linear_op.slope() != 0.0

        # Both recorded improvements
        assert len(cosine_op.improvements) == 1
        assert len(linear_op.improvements) == 1


# ---------------------------------------------------------------------------
# 6. DummyProtocol — DummySuperOperation retries 3× (accessed via root branch)
# ---------------------------------------------------------------------------


class TestDummySuperOperationRetries:
    def test_retries_3_times_and_succeeds(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(_dp_module, "USE_BRANCH_A", True)

        proto = DummyProtocol(report_path=tmp_path)
        proto.execute()

        # DummySuperOperation is the second item in the main branch
        super_op = proto.root_branch.items[1]
        assert isinstance(super_op, DummySuperOperation)
        assert proto.success is True
        assert super_op.total_attempts_made == 3


# ---------------------------------------------------------------------------
# 7. DummyProtocol — Condition routes to BranchA or BranchB via USE_BRANCH_A
# ---------------------------------------------------------------------------


class TestConditionRoutingInFullProtocol:
    def test_branch_a_taken_when_flag_true(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(_dp_module, "USE_BRANCH_A", True)

        proto = DummyProtocol(report_path=tmp_path)
        proto.execute()

        assert proto.success is True
        # Condition is the third item in the main branch
        branch_condition = proto.root_branch.items[2]
        assert branch_condition.taken_branch.name == "BranchA"

    def test_branch_b_taken_when_flag_false(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(_dp_module, "USE_BRANCH_A", False)
        monkeypatch.setattr(_dp_module, "USE_BRANCH_C", True)

        proto = DummyProtocol(report_path=tmp_path)
        proto.execute()

        assert proto.success is True
        branch_condition = proto.root_branch.items[2]
        assert branch_condition.taken_branch.name == "BranchB"


# ---------------------------------------------------------------------------
# 8. Failure on bad data — nearly pure noise → SNR < threshold → FAILURE
# ---------------------------------------------------------------------------


class _NoisyGaussian(GaussianOperation):
    """Override _measure_dummy to produce nearly pure noise (no signal)."""

    def _measure_dummy(self):
        x_values = np.linspace(-10, 10, 100)

        @recording(independent("x"), dependent("y"))
        def measure(x_val):
            # Tiny signal swamped by large noise → SNR << threshold
            return x_val, np.random.normal(0, 50.0)

        sweep = Sweep(x_values, measure)
        loc, _ = run_and_save_sweep(sweep, "data", self.name)
        return loc


class TestFailureOnBadData:
    def test_noisy_data_causes_failure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        np.random.seed(42)  # deterministic noise

        op = _NoisyGaussian()
        op.max_attempts = 1  # don't retry — fail fast
        proto = make_protocol([op], report_path=tmp_path)
        proto.execute()

        assert proto.success is False
