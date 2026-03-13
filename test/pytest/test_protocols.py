"""
Unit tests for labcore.protocols.base

Covers: ProtocolParameterBase, OperationStatus, ProtocolOperation,
SuperOperationBase, BranchBase, Condition, ProtocolBase, and the
parameter optimization lifecycle (success/retry/failure).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

import labcore.protocols.base as proto_base
from labcore.protocols.base import (
    BranchBase,
    Condition,
    OperationStatus,
    ParamImprovement,
    PlatformTypes,
    ProtocolBase,
    ProtocolOperation,
    ProtocolParameterBase,
    SuperOperationBase,
)


# ---------------------------------------------------------------------------
# Fixture: DUMMY platform for all tests, restored after each
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def dummy_platform():
    proto_base.PLATFORMTYPE = PlatformTypes.DUMMY
    yield
    proto_base.PLATFORMTYPE = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_param(store: dict | None = None):
    """Return a concrete ProtocolParameterBase and the dict backing it."""
    if store is None:
        store = {"value": 0}

    @dataclass
    class _Param(ProtocolParameterBase):
        name: str = field(default="test_param", init=False)
        description: str = field(default="A test parameter", init=False)

        def _dummy_getter(self):
            return store["value"]

        def _dummy_setter(self, value):
            store["value"] = value

    return _Param(params=None), store


def make_simple_op(
    status: OperationStatus = OperationStatus.SUCCESS, call_log: list | None = None
):
    """Return a ProtocolOperation that records execution steps and returns status."""
    log = call_log if call_log is not None else []

    class _Op(ProtocolOperation):
        def _measure_dummy(self):
            log.append("measure")
            return Path(".")

        def _load_data_dummy(self):
            log.append("load_data")

        def analyze(self):
            log.append("analyze")

        def evaluate(self) -> OperationStatus:
            log.append("evaluate")
            return status

    return _Op(), log


def make_protocol(ops, report_path: Path = Path("")):
    """Minimal ProtocolBase with all ops in a single root branch."""

    class _Proto(ProtocolBase):
        def __init__(self):
            super().__init__(report_path=report_path)
            branch = BranchBase("root")
            for op in ops:
                branch.append(op)
            self.root_branch = branch

    return _Proto()


# ===========================================================================
# 1. ProtocolParameterBase
# ===========================================================================


class TestProtocolParameterBase:
    def test_dispatch_getter(self):
        param, _ = make_param({"value": 42})
        assert param() == 42

    def test_dispatch_setter(self):
        param, store = make_param({"value": 0})
        param(99)
        assert store["value"] == 99

    def test_post_init_inherits_global_platformtype(self):
        @dataclass
        class _P(ProtocolParameterBase):
            name: str = field(default="p", init=False)
            description: str = field(default="d", init=False)

            def _dummy_getter(self):
                return 1

            def _dummy_setter(self, v):
                pass

        p = _P(params=None, platform_type=None)
        assert p.platform_type == PlatformTypes.DUMMY

    def test_post_init_requires_params_for_non_dummy(self):
        proto_base.PLATFORMTYPE = PlatformTypes.QICK

        @dataclass
        class _P(ProtocolParameterBase):
            name: str = field(default="p", init=False)
            description: str = field(default="d", init=False)

        with pytest.raises(ValueError):
            _P(params=None)


# ===========================================================================
# 2. OperationStatus
# ===========================================================================


class TestOperationStatus:
    def test_enum_values(self):
        assert str(OperationStatus.SUCCESS) == "success"
        assert str(OperationStatus.RETRY) == "retry"
        assert str(OperationStatus.FAILURE) == "failure"


# ===========================================================================
# 3. ProtocolOperation
# ===========================================================================


class TestProtocolOperation:
    def test_register_inputs_sets_attr_and_dict(self):
        param, _ = make_param()
        op, _ = make_simple_op()
        op._register_inputs(my_param=param)
        assert op.my_param is param
        assert op.input_params["my_param"] is param

    def test_register_outputs_sets_attr_and_dict(self):
        param, _ = make_param()
        op, _ = make_simple_op()
        op._register_outputs(out=param)
        assert op.out is param
        assert op.output_params["out"] is param

    def test_execute_calls_workflow_in_order(self):
        log = []
        op, _ = make_simple_op(call_log=log)
        op.execute()
        assert log == ["measure", "load_data", "analyze", "evaluate"]

    def test_execute_increments_attempt_counters(self):
        op, _ = make_simple_op()
        op.execute()
        op.execute()
        assert op.current_attempt == 2
        assert op.total_attempts_made == 2

    def test_execute_adds_retry_header_on_second_attempt(self):
        op, _ = make_simple_op()
        op.execute()
        assert not any("ATTEMPT" in str(r) for r in op.report_output)
        op.execute()
        assert any("ATTEMPT 2" in str(r) for r in op.report_output)

    @pytest.mark.parametrize(
        "case,expected",
        [
            ("empty_independent", False),
            ("matching_shapes", True),
            ("mismatched_shapes", False),
            ("empty_dicts", True),
        ],
    )
    def test_verify_shape_cases(self, case, expected):
        op, _ = make_simple_op()
        if case == "empty_independent":
            op.independents = {"x": np.array([])}
            op.dependents = {"y": np.array([1, 2])}
        elif case == "matching_shapes":
            op.independents = {"x": np.array([1, 2, 3])}
            op.dependents = {"y": np.array([4, 5, 6])}
        elif case == "mismatched_shapes":
            op.independents = {"x": np.array([1, 2])}
            op.dependents = {"y": np.array([1, 2, 3])}
        elif case == "empty_dicts":
            op.independents = {}
            op.dependents = {}
        assert op._verify_shape() == expected


# ===========================================================================
# 4. SuperOperationBase
# ===========================================================================


class TestSuperOperationBase:
    def _make_super(self, sub_ops, evaluate_status=OperationStatus.SUCCESS):
        class _Super(SuperOperationBase):
            def evaluate(self) -> OperationStatus:
                return evaluate_status

        s = _Super()
        s.operations = sub_ops
        return s

    def test_validate_rejects_condition_in_operations(self):
        branch = BranchBase("b")
        cond = Condition(lambda: True, branch, branch)
        s = self._make_super([cond])
        with pytest.raises(ValueError, match="Condition"):
            s._validate_operations()

    def test_validate_rejects_non_operation(self):
        s = self._make_super(["not_an_op"])
        with pytest.raises(TypeError):
            s._validate_operations()

    def test_execute_aggregates_sub_op_reports(self):
        op1, _ = make_simple_op()
        op1.report_output = ["op1 result"]
        op2, _ = make_simple_op()
        op2.report_output = ["op2 result"]
        s = self._make_super([op1, op2])
        s.execute()
        combined = " ".join(str(r) for r in s.report_output)
        assert "op1 result" in combined
        assert "op2 result" in combined

    def test_execute_returns_failure_on_sub_op_exception(self):
        class _BadOp(ProtocolOperation):
            def execute(self) -> OperationStatus:
                raise RuntimeError("boom")

            def _measure_dummy(self):
                pass

            def _load_data_dummy(self):
                pass

            def analyze(self):
                pass

            def evaluate(self):
                return OperationStatus.SUCCESS

        s = self._make_super([_BadOp()])
        result = s.execute()
        assert result == OperationStatus.FAILURE

    def test_execute_returns_failure_on_sub_op_failure(self):
        op, _ = make_simple_op(status=OperationStatus.FAILURE)
        s = self._make_super([op])
        result = s.execute()
        assert result == OperationStatus.FAILURE

    def test_execute_calls_evaluate_at_end(self):
        called = []

        class _Super(SuperOperationBase):
            def evaluate(self) -> OperationStatus:
                called.append(True)
                return OperationStatus.SUCCESS

        op, _ = make_simple_op()
        s = _Super()
        s.operations = [op]
        result = s.execute()
        assert called == [True]
        assert result == OperationStatus.SUCCESS


# ===========================================================================
# 5. BranchBase
# ===========================================================================


class TestBranchBase:
    def test_append_and_extend(self):
        op1, _ = make_simple_op()
        op2, _ = make_simple_op()
        op3, _ = make_simple_op()
        branch = BranchBase("test")
        branch.append(op1).extend([op2, op3])
        assert branch.items == [op1, op2, op3]

    def test_repr(self):
        op, _ = make_simple_op()
        branch = BranchBase("MyBranch")
        branch.append(op)
        r = repr(branch)
        assert "MyBranch" in r
        assert "1" in r


# ===========================================================================
# 6. Condition
# ===========================================================================


class TestCondition:
    def test_evaluate_true_branch(self):
        true_b = BranchBase("True")
        false_b = BranchBase("False")
        cond = Condition(lambda: True, true_b, false_b, name="TestCond")
        result = cond.evaluate()
        assert result is true_b
        assert cond.condition_result is True
        assert cond.taken_branch is true_b

    def test_evaluate_false_branch(self):
        true_b = BranchBase("True")
        false_b = BranchBase("False")
        cond = Condition(lambda: False, true_b, false_b, name="TestCond")
        result = cond.evaluate()
        assert result is false_b
        assert cond.condition_result is False
        assert cond.taken_branch is false_b

    def test_evaluate_appends_to_report(self):
        branch = BranchBase("b")
        cond = Condition(lambda: True, branch, branch, name="X")
        cond.evaluate()
        assert len(cond.report_output) > 0


# ===========================================================================
# 7. ProtocolBase
# ===========================================================================


class TestProtocolBase:
    def test_raises_if_platformtype_none(self):
        proto_base.PLATFORMTYPE = None
        with pytest.raises(ValueError, match="platform"):
            make_protocol([])

    def test_verify_all_parameters_passes(self):
        op, _ = make_simple_op()
        param, _ = make_param()
        op._register_inputs(p=param)
        proto = make_protocol([op])
        assert proto.verify_all_parameters() is True

    def test_verify_all_parameters_raises_on_bad_param(self):
        @dataclass
        class _BadParam(ProtocolParameterBase):
            name: str = field(default="bad", init=False)
            description: str = field(default="d", init=False)

            def _dummy_getter(self):
                raise RuntimeError("can't reach hardware")

            def _dummy_setter(self, v):
                pass

        op, _ = make_simple_op()
        op._register_inputs(bad=_BadParam(params=None))
        proto = make_protocol([op])
        with pytest.raises(AttributeError):
            proto.verify_all_parameters()

    def test_execute_success(self, tmp_path):
        op, _ = make_simple_op(status=OperationStatus.SUCCESS)
        proto = make_protocol([op], report_path=tmp_path)
        proto.execute()
        assert proto.success is True

    def test_execute_failure_stops_protocol(self, tmp_path):
        op1, _ = make_simple_op(status=OperationStatus.FAILURE)
        op2, log2 = make_simple_op()
        proto = make_protocol([op1, op2], report_path=tmp_path)
        proto.execute()
        assert proto.success is False
        assert "evaluate" not in log2  # second op never ran

    def test_execute_generates_html_report(self, tmp_path):
        op, _ = make_simple_op(status=OperationStatus.SUCCESS)
        proto = make_protocol([op], report_path=tmp_path)
        proto.execute()
        report_dir = tmp_path / f"{proto.name}_report"
        html_file = report_dir / f"{proto.name}_report.html"
        assert html_file.exists()


# ===========================================================================
# 8. Parameter optimization lifecycle
# ===========================================================================


class TestParameterOptimizationLifecycle:
    def test_success_updates_output_parameter(self, tmp_path):
        """evaluate() updates the output param and records a ParamImprovement."""
        store = {"value": 10}

        @dataclass
        class _OutParam(ProtocolParameterBase):
            name: str = field(default="output", init=False)
            description: str = field(default="output param", init=False)

            def _dummy_getter(self):
                return store["value"]

            def _dummy_setter(self, v):
                store["value"] = v

        class _Op(ProtocolOperation):
            def __init__(self):
                super().__init__()
                self._register_outputs(result=_OutParam(params=None))

            def _measure_dummy(self):
                return Path(".")

            def _load_data_dummy(self):
                pass

            def analyze(self):
                pass

            def evaluate(self) -> OperationStatus:
                old = self.result()
                new = old + 5
                self.improvements.append(
                    ParamImprovement(old_value=old, new_value=new, param=self.result)
                )
                self.result(new)
                return OperationStatus.SUCCESS

        op = _Op()
        proto = make_protocol([op], report_path=tmp_path)
        proto.execute()

        assert proto.success is True
        assert store["value"] == 15
        assert len(op.improvements) == 1
        assert op.improvements[0].old_value == 10
        assert op.improvements[0].new_value == 15

    def test_retry_reruns_until_success(self, tmp_path):
        """RETRY on first two attempts, SUCCESS on third → protocol succeeds after 3 runs."""
        attempt = {"count": 0}

        class _Op(ProtocolOperation):
            def _measure_dummy(self):
                return Path(".")

            def _load_data_dummy(self):
                pass

            def analyze(self):
                pass

            def evaluate(self) -> OperationStatus:
                attempt["count"] += 1
                if attempt["count"] < 3:
                    return OperationStatus.RETRY
                return OperationStatus.SUCCESS

        op = _Op()
        op.max_attempts = 3
        proto = make_protocol([op], report_path=tmp_path)
        proto.execute()

        assert proto.success is True
        assert op.total_attempts_made == 3

    def test_retry_exhausted_marks_failure(self, tmp_path):
        """Operation always returns RETRY → exhausts max_attempts → protocol fails."""

        class _Op(ProtocolOperation):
            def _measure_dummy(self):
                return Path(".")

            def _load_data_dummy(self):
                pass

            def analyze(self):
                pass

            def evaluate(self) -> OperationStatus:
                return OperationStatus.RETRY

        op = _Op()
        op.max_attempts = 2
        proto = make_protocol([op], report_path=tmp_path)
        proto.execute()

        assert proto.success is False
        assert op.total_attempts_made == 2
