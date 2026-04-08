import pytest

from labcore.measurement.record import (
    DataType,
    FunctionToRecords,
    IteratorToRecords,
    combine_data_specs,
    dep,
    dependent,
    independent,
    make_data_spec,
    produces_record,
    record_as,
    recording,
)


def test_independent_creates_correct_dataspec():
    spec = independent("x", unit="m")
    assert spec.name == "x"
    assert spec.depends_on is None
    assert spec.unit == "m"
    assert spec.type == DataType.scalar


def test_dependent_creates_correct_dataspec():
    spec = dep("y", depends_on=["x"], unit="V")
    assert spec.name == "y"
    assert spec.depends_on == ["x"]
    assert spec.unit == "V"
    assert spec.type == DataType.scalar


def test_dependent_raises_when_depends_on_is_none():
    with pytest.raises(TypeError):
        dependent("y", depends_on=None)


def test_make_data_spec_from_string():
    # A bare string creates a dependent with no explicit axes
    spec = make_data_spec("z")
    assert spec.name == "z"
    assert spec.depends_on == []


def test_make_data_spec_from_tuple():
    spec = make_data_spec(("z", ["x", "y"], "scalar", "Hz"))
    assert spec.name == "z"
    assert spec.depends_on == ["x", "y"]
    assert spec.unit == "Hz"


def test_make_data_spec_from_dict():
    spec = make_data_spec({"name": "z", "depends_on": ["x"], "unit": "A"})
    assert spec.name == "z"
    assert spec.depends_on == ["x"]
    assert spec.unit == "A"


def test_make_data_spec_from_dataspec():
    original = independent("x", unit="s")
    spec = make_data_spec(original)
    assert spec is original


def test_make_data_spec_raises_on_invalid_type():
    with pytest.raises(TypeError):
        make_data_spec(42)


def test_record_as_with_function_returns_function_to_records():
    wrapped = record_as(lambda x: x * 2, dep("y", ["x"]))
    assert isinstance(wrapped, FunctionToRecords)


def test_function_to_records_call_returns_correct_dict():
    wrapped = record_as(lambda x: x * 2, dep("y", ["x"]))
    result = wrapped(3)
    assert result == {"y": 6}


def test_record_as_with_iterable_returns_iterator_to_records():
    wrapped = record_as(range(3), independent("x"))
    assert isinstance(wrapped, IteratorToRecords)


def test_iterator_to_records_yields_correct_dicts():
    wrapped = record_as(range(3), independent("x"))
    records = list(wrapped)
    assert records == [{"x": 0}, {"x": 1}, {"x": 2}]


def test_produces_record_true_for_wrapped():
    wrapped = record_as(lambda x: x, "y")
    assert produces_record(wrapped) is True


def test_produces_record_false_for_plain_function():
    assert produces_record(lambda x: x) is False


def test_produces_record_false_for_plain_iterable():
    assert produces_record(range(5)) is False


def test_function_to_records_using_prefills_args():
    wrapped = record_as(lambda x, offset: x + offset, dep("y", ["x"]))
    bound = wrapped.using(offset=10)
    # original is not mutated
    assert wrapped._kwargs == {}
    # bound version uses the pre-filled kwarg
    result = bound(5)
    assert result == {"y": 15}


def test_dataspec_repr_without_dependencies():
    spec = independent("x")
    assert repr(spec) == "x"


def test_dataspec_repr_with_dependencies():
    spec = dep("y", depends_on=["x", "z"])
    assert repr(spec) == "y(x, z)"


def test_recording_decorator_wraps_function():
    @recording(dep("y", ["x"]))
    def measure(x):
        return x * 3

    assert isinstance(measure, FunctionToRecords)
    result = measure(4)
    assert result == {"y": 12}


def test_recording_decorator_with_multiple_dataspecs():
    @recording(independent("x"), independent("y"), independent("z"))
    def measure(t):
        return t, t * 2, t * 3

    assert isinstance(measure, FunctionToRecords)
    result = measure(5)
    assert result == {"x": 5, "y": 10, "z": 15}


def test_combine_data_specs_removes_duplicates():
    x = independent("x")
    y = dep("y", ["x"])
    x_dup = independent("x", unit="m")  # same name, different unit
    result = combine_data_specs(x, y, x_dup)
    assert len(result) == 2
    assert result[0].name == "x"
    assert result[1].name == "y"
    # first occurrence wins
    assert result[0].unit == ""
