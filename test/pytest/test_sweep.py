
from labcore.measurement.record import record_as, recording, independent, dep
from labcore.measurement.sweep import (
    Sweep,
    sweep_parameter,
    once,
    as_pointer,
)


def test_sweep_parameter_iterates_correct_number_of_steps():
    sweep = sweep_parameter("x", range(5))
    records = list(sweep)
    assert len(records) == 5
    assert records == [{"x": 0}, {"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}]


def test_sweep_parameter_record_contains_pointer_and_action_values():
    sweep = sweep_parameter("x", range(3), record_as(lambda x: x * 2, dep("y", ["x"])))
    records = list(sweep)
    assert records == [{"x": 0, "y": 0}, {"x": 1, "y": 2}, {"x": 2, "y": 4}]


# --- Direct Sweep construction ---


def test_sweep_direct_construction_with_annotated_pointer():
    # Sweep(pointer) where pointer is a record_as-wrapped iterable
    sweep = Sweep(record_as(range(3), independent("x")))
    records = list(sweep)
    assert records == [{"x": 0}, {"x": 1}, {"x": 2}]


def test_sweep_direct_construction_with_pointer_and_action():
    # Sweep(pointer, action) — pointer and action both annotated
    sweep = Sweep(
        record_as(range(3), independent("x")),
        record_as(lambda x: x**2, dep("y", ["x"])),
    )
    records = list(sweep)
    assert records == [{"x": 0, "y": 0}, {"x": 1, "y": 1}, {"x": 2, "y": 4}]


def test_sweep_direct_construction_with_multiple_actions():
    # Multiple actions are all called and merged into each record
    sweep = Sweep(
        record_as(range(3), independent("x")),
        record_as(lambda x: x * 2, dep("y", ["x"])),
        record_as(lambda x: x * 3, dep("z", ["x"])),
    )
    records = list(sweep)
    assert records == [
        {"x": 0, "y": 0, "z": 0},
        {"x": 1, "y": 2, "z": 3},
        {"x": 2, "y": 4, "z": 6},
    ]


# --- Combination operators ---


def test_append_operator_runs_sweeps_sequentially():
    # + : all of A, then all of B
    a = sweep_parameter("x", range(3))
    b = sweep_parameter("x", range(10, 13))
    records = list(a + b)
    assert records == [
        {"x": 0},
        {"x": 1},
        {"x": 2},
        {"x": 10},
        {"x": 11},
        {"x": 12},
    ]


def test_zip_operator_runs_sweeps_elementwise():
    # * : A and B advance together, stops at the shortest
    a = sweep_parameter("x", range(3))
    b = sweep_parameter("y", range(10, 14))  # 4 elements, zip stops at 3
    records = list(a * b)
    assert records == [
        {"x": 0, "y": 10},
        {"x": 1, "y": 11},
        {"x": 2, "y": 12},
    ]


def test_nest_operator_runs_inner_sweep_for_each_outer_step():
    # @ : full inner sweep for every outer step
    outer = sweep_parameter("x", range(3))
    inner = sweep_parameter("y", range(4))
    records = list(outer @ inner)
    assert records == [
        {"x": 0, "y": 0},
        {"x": 0, "y": 1},
        {"x": 0, "y": 2},
        {"x": 0, "y": 3},
        {"x": 1, "y": 0},
        {"x": 1, "y": 1},
        {"x": 1, "y": 2},
        {"x": 1, "y": 3},
        {"x": 2, "y": 0},
        {"x": 2, "y": 1},
        {"x": 2, "y": 2},
        {"x": 2, "y": 3},
    ]


def test_once_executes_action_exactly_once():
    call_count = 0

    def side_effect():
        nonlocal call_count
        call_count += 1

    sweep = once(side_effect) + sweep_parameter("x", range(5))
    list(sweep)
    assert call_count == 1


def test_as_pointer_creates_pointer_from_generator_function():
    def gen():
        yield from range(3)

    sweep = Sweep(as_pointer(gen, independent("x")))
    records = list(sweep)
    assert records == [{"x": 0}, {"x": 1}, {"x": 2}]


def test_all_three_operators_combined():
    @recording(dep("a", ["y"]))
    def compute_a(y):
        return y + 1

    @recording(dep("b", ["z"]))
    def compute_b(z):
        return z * 3

    x = Sweep(record_as(range(2), independent("x")))
    y = Sweep(record_as(range(6), independent("y")), compute_a)
    z = Sweep(record_as(range(2), independent("z")), compute_b)

    records = list(x + (y @ z))
    assert records == [
        {"a": None, "b": None, "x": 0, "y": None, "z": None},
        {"a": None, "b": None, "x": 1, "y": None, "z": None},
        {"a": 1, "b": 0, "x": None, "y": 0, "z": 0},
        {"a": 1, "b": 3, "x": None, "y": 0, "z": 1},
        {"a": 2, "b": 0, "x": None, "y": 1, "z": 0},
        {"a": 2, "b": 3, "x": None, "y": 1, "z": 1},
        {"a": 3, "b": 0, "x": None, "y": 2, "z": 0},
        {"a": 3, "b": 3, "x": None, "y": 2, "z": 1},
        {"a": 4, "b": 0, "x": None, "y": 3, "z": 0},
        {"a": 4, "b": 3, "x": None, "y": 3, "z": 1},
        {"a": 5, "b": 0, "x": None, "y": 4, "z": 0},
        {"a": 5, "b": 3, "x": None, "y": 4, "z": 1},
        {"a": 6, "b": 0, "x": None, "y": 5, "z": 0},
        {"a": 6, "b": 3, "x": None, "y": 5, "z": 1},
    ]
