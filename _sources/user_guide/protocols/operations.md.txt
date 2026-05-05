# Operations

An **operation** is a single measurement step inside a protocol — a
resonator spectroscopy, a Rabi calibration, a T1 fit. Every operation
follows the same five-step lifecycle on every attempt and shares the same
hooks for declaring inputs and outputs, assessing results, and reacting to
failure. Most of writing a custom operation is filling in a handful of
methods on a subclass of
{py:class}`ProtocolOperation <labcore.protocols.base.ProtocolOperation>`.

This page assumes you have read {doc}`parameters`.

## The lifecycle of an operation

```
  ◀── platform-specific ──▶ ◀───── platform-agnostic ──────▶

  measure ──▶ load_data ──▶ analyze ──▶ evaluate ──▶ correct
     │            │            │            │            │
   write       pull and     compute      check        parameter
   hardware    normalize    (fitting,    results      writes;
   / save      shape and    statistics)  (pure        apply any
   raw data    names                     assessment)  correction
               across
               platforms
```

The split between platform-specific and platform-agnostic steps is
deliberate: `analyze`, `evaluate`, and `correct` should run identically no
matter which backend produced the data. Whatever per-platform quirks exist
in field names, units, or array shapes have to be reconciled by
`load_data` so that everything downstream sees a single canonical shape.

- **`measure`** performs the measurement (or generates fake data on
  `DUMMY`) and saves it to disk via the standard sweep + DDH5 machinery.
  Dispatches to `_measure_dummy` / `_measure_qick` / `_measure_opx`.
  Returns the path the data was written to.
- **`load_data`** reads that path back into memory and **normalizes the
  data so that downstream steps see the same shape and variable names
  regardless of platform**. Different backends can save data with
  different field names or slightly different shapes; reconciling those
  differences here is what lets `analyze` be platform-agnostic. Stores
  the result on the operation as `independents` and `dependents`
  dictionaries. Dispatches to `_load_data_dummy` / `_load_data_qick` /
  `_load_data_opx`.
- **`analyze`** is platform-agnostic. Run your fits, compute summary
  statistics, attach results to `self`. Do **not** mutate parameters here.
- **`evaluate`** is **pure assessment**. It returns named check results
  and an overall status (`SUCCESS` / `RETRY` / `FAILURE`). No side
  effects. By default this just runs every check registered with
  `_register_check`.
- **`correct`** is the **only** place an operation modifies parameters.
  On `SUCCESS` it writes any computed outputs back. On `RETRY` it applies
  a correction strategy for the failed check. On `FAILURE` it usually
  does nothing — the operation has already given up.

### Running an operation on its own

While developing a new operation it is often easier to exercise it
standalone than to wrap it in a
{py:class}`ProtocolBase <labcore.protocols.base.ProtocolBase>` subclass.
Every operation has its own `execute()` that runs the full lifecycle once
and returns the
{py:class}`EvaluateResult <labcore.protocols.base.EvaluateResult>`:

```python
from labcore.protocols import select_platform

select_platform("DUMMY")

op = MyOperation()
result = op.execute()

result.status      # SUCCESS / RETRY / FAILURE for this attempt
result.checks      # CheckResult list from evaluate()
op.report_output   # markdown strings and figure paths the operation produced
op.figure_paths    # figures attached during analyze
op.improvements    # ParamImprovements from registered success updates
```

A few things to keep in mind:

- `op.execute()` runs **one attempt**. The retry-on-`RETRY` loop lives in
  the protocol layer — to exercise corrections end-to-end you either call
  `op.execute()` again while `result.status == OperationStatus.RETRY`, or
  wrap the operation in a small one-operation protocol like the runnable
  example at the bottom of this page.
- The HTML report is **not** assembled — that happens only inside
  `ProtocolBase.execute()`. For development you typically just inspect
  `result.status` and `op.report_output` directly.
- {py:func}`select_platform <labcore.protocols.select_platform>` still
  has to be called first, exactly as it does for a protocol.

## Registering inputs, outputs, and platform code

Operations declare their inputs and outputs with three registration calls
inside `__init__`:

```python
self._register_inputs(
    center=GaussianCenter(params),
    sigma=GaussianSigma(params),
    offset=GaussianOffset(params),
)
self._register_outputs(amplitude=GaussianAmplitude(params))
self._register_correction_params(
    noise_reduction_factor=GaussianNoiseReductionFactor(params),
)
```

Each call does two things: it stores the parameter in a dictionary
(`input_params`, `output_params`, `correction_params`) and exposes it as
an attribute on the operation. After the calls above, `self.center()`,
`self.amplitude()`, and `self.noise_reduction_factor()` all work. Inputs
get verified before the protocol runs; outputs are written by `correct()`
on success; correction parameters skip the hardware verification check.

Platform-specific work — measurement and data loading — is split exactly
the way parameter getters and setters are:

```python
def _measure_dummy(self) -> Path:
    # generate fake data and run a sweep into a DDH5 file
    ...

def _measure_qick(self) -> Path:
    # write QICK pulse sequence, run, save
    ...

def _load_data_dummy(self) -> None:
    data = datadict_from_hdf5(self.data_loc / "data.ddh5")
    self.independents["x_values"] = data["x"]["values"]
    self.dependents["y_values"]   = data["y"]["values"]
```

The base class's `measure()` and `load_data()` dispatch to the right
method based on the platform selected with
{py:func}`select_platform <labcore.protocols.select_platform>`. You only
implement the platforms you actually run on; the others raise
`NotImplementedError` if invoked.

:::{note}
The leading underscore on methods like `_register_inputs`,
`_register_check`, `_measure_dummy`, and `_load_data_dummy` is the Python
convention for *"internal — don't call from outside the class."* It is a
signal to whoever is **using** an operation: instantiate it, hand it to a
protocol, and let the framework call these for you. Whoever is **writing**
an operation absolutely does use them — in `__init__` and in overrides.
The same convention applies everywhere on this page (`_register_outputs`,
`_register_correction_params`, `_register_check`,
`_register_success_update`, `_measure_*`, `_load_data_*`, …).
:::

## Correcting itself

### Checks: assessing the result

A **check** is a pure function that returns a
{py:class}`CheckResult <labcore.protocols.base.CheckResult>` — a name, a
boolean `passed`, and a one-line description that ends up in the report:

```python
def _check_snr(self) -> CheckResult:
    return CheckResult(
        name="snr",
        passed=self.snr >= self.SNR_THRESHOLD,
        description=f"SNR={self.snr:.2f}, threshold={self.SNR_THRESHOLD}",
    )
```

Register the check inside `__init__`:

```python
self._register_check(
    name="snr",
    check_func=self._check_snr,
    correction=self._noise_reduction,
)
```

The `correction` argument is the strategy to apply when this specific check
fails — covered next. Pass `None` if there is no correction (the operation
fails immediately when this check fails) or a list to declare a fallback
chain.

The default {py:meth}`evaluate <labcore.protocols.base.ProtocolOperation.evaluate>`
runs every registered check and returns `SUCCESS` if all pass, `RETRY` if
any fail. You only need to override `evaluate` for non-trivial logic that
cannot be expressed as a simple AND of independent checks.

### Corrections: doing something between retries

A **correction** object represents a strategy applied between retries when a specific
check fails. It is a subclass of
{py:class}`Correction <labcore.protocols.base.Correction>`:

```python
from labcore.protocols import Correction


class _ReduceNoiseLevelCorrection(Correction):
    name = "reduce_noise_level"
    description = "Divide measurement noise std by the noise_reduction_factor parameter"

    def __init__(self, operation, max_applications: int = 3):
        self.operation = operation
        self.max_applications = max_applications
        self._applications = 0

    def can_apply(self) -> bool:
        return self._applications < self.max_applications

    def apply(self) -> None:
        factor = self.operation.noise_reduction_factor()
        self.operation._noise_std /= factor
        self._applications += 1
```

A correction has four pieces:

- **Class-level metadata** — `name`, and `description`.
  All three end up in the protocol's report. `name` and `description`
  identify the strategy.
- **`__init__`** — usually takes a reference to the operation (so the
  correction can read or write its parameters), any configuration values
  it needs (a maximum number of applications, a list of frequency windows
  to scan, etc.), and any internal state used to track progress (a
  counter, an index, …).
- **`can_apply() -> bool`** — defines the **fail state** for the
  correction strategy. This is the mechanism that guarantees an operation
  does not retry forever: when `can_apply()` returns `False`, the default
  `correct()` escalates the operation to `FAILURE` and the protocol moves
  on (or stops). Every correction **must** have a meaningful exit
  condition encoded here — a counter, an end-of-list check, an
  out-of-range guard, anything that bounds the work. A correction that
  always returns `True` will keep an operation retrying until the
  protocol's hard ceiling on attempts (`DEFAULT_MAX_ATTEMPTS = 100`)
  finally stops it, which is a backstop, not a design.
- **`apply() -> None`** — performs the correction. Called between
  attempts, before the next `measure` runs. This is where the actual
  mutation happens — write a hardware parameter, advance an internal
  pointer, increase an averaging count, etc.

A subtle but important constraint: **the correction is one instance per
operation**, created in the operation's `__init__` and reused across every
retry. That is what lets stateful strategies work — `_applications` in
the example above counts across attempts. If a fresh correction were
built per retry, the counter would always be zero and `can_apply()` could
never return `False`.

The mapping between a check and its correction is set up at registration:

```python
self._noise_reduction = _ReduceNoiseLevelCorrection(self, max_applications=3)
self._register_check("snr", self._check_snr, correction=self._noise_reduction)
```

#### Fallback chains

`correction` accepts a list. The default `correct()` walks the list in
order and uses the first one whose `can_apply()` returns `True`. This is
how to express "first try a frequency-window scan; if that runs out, fall
back to a wide sweep":

```python
self._register_check(
    "peak_exists",
    self._check_peak,
    correction=[self._frequency_sweep, self._wide_sweep_fallback],
)
```

If every correction in the chain reports exhausted, the operation moves to
`FAILURE`.

## Writing back on success

Most operations need to write a fitted output back to a parameter when the
checks all pass. Register a *success update* in `__init__`:

```python
self._register_success_update(
    param=self.amplitude,
    value_func=lambda: self.fit_result.params["A"].value,
)
```

`value_func` is called lazily — at `correct()` time — so it can safely
reference attributes that were only set during `analyze` (like
`self.fit_result`). On every successful run the default `correct()` calls
each registered `value_func`, writes the result to the matching parameter,
records a {py:class}`ParamImprovement <labcore.protocols.base.ParamImprovement>`,
and appends a "*old → new*" line to the report. Multiple success updates
are applied in registration order.

If your only success-time work is writing a value back, that is all you
need. You do not have to override `correct()` at all.

### When to override `correct()`

Override `correct()` when you want to do something the registration API
cannot express — usually custom report messages or work that depends on
cross-check state. **Always call `super().correct(result)` first** so the
default check table, correction routing, and registered success updates
still run:

```python
def correct(self, result: EvaluateResult) -> EvaluateResult:
    result = super().correct(result)
    if result.status == OperationStatus.SUCCESS:
        self.report_output.append(
            f"Fit **SUCCESSFUL** (SNR={self.snr:.3f}). "
            f"{self.amplitude.name}: {old} → {new:.3f}\n"
        )
    return result
```

The base implementation also escalates `RETRY` to `FAILURE` when a
correction is exhausted, so the returned `result.status` may differ from
the input status — always inspect the returned value, not the original.

## Adding to the report from an operation

Each operation accumulates a list of report fragments in
`self.report_output`. The protocol's final HTML report concatenates these
in order, embedding figure paths as base64 images.

You can append two kinds of items:

- **Markdown strings**, formatted with backticks, bold, lists, and so on.
  These are rendered as-is.
- **`pathlib.Path` objects** pointing at image files (typically the
  `figure_paths` accumulated during `analyze`). These are read and
  embedded as data URIs so the final report HTML stands on its own.

Most of the time you will not have to touch this directly:

- The default `correct()` already appends a check-results table on every
  attempt and a parameter-improvement line for each registered success
  update.
- Whatever figure paths you append to `self.figure_paths` during
  `analyze` get attached to the report by the default check-table block.

You only need to write to `self.report_output` for messages the framework
does not produce on its own — for example, a one-line summary of the SNR
result tailored to your operation. The pattern in
`GaussianWithCorrectionOperation.correct()` (linked at the bottom)
is the simple case.

For a richer real-world example, see `T1Operation.correct()` in
[`CQEDToolbox/.../single_qubit/t1.py`](https://github.com/toolsforexperiments/CQEDToolbox/blob/main/src/cqedtoolbox/protocols/operations/single_qubit/t1.py#L421-L448).
It builds a full per-attempt section: a Markdown header with the data
path and SNR threshold, then one sub-section per fit component
(real / imaginary / magnitude) with the corresponding figure embedded
inline and the lmfit fit report dumped in a code block — all written by
`append`-ing strings and `Path`s to `self.report_output` before calling
`super().correct(result)` to attach the check table.

## Putting it all together

Here is a complete, runnable operation that uses every concept introduced
above — a registered output, a registered check, a registered success
update, platform-specific `measure` and `load_data`, and a
platform-agnostic `analyze`. Copy it into a script, run it, and the
protocol will execute end-to-end on the `DUMMY` platform:

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from labcore.analysis import DatasetAnalysis
from labcore.analysis.fitfuncs.generic import Gaussian
from labcore.data.datadict_storage import datadict_from_hdf5
from labcore.measurement.record import dependent, independent, recording
from labcore.measurement.storage import run_and_save_sweep
from labcore.measurement.sweep import Sweep
from labcore.protocols import (
    BranchBase, CheckResult, ProtocolBase, ProtocolOperation, select_platform,
)
from labcore.testing.protocol_dummy.parameters import GaussianAmplitude

plt.switch_backend("agg")


class MinimalGaussianFit(ProtocolOperation):
    SNR_THRESHOLD = 2.0

    def __init__(self, params=None):
        super().__init__()
        self.amplitude: GaussianAmplitude
        self._register_outputs(amplitude=GaussianAmplitude(params))

        self._register_check("snr", self._check_snr, correction=None)
        self._register_success_update(
            param=self.amplitude,
            value_func=lambda: self.fit_result.params["A"].value,
        )

        self.fit_result = None
        self.snr = None

    def _measure_dummy(self) -> Path:
        x = np.linspace(-10, 10, 100)

        @recording(independent("x"), dependent("y"))
        def measure(xv):
            y_clean = 10.0 * np.exp(-((xv - 0.5) ** 2) / 8.0)
            return xv, y_clean + np.random.normal(0, 0.3)

        loc, _ = run_and_save_sweep(Sweep(x, measure), "data", self.name)
        return Path(loc)

    def _load_data_dummy(self) -> None:
        data = datadict_from_hdf5(self.data_loc / "data.ddh5")
        self.independents["x_values"] = data["x"]["values"]
        self.dependents["y_values"]   = data["y"]["values"]

    def analyze(self) -> None:
        with DatasetAnalysis(self.data_loc, self.name) as ds:
            x = np.asarray(self.independents["x_values"])
            y = np.asarray(self.dependents["y_values"])
            self.fit_result = Gaussian(x, y).run()
            residuals = y - self.fit_result.eval()
            amp = self.fit_result.params["A"].value
            self.snr = float(np.abs(amp / (4 * np.std(residuals))))
            ds.add(snr=self.snr)

    def _check_snr(self) -> CheckResult:
        return CheckResult(
            name="snr",
            passed=self.snr >= self.SNR_THRESHOLD,
            description=f"SNR={self.snr:.2f}, threshold={self.SNR_THRESHOLD}",
        )


class MinimalProtocol(ProtocolBase):
    def __init__(self):
        super().__init__()
        self.root_branch = BranchBase("minimal")
        self.root_branch.extend([MinimalGaussianFit()])


select_platform("DUMMY")
MinimalProtocol().execute()
```

Two things to notice:

- `evaluate` and `correct` are not overridden. The base class runs every
  registered check, marks the operation `RETRY` if any fail, and on
  `SUCCESS` calls each registered `value_func` and writes the result to
  the corresponding parameter — exactly what we want for an operation
  this simple.
- No correction is registered, so any failed check immediately fails the
  operation. The next step up is a stateful correction strategy.

For an operation that adds corrections and overrides `correct()` for a
tailored report, see
{py:class}`GaussianWithCorrectionOperation <labcore.testing.protocol_dummy.gaussian_with_correction.GaussianWithCorrectionOperation>`
— full source at
[`src/labcore/testing/protocol_dummy/gaussian_with_correction.py`](https://github.com/toolsforexperiments/labcore/blob/main/src/labcore/testing/protocol_dummy/gaussian_with_correction.py).
That file maps onto the sections of this page like so:

| Section above | Where it appears |
|---|---|
| Registering inputs / outputs / correction params | top of `__init__` |
| Registering a check + correction | `_register_check` call in `__init__` |
| Correction subclass | `_ReduceNoiseLevelCorrection` |
| Platform code | `_measure_dummy`, `_load_data_dummy` |
| Analyze | `analyze()` |
| Override of `correct()` | bottom of the class |

## Where to read next

{doc}`protocols` — wrapping operations into a
{py:class}`ProtocolBase <labcore.protocols.base.ProtocolBase>` and running
them.
