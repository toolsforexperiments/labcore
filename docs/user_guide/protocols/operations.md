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
   ◀── platform-specific ──▶   ◀──── platform-agnostic ────▶

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

- **`measure`** writes hardware (or generates fake data on `DUMMY`) and
  saves it to disk via the standard sweep + DDH5 machinery. Dispatches to
  `_measure_dummy` / `_measure_qick` / `_measure_opx`. Returns the path
  the data was written to.
- **`load_data`** reads that path back into memory and **normalizes the
  data so that downstream steps see the same shape and variable names
  regardless of platform**. Different backends can save data with
  different field names or slightly different shapes; reconciling those
  differences here is what lets `analyze` be platform-agnostic. Stores
  the result on the operation as `independents` and `dependents`
  dictionaries. Dispatches to `_load_data_dummy` / `_load_data_qick` /
  `_load_data_opx`.
- **`analyze`** is platform-agnostic. Run your fits, compute summary
  statistics, attach results to `self`. Do not mutate parameters here.
- **`evaluate`** is **pure assessment**. It returns named check results
  and an overall status (`SUCCESS` / `RETRY` / `FAILURE`). No side
  effects. By default this just runs every check registered with
  `_register_check`.
- **`correct`** is the **only** place an operation modifies parameters.
  On `SUCCESS` it writes any computed outputs back. On `RETRY` it applies
  a correction strategy for the failed check. On `FAILURE` it is usually a
  no-op.

This split exists because the old combined `evaluate-and-mutate` shape
made retries blunt and side-effects hard to reason about. With it, a
report can show *what* was checked and *what* was changed as two distinct
pieces of information.

## A minimal operation

The smallest useful operation has one output parameter, a measurement, an
analysis, a single check, and a success update. No corrections, no
overrides:

```python
from labcore.protocols import ProtocolOperation, CheckResult


class MinimalGaussianFit(ProtocolOperation):
    SNR_THRESHOLD = 2.0

    def __init__(self, params=None):
        super().__init__()
        self._register_outputs(amplitude=GaussianAmplitude(params))

        self._register_check("snr", self._check_snr, correction=None)
        self._register_success_update(
            param=self.amplitude,
            value_func=lambda: self.fit_result.params["A"].value,
        )

        self.fit_result = None
        self.snr = None

    def _measure_dummy(self): ...        # see "Registering platform code"
    def _load_data_dummy(self): ...

    def analyze(self):
        # ... fit, compute SNR, store on self ...
        self.fit_result = ...
        self.snr = ...

    def _check_snr(self) -> CheckResult:
        return CheckResult(
            name="snr",
            passed=self.snr >= self.SNR_THRESHOLD,
            description=f"SNR={self.snr:.2f}, threshold={self.SNR_THRESHOLD}",
        )
```

That is enough for a working operation. `evaluate` and `correct` are not
overridden — the base class runs every registered check, marks the
operation `RETRY` if any fail, and on `SUCCESS` calls each registered
`value_func` and writes the result to the corresponding parameter. The
sections below add the rest of the surface area, one piece at a time.

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

## Checks: assessing the result

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
self._register_check("snr", self._check_snr, correction=self._noise_reduction)
```

The `correction` argument is the strategy to apply when this specific check
fails — covered next. Pass `None` if there is no correction (the operation
fails immediately when this check fails) or a list to declare a fallback
chain.

The default {py:meth}`evaluate <labcore.protocols.base.ProtocolOperation.evaluate>`
runs every registered check and returns `SUCCESS` if all pass, `RETRY` if
any fail. You only need to override `evaluate` for non-trivial logic that
cannot be expressed as a simple AND of independent checks.

## Corrections: doing something between retries

A **correction** is a strategy applied between retries when a specific
check fails. It is a subclass of
{py:class}`Correction <labcore.protocols.base.Correction>`:

```python
from labcore.protocols import Correction


class _ReduceNoiseLevelCorrection(Correction):
    name = "reduce_noise_level"
    description = "Divide measurement noise std by the noise_reduction_factor parameter"
    triggered_by = "snr"

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

Three things to notice:

- **One instance per operation.** The correction is created in `__init__`
  and reused across every retry. This is what lets `_applications`
  count across attempts; if a fresh correction were built per retry, the
  counter would always be zero and `can_apply()` could never return
  `False`.
- **`can_apply` is the exhaustion gate.** When it returns `False`,
  `correct()` escalates the operation to `FAILURE` instead of retrying
  forever.
- **`triggered_by` names the check.** It is informational — used in
  reports to show which check was failing when the correction fired.

The mapping between a check and its correction is set up at registration:

```python
self._noise_reduction = _ReduceNoiseLevelCorrection(self, max_applications=3)
self._register_check("snr", self._check_snr, correction=self._noise_reduction)
```

### Fallback chains

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

## When to override `correct()`

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
`GaussianWithCorrectionOperation.correct()` (shown in the appendix below)
is representative.

## Putting it all together

The dummy package ships
{py:class}`GaussianWithCorrectionOperation <labcore.testing.protocol_dummy.gaussian_with_correction.GaussianWithCorrectionOperation>`,
which uses every feature on this page in one place: registered inputs and
outputs, a correction parameter, a stateful correction strategy with an
exhaustion counter, a registered check, and a custom override of
`correct()` for tailored report output. The full source — the simulated
measurement, the fit, the correction subclass, the operation — is in
[`src/labcore/testing/protocol_dummy/gaussian_with_correction.py`](https://github.com/toolsforexperiments/labcore/blob/main/src/labcore/testing/protocol_dummy/gaussian_with_correction.py).

The shape of that file maps onto the sections above:

| Section above | Where it appears |
|---|---|
| Registering inputs / outputs / correction params | top of `__init__` |
| Registering a check + correction | `_register_check` call in `__init__` |
| Correction subclass | `_ReduceNoiseLevelCorrection` |
| Platform code | `_measure_dummy`, `_load_data_dummy` |
| Analyze | `analyze()` |
| Override of `correct()` | bottom of the class |

## Where to read next

{doc}`building_protocols` — wrapping operations into a `ProtocolBase` and
running them.
