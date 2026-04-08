# Protocol Corrections Architecture

> **For implementors:** Read this file alongside
> `CQEDToolbox/src/cqedtoolbox/protocols/operations/single_qubit/res_spec.py`,
> which is the canonical reference implementation. The doc describes the API;
> `res_spec.py` shows a complete, real migration including `CorrectionParameter`
> subclasses with `_qick_getter`/`_qick_setter` bodies, `Correction` subclasses
> with full state tracking, and `__init__` wiring.

## Background

The protocol system (`src/labcore/protocols/base.py`) orchestrates multi-step lab
measurements. Each `ProtocolOperation` runs a fixed workflow:

```
measure() → load_data() → analyze() → evaluate() → correct()
```

Before this change, `evaluate()` did two things: assessed results **and** mutated
hardware parameters. The retry mechanism was blunt — just re-run the same operation
with the same settings.

## What Changed

### 1. Separated concerns across `evaluate()` and `correct()`

| Method | Responsibility |
|---|---|
| `evaluate()` | **Pure assessment.** Returns named check results + overall status. No side effects. |
| `correct()` | **Only place parameters are changed.** Applies found values on success, corrective actions on retry. |

`correct()` is always called inside `execute()` after `evaluate()`. Its return value
(an `EvaluateResult`) is what the protocol executor sees.

### 2. New types

#### `CheckResult`
```python
@dataclass
class CheckResult:
    name: str          # e.g. "snr_check", "peak_exists"
    passed: bool
    description: str   # e.g. "SNR=1.5, threshold=2.0"
```

#### `EvaluateResult`
```python
@dataclass
class EvaluateResult:
    status: OperationStatus          # SUCCESS / RETRY / FAILURE
    checks: list[CheckResult] = []   # named check outcomes
```
Return type for both `evaluate()` and `correct()`.

#### `Correction`
```python
class Correction:
    name: str = ""
    description: str = ""
    triggered_by: str = ""   # name of the CheckResult that triggers this

    def can_apply(self) -> bool:
        """Return False when strategy is exhausted → correct() escalates to FAILURE."""
        return True

    def apply(self) -> None:
        """Apply the correction in-place. Called before the next retry attempt."""
        raise NotImplementedError

    def report_output(self) -> str:
        """Optional. Return a human-readable description of what apply() just changed.
        Called by correct() after apply() and appended to the correction log line."""
        return ""
```

Subclass this for each corrective strategy. One **instance per operation**, created
in `__init__` and reused across retries so stateful strategies (e.g. stepping
through a frequency list) work correctly.

`report_output()` is called after `apply()` and appended inline to the correction
log entry as `| **Change:** <output>`. Implement it to surface the actual values
that changed (before → after), which makes the HTML report self-explanatory.

**Example:**
```python
class FrequencySweepCorrection(Correction):
    name = "scan_next_frequency_window"
    description = "Step through candidate frequency windows until a peak is found"
    triggered_by = "peak_exists"

    def __init__(self, freq_center_param, windows: list[float]):
        self.freq_center_param = freq_center_param
        self.windows = windows
        self._idx = 0
        self._last_change: str = ""

    def can_apply(self) -> bool:
        return self._idx < len(self.windows)

    def apply(self) -> None:
        old = self.freq_center_param()
        new = self.windows[self._idx]
        self.freq_center_param(new)
        self._idx += 1
        self._last_change = f"center: {old * 1e-9:.4f} → {new * 1e-9:.4f} GHz"

    def report_output(self) -> str:
        return self._last_change
```

#### `CorrectionParameter`
```python
class CorrectionParameter(ProtocolParameterBase):
    is_correction: ClassVar[bool] = True
    # Skips hardware params validation in __post_init__
    # Otherwise identical to ProtocolParameterBase — same callable interface,
    # same platform-specific getter/setter pattern for unit differences.
```

Used for parameters that control correction strategy (window sizes, step counts,
noise tolerances) rather than actual hardware state. Subclass exactly like
`ProtocolParameterBase`.

**Important:** `CorrectionParameter` subclasses **must** be decorated with `@dataclass`
so that `name` and `description` fields with `init=False` defaults are resolved
correctly. Without `@dataclass` the fields are not processed and the class will not
instantiate correctly.

```python
@dataclass                          # required
class MyThreshold(CorrectionParameter):
    name: str = field(default="my_threshold", init=False)
    description: str = field(default="...", init=False)

    def _qick_getter(self): return self.params.corrections.my_op.threshold()
    def _qick_setter(self, v): self.params.corrections.my_op.threshold(v)
```

---

## Registration API

Operations can use a registration-based path (covers most cases) or override
`evaluate()` / `correct()` directly for complex logic.

### Registering checks

```python
# In __init__:
self._register_check(
    name="snr_check",
    check_func=self._check_snr,
    correction=self._snr_correction,  # single Correction, or list[Correction], or None
)
self._register_check(
    name="peak_exists",
    check_func=self._check_peak,
    correction=[self._freq_correction, self._fallback_correction],  # fallback chain
)
```

The `correction` argument accepts:
- `None` — no correction; failed check → immediate FAILURE
- A single `Correction` instance — normalized to a list of one internally
- A `list[Correction]` — tried in order on each retry; first where `can_apply()` is True is used

**Default `evaluate()`** runs all registered checks:
- All pass → `EvaluateResult(SUCCESS, checks)`
- Any fail → `EvaluateResult(RETRY, checks)`

**Default `correct()`**:
- Appends a check summary table to `report_output`
- If the operation has any `figure_paths`, appends `figure_paths[-1]` to `report_output` immediately after the table (so the plot appears below the check results in the HTML report). No override needed — this is automatic.
- On RETRY: for each failed check, finds the **first** registered `Correction` where `can_apply()` is True:
  - No corrections registered → returns `EvaluateResult(FAILURE, checks)`
  - All corrections exhausted → returns `EvaluateResult(FAILURE, checks)`
  - Otherwise → calls `apply()`, then `report_output()`, and logs both to `report_output`
- On SUCCESS: applies all registered success updates (see below)
- On FAILURE: no-op

### Registering success updates

```python
# In __init__:
self._register_success_update(
    param=self.frequency,
    value_func=lambda: self.peak_freq,   # called lazily at correct() time
)
```

On SUCCESS, `correct()` calls each registered `value_func`, writes the result to `param`,
records a `ParamImprovement`, and appends a line to `report_output`. Multiple updates are
applied in registration order.

`value_func` is called lazily so it can safely reference attributes set during `analyze()`
(e.g. `self.fit_result`).

`self.improvements` is reset to `[]` at the start of each `execute()` call, so it always
reflects only the current attempt.

### Registering correction parameters

```python
# In __init__:
self._register_correction_params(
    window_size=WindowSizeParam(params),
    max_steps=MaxStepsParam(params),
)
```

Stored in `self.correction_params`. Excluded from `verify_all_parameters()` (no
hardware to check). Accessible as attributes: `self.window_size()`.

---

## Complete operation pattern

```python
class FindResonatorOperation(ProtocolOperation):
    SNR_THRESHOLD = 2.0

    def __init__(self, params=None):
        super().__init__()
        self._register_inputs(center=ResonatorCenter(params))
        self._register_outputs(frequency=ResonatorFrequency(params))

        # Correction strategies — persist across retries
        self._freq_sweep = FrequencySweepCorrection(
            freq_center_param=self.center,
            windows=[5.0e9, 5.5e9, 6.0e9, 6.5e9],
        )
        self._fallback_sweep = WideSweepCorrection(self.center)
        self._increase_avg = IncreaseAveragingCorrection(self.averages)

        # Register checks → corrections (list = fallback chain)
        self._register_check("peak_exists", self._check_peak,
                             [self._freq_sweep, self._fallback_sweep])
        self._register_check("snr_check",   self._check_snr,  self._increase_avg)

        # On success, write the found frequency automatically
        self._register_success_update(self.frequency, lambda: self.peak_freq)

        # Correction strategy parameters (platform-aware knobs)
        self._register_correction_params(
            window_size=FrequencyWindowSize(params),
        )

        self.peak_freq: float | None = None
        self.snr: float | None = None

    # --- platform-specific measurement (implement for QICK / OPX) ---
    def _measure_dummy(self) -> Path: ...
    def _load_data_dummy(self) -> None: ...

    def analyze(self) -> None:
        # detect peaks, compute SNR — no param mutations here
        ...

    # --- checks (pure assessment) ---
    def _check_peak(self) -> CheckResult:
        passed = self.peak_freq is not None
        return CheckResult("peak_exists", passed,
                           f"{'peak at ' + str(self.peak_freq) if passed else 'no peak detected'}")

    def _check_snr(self) -> CheckResult:
        snr = self.snr or 0.0
        passed = snr >= self.SNR_THRESHOLD
        return CheckResult("snr_check", passed,
                           f"SNR={snr:.2f}, threshold={self.SNR_THRESHOLD}")

    # No correct() override needed — base class handles:
    #   RETRY  → applies first applicable correction per failed check
    #   SUCCESS → writes self.frequency via _register_success_update
    #
    # Override correct() only for custom report messages or additional logic.
```

If extra reporting is needed on SUCCESS, override `correct()` and call `super()` first:

```python
def correct(self, result: EvaluateResult) -> EvaluateResult:
    result = super().correct(result)   # check table + corrections + success updates
    if result.status == OperationStatus.SUCCESS:
        self.report_output.append(
            f"Resonator found at {self.peak_freq:.3e} Hz (SNR={self.snr:.2f})\n"
        )
    return result
```

### Custom report layouts with multiple figures

The default `correct()` auto-appends `figure_paths[-1]` immediately after the check table.
This works for simple operations with one plot. For operations that produce several named
figures (e.g. colorbar, per-trace plots, summary plot) and need a specific report order,
pop the named figures out of `figure_paths` **before** calling `super()`, then clear the
list so the auto-append has nothing to fire on.

```python
def correct(self, result: EvaluateResult) -> EvaluateResult:
    # Pull named figures out before super() can auto-append the last one.
    # figure_paths order after analyze(): [0]=colorbar, [1..N-1]=traces, [-2]=snr_plot, [-1]=summary
    colorbar      = self.figure_paths.pop(0)  if len(self.figure_paths) >= 3 else None
    summary_plot  = self.figure_paths.pop(-1) if self.figure_paths else None
    snr_plot      = self.figure_paths.pop(-1) if self.figure_paths else None
    trace_figures = list(self.figure_paths)
    self.figure_paths.clear()          # prevent auto-append

    # Build header and main plots first
    self.report_output.extend([
        "## My Operation\n...\n",
        "**Colorbar:**\n",   colorbar,
        "**Summary:**\n",    summary_plot,
        "**SNR plot:**\n",   snr_plot,
    ])

    result = super().correct(result)   # adds check table; no auto-figure since list is empty

    if result.status == OperationStatus.SUCCESS:
        self.report_output.append("### Per-trace results\n")
        for fig in trace_figures:
            self.report_output.append(fig)

    return result
```

Key points:
- `report_output` is a plain `list`. Append strings (markdown) or `Path` objects (figures) in any order.
- Pop figures in reverse order from the end to avoid index shifting.
- Call `super().correct()` after building the preamble so the check table appears below the plots.

---

## `SuperOperationBase` changes

- Sub-operations call their own `correct()` internally (inside `execute()`).
- `SuperOperationBase.execute()` now returns `EvaluateResult`.
- `SuperOperationBase` has its own `correct()` — default is a no-op. Override for
  super-level parameter changes.

---

## Exported symbols (`protocols/__init__.py`)

New exports added:
- `CheckResult`
- `Correction`
- `CorrectionParameter`
- `EvaluateResult`

---

## Dummy package additions

| File | Addition |
|---|---|
| `parameters.py` | `_DummyCorrectionParameterBase(CorrectionParameter)` — in-memory correction params |
| All 6 operation files | `evaluate()` returns `EvaluateResult`; parameter updates moved to `correct()` |
| `dummy_protocol.py` | `DummySuperOperation.evaluate()` returns `EvaluateResult` |

---

## `_DummyCorrectionParameterBase` pattern

```python
@dataclass
class _DummyCorrectionParameterBase(CorrectionParameter):
    def __post_init__(self):
        super().__post_init__()
        self._value: float = 0.0

    def _dummy_getter(self) -> float:
        return self._value

    def _dummy_setter(self, v: float) -> None:
        self._value = v

# Concrete correction parameter:
@dataclass
class ResonatorWindowSize(_DummyCorrectionParameterBase):
    name: str = field(default="resonator_window_size", init=False)
    description: str = field(default="Frequency search window width (Hz)", init=False)
```

---

## Migrating an existing operation

Follow these steps to convert an operation that has an old-style `evaluate()` that both assesses and mutates parameters.

### Step 1 — Split `evaluate()` into check methods

Each condition that was tested in `evaluate()` becomes a `_check_*` method returning a `CheckResult`. Keep it pure — no side effects.

```python
# Before
def evaluate(self):
    if self.snr < THRESHOLD:
        return OperationStatus.FAILURE
    self.readout_freq(self.fit_result.params["f_0"].value)
    return OperationStatus.SUCCESS

# After
def _check_snr(self) -> CheckResult:
    passed = self.snr >= self.snr_threshold()
    return CheckResult("snr_check", passed, f"SNR={self.snr:.3f}, threshold={self.snr_threshold():.3f}")
```

### Step 2 — Define `Correction` classes for each failure mode

Each way you'd retry the measurement becomes a `Correction` subclass. Make it stateful so it steps through options across retries. Implement `report_output()` to log what changed.

```python
class MyCorrection(Correction):
    name = "my_correction"
    description = "Short description of what this does"

    def __init__(self, param):
        self.param = param
        self._count = 0
        self._last_change = ""

    def can_apply(self) -> bool:
        return self._count < MAX

    def apply(self) -> None:
        old = self.param()
        new = compute_new_value(old, self._count)
        self.param(new)
        self._count += 1
        self._last_change = f"{old} → {new}"

    def report_output(self) -> str:
        return self._last_change
```

### Step 3 — Define `CorrectionParameter` classes for configurable knobs

Any threshold or limit that should be adjustable from the parameter manager becomes a `CorrectionParameter`. Always add `@dataclass`. Parameters live under `params.corrections.<operation_name>.<param_name>` by convention.

```python
@dataclass
class MyThreshold(CorrectionParameter):
    name: str = field(default="my_op_threshold", init=False)
    description: str = field(default="...", init=False)

    def _qick_getter(self): return self.params.corrections.my_op.threshold()
    def _qick_setter(self, v): self.params.corrections.my_op.threshold(v)
```

Then add the parameter to the instrument server. Connect via the instrumentserver client and call
`add_parameter` for each correction parameter:

```python
from instrumentserver.client.proxy import Client

c = Client()
params = c.get_instrument("parameter_manager")

params.add_parameter("corrections.my_op.threshold", initial_value=2.0, unit="")
params.add_parameter("corrections.my_op.max_steps", initial_value=3, unit="")
```

The `unit` argument is required; use `unit=""` for dimensionless quantities.

### Step 4 — Wire everything up in `__init__`

```python
def __init__(self, params):
    super().__init__()
    self._register_inputs(...)
    self._register_outputs(...)

    # 1. Register correction parameters first (they become self.* attributes)
    self._register_correction_params(
        my_threshold=MyThreshold(params),
    )

    # 2. Create correction instances (can now reference self.my_threshold)
    self._my_correction = MyCorrection(self.some_param, self.my_threshold)

    # 3. Register checks with their correction fallback chains
    self._register_check("snr_check", self._check_snr, self._my_correction)

    # 4. Register what to write on success
    self._register_success_update(
        self.output_param,
        lambda: self.fit_result.params["f_0"].value,
    )
```

### Step 5 — Move parameter mutations out of `evaluate()`

Delete the old `evaluate()` — the default implementation now handles everything via
registered checks. Delete any `correct()` override that only applied found values —
`_register_success_update` handles that too.

Only keep a `correct()` override if you need custom report messages on SUCCESS/FAILURE:

```python
def correct(self, result: EvaluateResult) -> EvaluateResult:
    result = super().correct(result)   # always call super first
    if result.status == OperationStatus.SUCCESS:
        self.report_output.append(f"Found frequency: {self.fit_result.params['f_0'].value:.6e} Hz\n")
    return result
```

---

## Multi-level correction hierarchy

When a single check has multiple levels of corrective action (fast cheap fixes first,
slow expensive fixes last), pass a list to `_register_check`. The list is a **fallback
chain**: the first correction where `can_apply()` is True is used on each retry.

The pattern used in `ResonatorSpectroscopy`:

```
Level 0: WindowShiftCorrection    — shift measurement window ±1, ±2, ... spans
Level 1: IncreaseSamplingRate     — increase frequency steps × factor, reset window
Level 2: IncreaseAveraging        — increase repetitions × factor, reset window
```

Key design points:
- **Level 0** is tried first on every retry until exhausted (all ±n shifts attempted).
- **Level 1** fires only when Level 0 is exhausted. Its `apply()` increases steps
  AND calls `window_correction.reset()` so Level 0 starts over with the new settings.
- **Level 2** fires only when Level 1 is also exhausted. Same reset pattern.
- Higher-level corrections hold a reference to lower-level ones and call `reset()` on them.

```python
self._window_shift = WindowShiftCorrection(
    self.start_frequency, self.end_frequency, self.max_window_shifts
)
self._increase_sampling = IncreaseSamplingRateCorrection(
    self.steps, self._window_shift, self.sampling_factor, self.max_sampling_increases
)
self._increase_averaging = IncreaseAveragingCorrection(
    self.repetitions, self._window_shift, self.averaging_factor, self.max_averaging_increases
)

self._register_check(
    "quality_check",
    self._check_quality,
    [self._window_shift, self._increase_sampling, self._increase_averaging],
)
```

With defaults of `max_window_shifts=3`, `max_sampling_increases=2`, `max_averaging_increases=2`,
this gives up to 30 retries before FAILURE (6 window shifts × 5 sampling/averaging levels).

---

## Multi-criteria quality checks

A single `CheckResult` can combine multiple independent criteria. The convention is to
build a list of failures and join them into the description so the report is specific:

```python
def _check_quality(self) -> CheckResult:
    threshold = self.snr_threshold()
    snr_passed = self.snr >= threshold

    max_error = self.max_fit_param_error()   # e.g. 1.0 = 100%
    bad_params = []
    for pname, param in self.fit_result.params.items():
        if param.stderr is None:
            bad_params.append(f"{pname}(no stderr)")
        elif param.value == 0 or abs(param.stderr / param.value) > max_error:
            pct = abs(param.stderr / param.value) * 100 if param.value != 0 else float("inf")
            bad_params.append(f"{pname}({pct:.0f}%)")

    passed = snr_passed and len(bad_params) == 0
    parts = [f"SNR={self.snr:.3f} (threshold={threshold:.3f})"]
    if bad_params:
        parts.append(f"high-error params: {', '.join(bad_params)}")
    return CheckResult("quality_check", passed, "; ".join(parts))
```

The `max_fit_param_error` is a `CorrectionParameter` stored at
`params.corrections.res_spec.max_fit_param_error` (default `1.0`). Certain fit
parameters with known large uncertainties can be excluded by name before the loop.

---

## What is NOT yet done

- No new `CorrectionParameter` subclasses in the dummy package (the base class is
  there; concrete examples should be added alongside real operations).
- The `_assemble_report()` HTML does not yet have a dedicated "Correction
  Parameters" section — check tables appear in `report_output` via the default
  `correct()`, but `correction_params` values are not rendered separately.
- Dummy operations have not yet been updated to use `_register_success_update` —
  they still override `correct()` manually. That update is deferred.

---

## Files changed

### Initial corrections architecture
```
src/labcore/protocols/base.py
src/labcore/protocols/__init__.py
src/labcore/testing/protocol_dummy/parameters.py
src/labcore/testing/protocol_dummy/gaussian.py
src/labcore/testing/protocol_dummy/cosine.py
src/labcore/testing/protocol_dummy/linear.py
src/labcore/testing/protocol_dummy/exponential.py
src/labcore/testing/protocol_dummy/exponential_decay.py
src/labcore/testing/protocol_dummy/exponentially_decaying_sine.py
src/labcore/testing/protocol_dummy/dummy_protocol.py
test/pytest/test_protocols.py
test/pytest/test_protocols_realistic.py
```

### Gap fixes (registration-based success updates + fallback corrections)
```
src/labcore/protocols/base.py
test/pytest/test_protocols.py
```
