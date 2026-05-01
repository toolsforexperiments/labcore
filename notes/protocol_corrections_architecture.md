# Protocol Corrections Architecture

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
```

Subclass this for each corrective strategy. One **instance per operation**, created
in `__init__` and reused across retries so stateful strategies (e.g. stepping
through a frequency list) work correctly.

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

    def can_apply(self) -> bool:
        return self._idx < len(self.windows)

    def apply(self) -> None:
        self.freq_center_param(self.windows[self._idx])
        self._idx += 1
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
- On RETRY: for each failed check, finds the **first** registered `Correction` where `can_apply()` is True:
  - No corrections registered → returns `EvaluateResult(FAILURE, checks)`
  - All corrections exhausted → returns `EvaluateResult(FAILURE, checks)`
  - Otherwise → calls `apply()`, logs the correction
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
