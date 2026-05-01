# Plan — Protocols User Guide

Working plan for the `protocol_guide` branch. Scope: write the user-facing
documentation for the `labcore.protocols` subsystem, including the
corrections feature that just landed in PR #105.

## 1. Goals

- Teach an **operation author** (a physicist writing measurement code) how to
  use `labcore.protocols`: parameters, operations, the lifecycle, checks,
  corrections, and how to assemble operations into a protocol.
- Give a "10 lines and it runs" first-impression that's true to the real API.
- Surface the corrections feature (the headline of PR #105) prominently —
  it's tightly coupled to operations, so it lives on the operations page.
- Not in scope: framework-internals reference, contributor docs, end-user
  GUI manual.

Audience precedence: operation author > lab user running existing protocols
> framework contributor (already served by `notes/protocol_corrections_architecture.md`).

## 2. File structure

Replace the empty top-level `docs/user_guide/protocols.md` with a directory:

```
docs/user_guide/
├── index.md                       (already in repo — update toctree)
└── protocols/
    ├── index.md                   (intro)
    ├── parameters.md
    ├── operations.md
    └── building_protocols.md
```

Update `docs/user_guide/index.md` to point its toctree at `protocols/index`
(matches the existing `measurement/index`, `data/index`, `instruments/index`
pattern).

Asset path for the report screenshot: `docs/_static/protocols/qubit_tuneup_report.png`.
The doc will reference it via a standard image directive; the file can be
added later without re-touching the docs.

## 3. Code changes alongside the docs

- **Add `select_platform()` helper** in `src/labcore/protocols/__init__.py`.
  Wraps the global `PLATFORMTYPE` assignment so the public API is
  `from labcore.protocols import select_platform; select_platform("DUMMY")`
  instead of `proto_base.PLATFORMTYPE = ...`. Used by every snippet in the
  doc. Five-minute change; removes a wart from the marketing snippet.

No other code changes in scope. The TODO about `self.condition: str` and the
TODO about Conditions are out of scope for this branch.

## 4. Per-page outlines

### 4.1 `protocols/index.md` — Introduction

Shape: snippet within the first scroll, then brief diagrams.

```
# Protocols

(1 paragraph) What is a protocol? — defines it as a runnable sequence of
measurement steps, mentions QubitTuneup as a real example.

## Run a protocol in 10 lines
(snippet — see §5)

## How protocols are organized
(brief ASCII tree diagram: Protocol → Branch → Operation → {Parameters,
 Corrections}; 2–3 sentences naming each)
→ See parameters.md, building_protocols.md

## The lifecycle of an operation
(brief ASCII lifecycle diagram: measure → load_data → analyze → evaluate →
 correct; 2–3 sentences)
→ See operations.md

## Where to read next
parameters → operations → building_protocols
```

### 4.2 `protocols/parameters.md`

```
# Parameters

## Why parameters?
Two problems they solve:
- Persistence across processes (notebook + script + protocol runner share
  the same values via a pluggable backend; instrumentserver parameter
  manager is the common one, but config files / other stores work too)
- Hardware translation (QICK takes a frequency in GHz directly; OPX has to
  split it into IF + LO; the parameter is where that conversion lives)

Analysis layer never touches this — it just calls `param()`.

## The shape of a parameter
- Dataclass subclass of `ProtocolParameterBase`
- Fields: name, description, params (hardware handle, typed Any)
- Called QCoDeS-style: `param()` / `param(value)`
- Platform dispatch in `__call__` based on PLATFORMTYPE

## Writing a parameter
Walkthrough: write `QubitFrequency` from scratch — DUMMY + QICK only.
Use the simple `self.params.qubit.f()` style, NOT `nestedAttributeFromString`.
Sidebar: "QICK takes the GHz value directly. A future OPX getter would
split into IF + LO and mix here." This grounds the §"Why parameters?" claim.

## You only implement the platforms you use
The base class raises `NotImplementedError` per platform. Many real
parameters support DUMMY + QICK only; some are QICK-only
(`SaturationSpecDriveGain`); flux/`ECParam`/`ELParam`/`EJParam` are DUMMY-only.

## Reusing a parameter across operations
Short: same `QubitFrequency` wired into two different operations'
`_register_inputs(...)`.

## Real-world parameters: persistence backends
~10 lines. Show one snippet using `instrumentserver.helpers.nestedAttributeFromString`
inline so the doc is self-contained. Mention by name:
- The `instrumentserver` parameter manager as the common backend
- `CQEDToolbox/protocols/parameters.py` as a real-world catalogue (note:
  CQEDToolbox is currently undocumented)

## Correction parameters
Brief: `CorrectionParameter` subclass. Skips hardware verification. Example:
`GaussianNoiseReductionFactor`. Operations register them via
`_register_correction_params(...)`.

## Where to read next
operations.md
```

### 4.3 `protocols/operations.md`

Outline C: topical body, then a "putting it all together" appendix with the
full `gaussian_with_correction.py` inline. Single page, length is fine.

```
# Operations

(1 paragraph) What an operation is; pointer back to the lifecycle diagram
on the index page.

## The lifecycle of an operation
Reproduce the ASCII lifecycle diagram. 1–2 paragraphs per step:
- measure (writes raw data; platform-specific)
- load_data (pulls it back; platform-specific)
- analyze (computation, fitting, attaches results to self; no parameter
  writes)
- evaluate (pure assessment; returns EvaluateResult with check results)
- correct (the only place parameters are written)

## A minimal operation
A stripped-down GaussianFit (no corrections): measure + analyze + one
check, no Correction registered. Smallest thing that runs.

## Registering inputs, outputs, and platform code
- `_register_inputs(...)` / `_register_outputs(...)` / `_register_correction_params(...)`
- `_measure_dummy` / `_measure_qick` / `_measure_opx`
- `_load_data_dummy` / `_load_data_qick` / `_load_data_opx`
- Platform dispatch is the same as for parameters

## Checks: assessing the result
- `_register_check(name, check_func, correction)`
- `CheckResult(name, passed, description)`
- Default `evaluate()`: all pass → SUCCESS, any fail → RETRY

## Corrections: doing something between retries
- `Correction` subclass: `name`, `description`, `triggered_by`,
  `can_apply()`, `apply()`
- One instance per operation, persists across retries (state lives in the
  correction)
- Walk through `_ReduceNoiseLevelCorrection` from gaussian_with_correction
- Fallback chain: pass `list[Correction]`; first applicable one used

## Writing back on success
- `_register_success_update(param=..., value_func=lambda: ...)`
- Lazy: value_func runs at correct() time
- When this is enough, you don't override correct() at all

## When to override correct()
- Custom report messages
- Logic that doesn't fit success-update or correction
- Always call `super().correct(result)` first

## Adding to the report from an operation
- `self.report_output.append(...)` for markdown strings and figure paths
- Default `correct()` already adds a check table on RETRY/FAILURE and
  parameter-improvement lines on SUCCESS
- Show the SNR-on-success/failure pattern from `gaussian_with_correction`

## Putting it all together
Full `gaussian_with_correction.py` inline (~150 lines). Light annotations
calling out which earlier section each piece corresponds to.

## Where to read next
building_protocols.md
```

### 4.4 `protocols/building_protocols.md`

```
# Building Protocols

(1 paragraph) A protocol is a tree of operations + branches. Simplest case
is one root branch with a flat list. Branches and conditions are there
when the flow needs to be dynamic.

## Picking a platform
- `select_platform("DUMMY")` / `("QICK")` / `("OPX")`
- Required before instantiating any Protocol
- Top of script / notebook
- (uses the helper added alongside this doc)

## A simple protocol — the flat case
Walkthrough of `QubitTuneup` from CQEDToolbox:
- Subclass `ProtocolBase`
- Set `self.root_branch = BranchBase("name")`
- `self.root_branch.extend([Op(params), Op(params), ...])`
- `params` flows down to every operation

## Running and inspecting a protocol
- `protocol.execute()` — runs the tree
- `protocol.success` — True / False / None
- `verify_all_parameters()` runs before execute and bails if any
  parameter is missing or invalid

## The protocol report
- Auto-assembled HTML at the end of `execute()`
- `report_path` argument → where it lands; default cwd
- Self-contained: figures embedded as base64 data URIs (one file, mailable)
- TOC + per-operation sections + condition routing + retry attempts visible
- Show a screenshot: `docs/_static/protocols/qubit_tuneup_report.png`
  (placeholder — real asset added later)
- :::{warning}
  Re-running a protocol **overwrites** the previous report directory.
  Copy or rename `<report_path>/<ProtocolName>_report` before re-running
  if you want to keep a prior run.
  :::

## Super-operations: a retry boundary around several operations
- `SuperOperationBase` — composite operation that groups N operations under
  one retry boundary
- Sub-operations have their own measure/load_data/analyze; the super does
  not
- Use case: a calibration suite where the full sequence should retry as a
  unit
- One small worked example; mention `DummySuperOperation` as a runnable
  reference and the `CalibrationSuite` from the docstring

## Branches and conditions
- `BranchBase`: `extend([...])` for a sequence
- `Condition(condition=callable, true_branch=..., false_branch=...)` for
  dynamic routing
- Show the SNR-based routing example from the Condition docstring
- (No mention of the `self.condition: str` field — it's being phased out)

## Where to read next
- The dummy package (`labcore.testing.protocol_dummy`) — runnable catalogue
- `CQEDToolbox/protocols/` — real-world reference (currently undocumented)
```

## 5. The 10-line snippet (intro page)

Shape B: one-operation protocol, mirrors `QubitTuneup`. Final form:

```python
from labcore.protocols import select_platform, ProtocolBase, BranchBase
from labcore.testing.protocol_dummy.gaussian_with_correction import (
    GaussianWithCorrectionOperation,
)

select_platform("DUMMY")

class HelloProtocol(ProtocolBase):
    def __init__(self):
        super().__init__()
        self.root_branch = BranchBase("hello")
        self.root_branch.extend([GaussianWithCorrectionOperation()])

HelloProtocol().execute()
```

10 functional lines. Demonstrates: platform selection, ProtocolBase subclass,
root branch, an operation with a registered correction (which fires twice
before the SNR check passes — visible in logs / report).

Annotation in the doc highlights:
- `select_platform` is required before any Protocol can be instantiated
- The operation contains a correction strategy → see operations.md
- The HTML report lands in cwd → see building_protocols.md

## 6. Domain artifacts (already in place)

- `CONTEXT.md` — populated with: Protocol, Operation, Parameter, Correction
  Parameter, Check, Correction, Branch, Platform, Report
- `docs/adr/0001-parameters-abstract-persistence-and-hardware.md` —
  records the rationale for the parameter abstraction (persistence +
  hardware translation), three rejected alternatives, and consequences

## 7. Silent omissions

- The `self.condition: str` legacy field on `ProtocolOperation` (being
  phased out per TODO at base.py:758)
- OPX getter/setter implementations (no real OPX hardware to validate
  against yet)
- Framework-internals: `_RegisteredCheck`, `_RegisteredSuccessUpdate`,
  `_flatten_branch_for_execution`, `_collect_all_operations_from_branch`,
  `_assemble_report` internals
- The `qick_path` field on dummy parameters (looks like a leak; not user-facing)

## 8. Implementation order

1. **`select_platform()` helper** in `protocols/__init__.py` — first, so the
   snippet works.
2. **`docs/user_guide/index.md` toctree update** — point at `protocols/index`.
3. **`docs/user_guide/protocols/index.md`** — write the intro, validate the
   snippet runs end-to-end against the new helper.
4. **`docs/user_guide/protocols/parameters.md`**.
5. **`docs/user_guide/protocols/operations.md`**.
6. **`docs/user_guide/protocols/building_protocols.md`**.
7. **Delete the empty `docs/user_guide/protocols.md`** placeholder file.
8. **Local doc build** to confirm everything renders, ASCII diagrams hold up,
   internal links resolve.

## 9. Open items (handled later)

- Screenshot of an actual report HTML at
  `docs/_static/protocols/qubit_tuneup_report.png` — author runs
  `QubitTuneup` once and saves a screenshot.
- Decide whether to publish CQEDToolbox docs (out of scope for this
  branch).
- The `condition: str` cleanup and the `Condition` API stabilization (out
  of scope).
- A `select_platform`-style change for `report_path` ergonomics (e.g.
  timestamped report dirs) — out of scope, but the warning admonition
  in §4.4 documents the current behavior honestly.
