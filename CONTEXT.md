# labcore — Domain Context

The vocabulary used across the codebase and docs. Update entries as terms are
clarified; remove or rewrite entries that go stale.

## Protocols subsystem

- **Protocol** — the top-level entity a lab user runs end-to-end (e.g. a qubit
  tune-up). A protocol holds a tree of branches and operations that execute in
  sequence, with optional conditional branching. Implemented as a subclass of
  `ProtocolBase` whose `__init__` builds `self.root_branch`.

- **Operation** — a single measurement step inside a protocol (e.g. a resonator
  spectroscopy, a power Rabi). Each operation follows a fixed lifecycle:
  `measure → load_data → analyze → evaluate → correct`. Implemented as a
  subclass of `ProtocolOperation`.

- **Parameter** — a named handle that an operation reads from or writes to.
  Sits between operations and two concerns the operation does not want to know
  about:
  1. **Persistence across processes.** Lab work runs in many processes — a
     notebook for ad-hoc operations, a script for a full protocol — and
     parameter values must survive process boundaries. Each parameter holds a
     `params` proxy to whatever persistence layer is in use (typically the
     `instrumentserver` parameter manager, but a config file or any other
     store works equally well).
  2. **Hardware translation.** Different platforms speak different languages.
     QICK can program a qubit frequency in GHz directly; OPX has to split the
     same value into IF + LO and mix. Each platform-specific getter/setter
     (`_qick_getter`, `_opx_getter`, `_dummy_getter`) carries whatever
     conversion logic that platform needs.

  The analysis layer only sees the resolved value via `param()`; it does not
  care how it was produced. Operations register parameters via
  `_register_inputs`, `_register_outputs`, and `_register_correction_params`.

- **Correction parameter** — a parameter that controls a *correction strategy*
  rather than hardware state (e.g. a noise tolerance, a step count). Subclass
  of `CorrectionParameter`. Excluded from hardware verification; otherwise
  identical to `ProtocolParameterBase`.

- **Check** — a pure, side-effect-free assessment performed during `evaluate()`,
  producing a `CheckResult(name, passed, description)`. An operation can
  register multiple checks; the default `evaluate()` runs them all and returns
  RETRY if any fail.

- **Correction** — a strategy applied *between retries* when a specific check
  fails. One instance per operation, created in `__init__` and reused across
  retries so stateful strategies (e.g. stepping through a list of windows) work
  correctly. A correction declares which check it is `triggered_by`.

- **Branch** — a named sequence of operations and conditions inside a protocol.
  Implemented as `BranchBase`. The simplest protocol is one root branch
  containing a flat list of operations (see `QubitTuneup`).

- **Platform** — the hardware backend a protocol runs against (`DUMMY`, `QICK`,
  `OPX`). Selected globally via the `PLATFORMTYPE` module variable in
  `labcore.protocols.base`; parameters and operations dispatch to
  platform-specific code (`_dummy_getter`, `_qick_getter`, …) based on it.

- **Report** — a self-contained HTML document assembled by
  `ProtocolBase._assemble_report()` after a protocol runs. Each operation
  contributes by appending strings (markdown) and figure paths to
  `self.report_output`; figures are embedded as base64 data URIs so the
  resulting file stands on its own. The default `correct()` adds a check
  table; `_register_success_update` adds parameter-improvement lines.
  SuperOperations aggregate their sub-operations' contributions. Saved under
  `report_path / "{ProtocolName}_report"`.
