# Parameters abstract persistence and hardware translation

Parameters in `labcore.protocols` are dataclass subclasses of
`ProtocolParameterBase` that hold a `params` proxy to a persistence backend
and implement platform-specific getter/setter methods (`_qick_getter`,
`_opx_getter`, `_dummy_getter`). Operations interact with parameters through
a uniform `param() / param(value)` call, never with the backing store or the
target platform directly. We chose this shape because parameters sit between
operations and two concerns the operation must not be coupled to: a
persistence layer that survives Python-process boundaries (notebook running
one operation, script running a full protocol — same parameter values), and
hardware platforms that handle parameters in non-equivalent ways (QICK takes
a qubit frequency in GHz directly; OPX has to split it into IF + LO and mix).

## Considered Options

- **Flat dict-of-values.** A `dict[str, float]` shared via a module global
  or passed into operations. Rejected: provides no place for hardware
  translation logic, and forces persistence to be solved in user code.
- **Direct coupling to `instrumentserver`.** Make every parameter call
  `instrumentserver.helpers.nestedAttributeFromString` directly, no
  abstraction. Rejected: hard-codes one persistence backend; users wanting
  config files or other stores would have to fork. Also still leaves the
  hardware-translation problem unsolved.
- **Per-operation hardcoding.** Each operation reads/writes hardware in its
  own `_measure_*` body. Rejected: parameters are typically reused across
  many operations (a `QubitFrequency` shows up in spectroscopy, Rabi, T1, …)
  and duplicating the read/write/translate logic per operation is a
  maintenance hazard.

## Consequences

- **More boilerplate per parameter.** A parameter that needs to support
  three platforms is ~30 lines of dataclass + getter/setter pairs even when
  the logic is trivial. Mitigated by the "implement only the platforms you
  use" pattern — most parameters today implement DUMMY + QICK only and let
  the others raise `NotImplementedError`.
- **Persistence backend is swappable.** A toolbox can use the
  `instrumentserver` parameter manager (the common choice today), a config
  file, or any other store, without changes to operations or to labcore.
- **New platforms add zero churn to existing operations.** Adding OPX
  support for a parameter is a localized change — implement
  `_opx_getter`/`_opx_setter`. Operations and the analysis layer don't move.
- **Analysis layer stays clean.** Analysis only ever calls `param()` and
  receives the resolved value; it does not see the platform-specific
  conversion logic.
