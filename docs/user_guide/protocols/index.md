# Protocols

A **protocol** ties several experiments together to achieve a complete goal that no
single one of them can — *calibrating a qubit*, rather than just finding
its frequency. Each experiment is wrapped as an **operation**: a
self-contained unit that measures, analyses, and defines for itself what
counts as success, usually to nail down some number, or provides next steps with an attempt to solve its failures.
The protocol runs its operations in sequence (more complex protocols can have more complex execution flows), 
lets each one retry itself with adjusted settings if needed, and records the whole run as a self-contained HTML report. 
The result is the calibrated system, with a report that shows how you got there.

A protocol is built out of three concepts, one per sub-page:

- {doc}`parameters` — the named handles operations read from and write to
- {doc}`operations` — a single experiment, including its checks and corrections
- {doc}`protocols` — composing operations into a runnable protocol

## How protocols are organized

Every protocol is a tree of branches and operations.

```
Protocol
└── Branch                    a named sequence of items
    ├── Operation             a single measurement step
    │   ├── Parameters        named handles for inputs and outputs
    │   ├── Checks            pure assessments after analysis
    │   └── Corrections       strategies applied between retries
    └── Condition (optional)  routes execution to one of two branches
```

The simplest shape — and the one most protocols use — is a single root
branch with a flat list of operations. See {doc}`protocols` for
super-operations, conditions, and the assembled report.

## The lifecycle of an operation

Every operation runs the same five steps in order, on every attempt:

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

- `measure` — performs the measurement (or generates fake data on `DUMMY`) and saves the raw data to disk.
- `load_data` — reads the raw data back into memory and normalizes its shape and field names so the rest of the lifecycle is platform-agnostic.
- `analyze` — runs fits and statistics over the loaded data and attaches the results to the operation.
- `evaluate` — returns named check results and an overall status; pure assessment, no side effects.
- `correct` — the only place parameters get written: fitted outputs on success, a correction strategy on retry.

See {doc}`operations` for how each step is implemented and customized.

## Run a protocol in 10 lines

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

This protocol has one operation. The operation runs a noisy Gaussian fit
and assesses its own signal-to-noise ratio. The first attempt fails, a
**correction** fires that lowers the simulated noise level, and the operation
retries. After two corrections the SNR check passes, the fit succeeds, and
the protocol writes an HTML report to the current directory.

A few things to notice:

- {py:func}`select_platform <labcore.protocols.select_platform>` is required
  before any protocol can be instantiated. It tells parameters and operations
  which hardware backend to dispatch to. `"DUMMY"` is the in-memory backend
  used for testing.
- The protocol is just a class with a `root_branch`. The branch holds a
  flat list of operations.
- The correction strategy lives **inside** the operation. The protocol does
  not know or care that this particular operation retries itself.

:::{note}
At the moment, protocols only support the `DUMMY`, `QICK`, and `OPX`
platforms. Adding a new platform is a small change — if you need one,
please [open an issue on GitHub](https://github.com/toolsforexperiments/labcore/issues).
:::

## Where to read next

Read in order: {doc}`parameters` → {doc}`operations` → {doc}`protocols`.
Each page builds on the previous one.

```{toctree}
:hidden:

parameters
operations
protocols
```
