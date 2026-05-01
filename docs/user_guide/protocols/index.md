# Protocols

A **protocol** is a runnable sequence of measurement steps that take a lab
experiment from setup to a final report. A typical protocol — for example a
qubit tune-up — runs a resonator spectroscopy, a Rabi calibration, a T1
measurement, and so on, one after another. Each step assesses its own
results and can retry itself with adjusted settings before moving on. When
the whole thing finishes you get a single self-contained HTML report.

This part of the user guide is split into three pages, one for each
authoring concern:

- {doc}`parameters` — the named handles operations read from and write to
- {doc}`operations` — a single measurement step, including checks and corrections
- {doc}`building_protocols` — composing operations into a runnable protocol

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
branch with a flat list of operations. See {doc}`building_protocols` for
super-operations, conditions, and the assembled report.

## The lifecycle of an operation

Every operation runs the same five steps in order, on every attempt:

```
measure ──▶ load_data ──▶ analyze ──▶ evaluate ──▶ correct
   │            │            │            │            │
 write       pull data    compute       check       parameter
 hardware    back into    (fitting,    results       writes;
 / save      memory       statistics)  (pure         apply any
 raw data                              assessment)   correction
```

The two halves are deliberately separated:

- `evaluate` is **pure assessment** — it produces named check results but
  never writes parameters.
- `correct` is the **only** place an operation modifies parameters. On
  success it writes the fitted output back; on failure it applies a
  correction strategy before the next retry.

See {doc}`operations` for how each step is implemented and customized.

## Where to read next

Read in order: {doc}`parameters` → {doc}`operations` → {doc}`building_protocols`.
Each page builds on the previous one.

```{toctree}
:hidden:

parameters
operations
building_protocols
```
