# Parameters

A **parameter** is a named handle that an operation reads from or writes to.
On the surface it looks like a single getter/setter pair:

```python
qubit_frequency()        # read
qubit_frequency(5.2e9)   # write
```

Underneath, it's an abstraction layer that solves two problems an operation
should not have to think about: where the value lives between Python
processes (running a protocol on a notebook first and then on a script for example),
and how each hardware platform actually programs it.

## Why parameters?

### Persistence across processes

Lab work runs in many processes — a notebook for ad-hoc operations, a
script for a full protocol, a dashboard for live monitoring. They all need
to see the same parameter values. Parameters do not store values in
themselves; they hold a `params` proxy to whatever persistence backend the
user wants. The common choice today we use is the parameter manager from
[`instrumentserver`](https://toolsforexperiments.github.io/instrumentserver/first_steps/overview.html#parameter-manager),
but a config file or any other store works equally well — the labcore-side
API does not change.

### Hardware translation

Different platforms speak different languages. A QICK FPGA can program a
qubit frequency in GHz directly. An OPX has to split the same value into an
intermediate frequency and a local-oscillator frequency, then mix them.
Each platform-specific getter/setter on the parameter holds whatever
conversion logic that platform needs. Operations never see this — they
just call `qubit_frequency()` and get the actual frequency back.

## The shape of a parameter

A parameter is a {py:class}`dataclass <dataclasses.dataclass>` subclass of
{py:class}`ProtocolParameterBase <labcore.protocols.base.ProtocolParameterBase>`
with three fields and one platform-specific getter/setter pair per backend:

| Field | What it is                                                                                                                         |
|---|------------------------------------------------------------------------------------------------------------------------------------|
| `name` | The parameter's display name. Used in reports and logs.                                                                            |
| `description` | Plain-English description of the value.                                                                                            |
| `params` | The hardware/persistence handle. `None` on `DUMMY`; on real hardware it's typically an `instrumentserver` parameter-manager proxy. |

The class implements `_dummy_getter` / `_dummy_setter`,
`_qick_getter` / `_qick_setter`, and `_opx_getter` / `_opx_setter`. The
right pair is dispatched inside `__call__` based on which platform was
selected with
{py:func}`select_platform <labcore.protocols.select_platform>`.

## Writing a parameter

Suppose your toolbox stores qubit frequencies in an `instrumentserver`
parameter manager exposed as `params.qubit.f()`. Here is what a
`QubitFrequency` parameter looks like:

```python
from dataclasses import dataclass, field
from labcore.protocols import ProtocolParameterBase


@dataclass
class QubitFrequency(ProtocolParameterBase):
    name: str = field(default="qubit_frequency", init=False)
    description: str = field(
        default="Intermediate frequency of the qubit", init=False,
    )

    def _dummy_getter(self):
        return self.params.qubit.f()

    def _dummy_setter(self, value):
        self.params.qubit.f(value)

    def _qick_getter(self):
        return self.params.qubit.freq()

    def _qick_setter(self, value):
        self.params.qubit.freq(value)
```

The `name` and `description` fields are declared with `init=False` so the
caller does not have to repeat them — every `QubitFrequency` instance has
the same identity. Only `params` (the hardware handle) is supplied at
construction time:

```python
from labcore.protocols import select_platform

select_platform("QICK")
freq = QubitFrequency(params=my_instrument_server_proxy)

freq()           # → 5.2e9   (reads via _qick_getter)
freq(5.21e9)     # writes via _qick_setter
```

:::{note}
This example writes the same value through both `DUMMY` and `QICK` because the QICK
takes a frequency in GHz directly. An OPX getter/setter would do more work:
it would split the requested frequency into IF + LO, write the LO to the
microwave source, and write the IF to the OPX channel. That conversion is
exactly the kind of platform-specific logic the parameter abstraction is
there to hold.
:::

## You only implement the platforms you use

The base class raises `NotImplementedError` for every platform, so a
parameter only needs to implement the platforms it will actually run on. A
parameter can support `DUMMY` and `QICK` only; or `QICK` only; or even
`DUMMY` only for things that have no hardware analogue (a pure
configuration knob, say). Calling a parameter under an unimplemented
platform raises immediately, which surfaces missing support fast rather
than silently falling through.

This is the common pattern in real toolboxes — see for example
`SaturationSpecDriveGain` in `CQEDToolbox`, which is QICK-only.

## Reusing a parameter across operations

A parameter class is defined once and instantiated wherever it is needed.
The same `QubitFrequency` shows up as an input to a spectroscopy operation
and an output of a Rabi calibration:

```python
class ResonatorSpectroscopy(ProtocolOperation):
    def __init__(self, params):
        super().__init__()
        self._register_inputs(qubit_frequency=QubitFrequency(params))
        # ...

class PiSpectroscopy(ProtocolOperation):
    def __init__(self, params):
        super().__init__()
        self._register_outputs(qubit_frequency=QubitFrequency(params))
        # ...
```

Because both instances point at the same persistence backend through
`params`, a write performed by `PiSpectroscopy` is visible to every later
operation that reads `QubitFrequency`. See {doc}`operations` for the
`_register_inputs` / `_register_outputs` API.

## Real-world parameters: instrumentserver-backed

Real toolbox parameters are usually a little more elaborate than the
example above. The `instrumentserver` helper
{py:func}`nestedAttributeFromString <instrumentserver.helpers.nestedAttributeFromString>`
lets the getter/setter resolve a dotted attribute path on the proxy, which
is convenient when the parameter manager organizes values under a
per-qubit subtree:

```python
from instrumentserver.helpers import nestedAttributeFromString


@dataclass
class QubitFrequency(ProtocolParameterBase):
    name: str = field(default="qubit_frequency", init=False)
    description: str = field(default="Intermediate frequency of the qubit", init=False)

    def _qick_getter(self):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        return nestedAttributeFromString(self.params, f"{active_qubit}.qubit.freq")()

    def _qick_setter(self, value):
        active_qubit = nestedAttributeFromString(self.params, "active.qubit")()
        nestedAttributeFromString(self.params, f"{active_qubit}.qubit.freq")(value)
```

The labcore-side API has not changed — the operation still just calls
`qubit_frequency()` — but the getter now resolves an "active qubit"
indirection and looks up a per-qubit attribute path. For a full catalogue
of this style of parameter, see
[`CQEDToolbox/protocols/parameters.py`](https://github.com/toolsforexperiments/CQEDToolbox/blob/main/src/cqedtoolbox/protocols/parameters.py).
That toolbox is a working real-world example built on labcore but is not
itself documented yet.

## Correction parameters

Some parameters control a *correction strategy* rather than hardware state
— for example, a noise tolerance threshold or the number of frequency
windows to scan through. These are declared as
{py:class}`CorrectionParameter <labcore.protocols.base.CorrectionParameter>`
subclasses instead. Apart from that, they look identical to a regular
parameter:

```python
from labcore.protocols import CorrectionParameter


@dataclass
class GaussianNoiseReductionFactor(CorrectionParameter):
    name: str = field(default="gaussian_noise_reduction_factor", init=False)
    description: str = field(
        default="Factor by which the measurement noise std is divided each correction step",
        init=False,
    )

    def _dummy_getter(self):
        return self._value           # in-memory storage, no hardware

    def _dummy_setter(self, v):
        self._value = v
```

Operations register correction parameters via `_register_correction_params`;
they are excluded from the protocol's pre-execution hardware-parameter
verification because there is no hardware to verify against. See
{doc}`operations` for how corrections use these parameters.

## Where to read next

{doc}`operations` — how an operation declares its parameters and runs the
five-step lifecycle.
