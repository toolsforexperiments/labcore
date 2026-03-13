from dataclasses import dataclass, field

from labcore.protocols.base import ProtocolParameterBase


@dataclass
class _DummyParameterBase(ProtocolParameterBase):
    """Base for all dummy protocol parameters.

    Provides simple in-memory value storage via the DUMMY getter/setter so
    that ``param()`` and ``param(value)`` work without any external hardware.
    The stored value is initialised to ``0.0`` in ``__post_init__``.
    """

    def __post_init__(self):
        super().__post_init__()
        self._value: float = 0.0

    def _dummy_getter(self):
        return self._value

    def _dummy_setter(self, v):
        self._value = v


# ---------------------------------------------------------------------------
# Gaussian parameters: A * exp(-((x - x0)^2) / (2 * sigma^2))
# ---------------------------------------------------------------------------


@dataclass
class GaussianCenter(_DummyParameterBase):
    name: str = field(default="gaussian_center", init=False)
    description: str = field(default="Center position (x0) of the Gaussian", init=False)
    qick_path: str = field(default="", init=False)


@dataclass
class GaussianSigma(_DummyParameterBase):
    name: str = field(default="gaussian_sigma", init=False)
    description: str = field(default="Width (sigma) of the Gaussian", init=False)
    qick_path: str = field(default="", init=False)


@dataclass
class GaussianOffset(_DummyParameterBase):
    name: str = field(default="gaussian_offset", init=False)
    description: str = field(default="Offset (y0) of the Gaussian baseline", init=False)
    qick_path: str = field(default="", init=False)


@dataclass
class GaussianAmplitude(_DummyParameterBase):
    name: str = field(default="gaussian_amplitude", init=False)
    description: str = field(default="Amplitude (A) of the Gaussian peak", init=False)
    qick_path: str = field(default="", init=False)


# ---------------------------------------------------------------------------
# Cosine parameters: A * cos(2*pi*f*x + phi) + of
# ---------------------------------------------------------------------------


@dataclass
class CosineAmplitude(_DummyParameterBase):
    name: str = field(default="cosine_amplitude", init=False)
    description: str = field(default="Amplitude (A) of the cosine", init=False)
    qick_path: str = field(default="", init=False)


@dataclass
class CosineFrequency(_DummyParameterBase):
    name: str = field(default="cosine_frequency", init=False)
    description: str = field(default="Frequency (f) of the cosine", init=False)
    qick_path: str = field(default="", init=False)


@dataclass
class CosinePhase(_DummyParameterBase):
    name: str = field(default="cosine_phase", init=False)
    description: str = field(default="Phase (phi) of the cosine", init=False)
    qick_path: str = field(default="", init=False)


@dataclass
class CosineOffset(_DummyParameterBase):
    name: str = field(default="cosine_offset", init=False)
    description: str = field(default="Offset (of) of the cosine", init=False)
    qick_path: str = field(default="", init=False)


# ---------------------------------------------------------------------------
# Exponential parameters: a * b^x
# ---------------------------------------------------------------------------


@dataclass
class ExponentialA(_DummyParameterBase):
    name: str = field(default="exponential_a", init=False)
    description: str = field(default="Coefficient (a) of the exponential", init=False)
    qick_path: str = field(default="", init=False)


@dataclass
class ExponentialB(_DummyParameterBase):
    name: str = field(default="exponential_b", init=False)
    description: str = field(default="Base (b) of the exponential", init=False)
    qick_path: str = field(default="", init=False)


# ---------------------------------------------------------------------------
# ExponentialDecay parameters: A * exp(-x/tau) + of
# ---------------------------------------------------------------------------


@dataclass
class ExponentialDecayAmplitude(_DummyParameterBase):
    name: str = field(default="exponential_decay_amplitude", init=False)
    description: str = field(
        default="Amplitude (A) of the exponential decay", init=False
    )
    qick_path: str = field(default="", init=False)


@dataclass
class ExponentialDecayOffset(_DummyParameterBase):
    name: str = field(default="exponential_decay_offset", init=False)
    description: str = field(default="Offset (of) of the exponential decay", init=False)
    qick_path: str = field(default="", init=False)


@dataclass
class ExponentialDecayTau(_DummyParameterBase):
    name: str = field(default="exponential_decay_tau", init=False)
    description: str = field(
        default="Time constant (tau) of the exponential decay", init=False
    )
    qick_path: str = field(default="", init=False)


# ---------------------------------------------------------------------------
# Linear parameters: m * x + of
# ---------------------------------------------------------------------------


@dataclass
class LinearSlope(_DummyParameterBase):
    name: str = field(default="linear_slope", init=False)
    description: str = field(default="Slope (m) of the linear function", init=False)
    qick_path: str = field(default="", init=False)


@dataclass
class LinearOffset(_DummyParameterBase):
    name: str = field(default="linear_offset", init=False)
    description: str = field(default="Offset (of) of the linear function", init=False)
    qick_path: str = field(default="", init=False)


# ---------------------------------------------------------------------------
# ExponentiallyDecayingSine parameters: A * sin(2*pi*(f*x + phi/360)) * exp(-x/tau) + of
# ---------------------------------------------------------------------------


@dataclass
class ExponentiallyDecayingSineAmplitude(_DummyParameterBase):
    name: str = field(default="exp_decay_sine_amplitude", init=False)
    description: str = field(
        default="Amplitude (A) of the exponentially decaying sine", init=False
    )
    qick_path: str = field(default="", init=False)


@dataclass
class ExponentiallyDecayingSineOffset(_DummyParameterBase):
    name: str = field(default="exp_decay_sine_offset", init=False)
    description: str = field(
        default="Offset (of) of the exponentially decaying sine", init=False
    )
    qick_path: str = field(default="", init=False)


@dataclass
class ExponentiallyDecayingSineFrequency(_DummyParameterBase):
    name: str = field(default="exp_decay_sine_frequency", init=False)
    description: str = field(
        default="Frequency (f) of the exponentially decaying sine", init=False
    )
    qick_path: str = field(default="", init=False)


@dataclass
class ExponentiallyDecayingSinePhase(_DummyParameterBase):
    name: str = field(default="exp_decay_sine_phase", init=False)
    description: str = field(
        default="Phase (phi) of the exponentially decaying sine", init=False
    )
    qick_path: str = field(default="", init=False)


@dataclass
class ExponentiallyDecayingSineTau(_DummyParameterBase):
    name: str = field(default="exp_decay_sine_tau", init=False)
    description: str = field(
        default="Time constant (tau) of the exponentially decaying sine", init=False
    )
    qick_path: str = field(default="", init=False)
