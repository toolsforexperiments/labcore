from labcore.testing.protocol_dummy.parameters import (
    _DummyParameterBase,
    GaussianCenter,
    GaussianSigma,
    GaussianOffset,
    GaussianAmplitude,
    CosineAmplitude,
    CosineFrequency,
    CosinePhase,
    CosineOffset,
    ExponentialA,
    ExponentialB,
    ExponentialDecayAmplitude,
    ExponentialDecayOffset,
    ExponentialDecayTau,
    LinearSlope,
    LinearOffset,
    ExponentiallyDecayingSineAmplitude,
    ExponentiallyDecayingSineOffset,
    ExponentiallyDecayingSineFrequency,
    ExponentiallyDecayingSinePhase,
    ExponentiallyDecayingSineTau,
)
from labcore.testing.protocol_dummy.gaussian import GaussianOperation
from labcore.testing.protocol_dummy.exponential import ExponentialOperation
from labcore.testing.protocol_dummy.exponential_decay import ExponentialDecayOperation
from labcore.testing.protocol_dummy.cosine import CosineOperation
from labcore.testing.protocol_dummy.linear import LinearOperation
from labcore.testing.protocol_dummy.exponentially_decaying_sine import (
    ExponentiallyDecayingSineOperation,
)
from labcore.testing.protocol_dummy.dummy_protocol import DummySuperOperation, DummyProtocol
