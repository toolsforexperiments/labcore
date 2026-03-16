import logging
from pathlib import Path
from typing import Any

from labcore.protocols.base import (
    BranchBase,
    Condition,
    OperationStatus,
    ProtocolBase,
    SuperOperationBase,
)
from labcore.testing.protocol_dummy.cosine import CosineOperation
from labcore.testing.protocol_dummy.exponential import ExponentialOperation
from labcore.testing.protocol_dummy.exponential_decay import ExponentialDecayOperation
from labcore.testing.protocol_dummy.exponentially_decaying_sine import (
    ExponentiallyDecayingSineOperation,
)
from labcore.testing.protocol_dummy.gaussian import GaussianOperation
from labcore.testing.protocol_dummy.linear import LinearOperation

logger = logging.getLogger(__name__)

# Global test variable - change this to switch branches
USE_BRANCH_A = True
USE_BRANCH_C = True


class DummySuperOperation(SuperOperationBase):
    """
    Example SuperOperation that groups multiple calibration operations together.

    This demonstrates:
    - Grouping Exponential and ExponentialDecay operations
    - Retry mechanism (will retry 2 times for testing)
    - Report aggregation
    """

    def __init__(self, params: Any = None) -> None:
        super().__init__()

        # Define the sequence of operations
        self.operations = [
            ExponentialOperation(params),
            ExponentialDecayOperation(params),
        ]

        # Configure retry behavior
        self.max_attempts = 3  # Will retry up to 3 times total

    def evaluate(self) -> OperationStatus:
        """
        Evaluate the overall success of all sub-operations.
        Uses same retry testing mechanism as GaussianProtocol.
        """
        logger.info(
            f"[{self.name}] All sub-operations completed at attempt {self.total_attempts_made}"
        )

        # Similar retry mechanism to GaussianProtocol for testing
        if self.total_attempts_made != 3:
            logger.info(
                f"[{self.name}] At {self.total_attempts_made} attempts, requesting retry for testing"
            )
            return OperationStatus.RETRY

        logger.info(f"[{self.name}] Reached 3 attempts, returning SUCCESS")
        return OperationStatus.SUCCESS


class DummyProtocol(ProtocolBase):
    def __init__(self, params: Any = None, report_path: Path = Path("")) -> None:
        super().__init__(report_path)

        # Create main branch
        main = BranchBase("Main")
        main.append(GaussianOperation(params))

        # Add SuperOperation to demonstrate grouping operations with retry
        main.append(DummySuperOperation(params))

        # Branch A: Cosine, Exponential, Linear
        branch_a = BranchBase("BranchA")
        branch_a.append(CosineOperation(params))
        branch_a.append(ExponentialOperation(params))
        branch_a.append(LinearOperation(params))

        # Branch B: ExponentialDecay, ExponentiallyDecayingSine
        branch_b = BranchBase("BranchB")
        branch_b.append(ExponentialDecayOperation(params))
        branch_b.append(ExponentiallyDecayingSineOperation(params))

        branch_c = BranchBase("BranchC")
        branch_c.append(LinearOperation(params))
        branch_c.append(GaussianOperation(params))

        branch_d = BranchBase("BranchD")
        branch_d.append(ExponentialOperation(params))
        branch_d.append(CosineOperation(params))

        branch_b.append(
            Condition(
                condition=lambda: USE_BRANCH_C,
                true_branch=branch_c,
                false_branch=branch_d,
                name="Sub-Branch Selection",
            )
        )

        # Add conditional branching
        main.append(
            Condition(
                condition=lambda: USE_BRANCH_A,
                true_branch=branch_a,
                false_branch=branch_b,
                name="Branch Selection",
            )
        )

        self.root_branch = main
