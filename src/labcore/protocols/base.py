import base64
import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

import markdown
import numpy as np
from numpy.typing import ArrayLike


logger = logging.getLogger(__name__)


def serialize_fit_params(params):
    return {n: dict(value=v.value, error=v.stderr) for n, v in params.items()}


class PlatformTypes(Enum):
    OPX = auto()
    QICK = auto()
    DUMMY = auto()


PLATFORMTYPE: PlatformTypes | None = None


@dataclass
class ProtocolParameterBase:
    """
    Base class for protocol parameters with platform-specific getter/setter methods.

    Subclasses must implement the platform-specific getter/setter methods:
    - _qick_getter() / _qick_setter(value) for QICK platform
    - _opx_getter() / _opx_setter(value) for OPX platform
    - _dummy_getter() / _dummy_setter(value) for DUMMY platform

    These methods should directly call the QCoDeS-style parameter with:
    - Get: my_param()
    - Set: my_param(value)

    Example:
        >>> @dataclass
        >>> class QubitFrequency(ProtocolParameterBase):
        ...     name: str = field(default="Qubit IF", init=False)
        ...     description: str = field(default="Qubit intermediate frequency", init=False)
        ...
        ...     def _qick_getter(self):
        ...         return self.params.qubit.f_ge()
        ...
        ...     def _qick_setter(self, value):
        ...         return self.params.qubit.f_ge(value)
        ...
        ...     def _opx_getter(self):
        ...         return self.params.qubit.frequency()
        ...
        ...     def _opx_setter(self, value):
        ...         return self.params.qubit.frequency(value)
    """

    global PLATFORMTYPE

    name: str
    # FIXME: this should be typed as ProxyInstrumentModule from instrumentserver,
    # but labcore should not depend on that package. params is only stored and
    # passed through here; hardware access lives in subclasses in the measurement
    # repo. Retype to a structural Protocol or remove the dependency when cleaning
    # up the migration.
    params: Any
    description: str
    platform_type: PlatformTypes = PLATFORMTYPE

    # dataclasses defaults are evaluated at import time, not runtime.
    # This means we need to re-apply the PLATFORMTYPE when an instance is created
    def __post_init__(self):
        if self.platform_type is None:
            self.platform_type = PLATFORMTYPE

        # Validate params is provided for non-DUMMY platforms
        if self.platform_type != PlatformTypes.DUMMY and self.params is None:
            raise ValueError(
                f"params argument is required for {self.platform_type} platform"
            )

    def __call__(self, value=None):
        """
        QCoDeS-style parameter calling convention.

        Usage:
            param()      # Get value (no arguments)
            param(42)    # Set value to 42
        """
        if value is None:
            # Getter: no arguments provided
            match self.platform_type:
                case PlatformTypes.QICK:
                    return self._qick_getter()
                case PlatformTypes.OPX:
                    return self._opx_getter()
                case PlatformTypes.DUMMY:
                    return self._dummy_getter()
            raise NotImplementedError(
                f"Platform type {self.platform_type} not implemented"
            )
        else:
            # Setter: value provided
            match self.platform_type:
                case PlatformTypes.QICK:
                    return self._qick_setter(value)
                case PlatformTypes.OPX:
                    return self._opx_setter(value)
                case PlatformTypes.DUMMY:
                    return self._dummy_setter(value)
            raise NotImplementedError(
                f"Platform type {self.platform_type} not implemented"
            )

    def _qick_getter(self):
        """Get parameter value for QICK platform. Subclasses must implement."""
        raise NotImplementedError(
            f"QICK getter not implemented for parameter '{self.name}'"
        )

    def _qick_setter(self, value):
        """Set parameter value for QICK platform. Subclasses must implement."""
        raise NotImplementedError(
            f"QICK setter not implemented for parameter '{self.name}'"
        )

    def _opx_getter(self):
        """Get parameter value for OPX platform. Subclasses must implement."""
        raise NotImplementedError(
            f"OPX getter not implemented for parameter '{self.name}'"
        )

    def _opx_setter(self, value):
        """Set parameter value for OPX platform. Subclasses must implement."""
        raise NotImplementedError(
            f"OPX setter not implemented for parameter '{self.name}'"
        )

    def _dummy_getter(self):
        """Get parameter value for DUMMY platform. Subclasses must implement."""
        raise NotImplementedError(
            f"DUMMY getter not implemented for parameter '{self.name}'"
        )

    def _dummy_setter(self, value):
        """Set parameter value for DUMMY platform. Subclasses must implement."""
        raise NotImplementedError(
            f"DUMMY setter not implemented for parameter '{self.name}'"
        )


class OperationStatus(Enum):
    """
    Return status for ProtocolOperation.evaluate()

    Indicates what the protocol executor should do next with this operation.
    """

    SUCCESS = "success"
    RETRY = "retry"
    FAILURE = "failure"

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"OperationStatus.{self.name}"


@dataclass
class ParamImprovement:
    old_value: Any
    new_value: Any
    param: ProtocolParameterBase


# TODO: How do we handle different saving for different scenarios? For example:
#  For the lab we use run_measurement, for something like the lccf it will be something different.
#  In the same way that if some other lab wants to run this, they might want to automatically save other stuff.
class ProtocolOperation:
    """ """

    DEFAULT_MAX_ATTEMPTS = 3  # Default max retry attempts for operations

    def __init__(self):
        global PLATFORMTYPE

        self.name = self.__class__.__name__

        self.platform_type: PlatformTypes = PLATFORMTYPE
        self.data_loc: Path | None = None

        self.input_params: dict[str, ProtocolParameterBase] = {}
        self.output_params: dict[str, ProtocolParameterBase] = {}

        # Specifies in plain english what is the condition for the measurement to be successful
        self.condition: str = ""
        self.report_output: list[str | Path] = []

        self.independents: dict[str, ArrayLike] = {}
        self.dependents: dict[str, ArrayLike] = {}

        self.figure_paths: list[Path] = []

        self.improvements: list[ParamImprovement] = []

        # Retry/attempt tracking
        self.max_attempts: int = self.DEFAULT_MAX_ATTEMPTS
        self.current_attempt: int = 0
        self.total_attempts_made: int = 0

    def _register_inputs(self, **kwargs):
        """Register input parameters as both attributes and in the dictionary"""
        for name, param in kwargs.items():
            setattr(self, name, param)
            self.input_params[name] = param

    def _register_outputs(self, **kwargs):
        """Register output parameters as both attributes and in the dictionary"""
        for name, param in kwargs.items():
            setattr(self, name, param)
            self.output_params[name] = param

    def _measure_qick(self) -> Path:
        raise NotImplementedError("QICK measurement not implemented")

    def _measure_opx(self) -> Path:
        raise NotImplementedError("OPX measurement not implemented")

    def _measure_dummy(self) -> Path:
        raise NotImplementedError("DUMMY measurement not implemented")

    # TODO: How do we verify directionality in the information with datasets that are 2D or bigger.
    #  e.i.: How do we know which independent is store in which axis? This needs to be standardized for each measurement
    def _verify_shape(self) -> bool:
        # Check if any arrays in independents are empty
        for name, array in self.independents.items():
            arr = np.asarray(array)
            if arr.size == 0:
                print(f"independents['{name}'] is empty")
                return False

        # Check if any arrays in dependents are empty
        for name, array in self.dependents.items():
            arr = np.asarray(array)
            if arr.size == 0:
                print(f"dependents['{name}'] is empty")
                return False

        all_arrays = {}
        all_arrays.update(self.independents)
        all_arrays.update(self.dependents)

        if not all_arrays:
            return True

        shapes = {}
        for name, array in all_arrays.items():
            arr = np.asarray(array)
            shapes[name] = arr.shape

        first_shape = next(iter(shapes.values()))

        if all(shape == first_shape for shape in shapes.values()):
            return True
        else:
            print("Shape mismatch detected:")
            for name, shape in shapes.items():
                print(f"  {name}: {shape}")
            return False

    def measure(self):
        match self.platform_type:
            case PlatformTypes.QICK:
                loc = self._measure_qick()
                self.data_loc = loc
                return loc
            case PlatformTypes.OPX:
                loc = self._measure_opx()
                self.data_loc = loc
                return loc
            case PlatformTypes.DUMMY:
                loc = self._measure_dummy()
                self.data_loc = loc
                return loc
        raise NotImplementedError(f"Platform type {self.platform_type} not implemented")

    def analyze(self):
        raise NotImplementedError("Analyze method not implemented")

    def _load_data_opx(self):
        raise NotImplementedError("Load OPX data method not implemented")

    def _load_data_qick(self):
        raise NotImplementedError("Load QICK data method not implemented")

    def _load_data_dummy(self):
        raise NotImplementedError("Load DUMMY data method not implemented")

    def load_data(self):
        match self.platform_type:
            case PlatformTypes.QICK:
                self._load_data_qick()
            case PlatformTypes.OPX:
                self._load_data_opx()
            case PlatformTypes.DUMMY:
                self._load_data_dummy()
            case _:
                raise NotImplementedError(
                    f"Platform type {self.platform_type} not implemented"
                )

        return self._verify_shape()

    def evaluate(self) -> OperationStatus:
        """
        Evaluate operation results and recommend next action.

        Subclasses must implement custom logic based on their domain knowledge.

        Returns:
            OperationStatus.SUCCESS: Proceed to next operation
            OperationStatus.RETRY: Retry this operation (if attempts remain)
            OperationStatus.FAILURE: Stop protocol execution
        """
        raise NotImplementedError("Subclasses must implement evaluate()")

    def execute(self) -> OperationStatus:
        """
        Execute the full operation workflow: measure -> load_data -> analyze -> evaluate.

        This method increments attempt counters and adds repetition headers to reports.

        Returns:
            OperationStatus from evaluate() method
        """
        # Increment attempt counter
        self.current_attempt += 1
        self.total_attempts_made += 1

        # Add repetition header to report if this is a retry
        if self.current_attempt > 1:
            repetition_header = f"### ATTEMPT {self.current_attempt}\n\n"
            self.report_output.append(repetition_header)

        # Execute the four-step workflow
        self.measure()
        self.load_data()
        self.analyze()
        status = self.evaluate()

        return status


class SuperOperationBase(ProtocolOperation):
    """
    A composite operation that groups multiple operations together.

    SuperOperations execute a sequence of operations as a single unit,
    sharing the same retry mechanism. If any sub-operation fails, the
    entire SuperOperation can be retried.

    Key features:
    - All sub-operations execute in sequence
    - Retry logic applies to the entire group
    - Reports are aggregated under the SuperOperation section
    - Branching (Conditions) is NOT permitted in SuperOperations
    - Protocol treats it the same as a regular operation

    Subclasses must:
    1. Call super().__init__() in their __init__
    2. Set self.operations to a list of ProtocolOperation instances
    3. NOT include any Condition instances in self.operations

    Example:
        >>> class CalibrationSuite(SuperOperationBase):
        ...     def __init__(self, params):
        ...         super().__init__()
        ...         self.operations = [
        ...             ResonatorSpectroscopy(params),
        ...             PowerRabi(params),
        ...             PiSpectroscopy(params)
        ...         ]
        ...
        ...     def evaluate(self) -> OperationStatus:
        ...         # All operations succeeded, check aggregate quality
        ...         if all_calibrations_good():
        ...             return OperationStatus.SUCCESS
        ...         else:
        ...             return OperationStatus.RETRY
    """

    def __init__(self):
        super().__init__()
        self.operations: list[ProtocolOperation] = []

    def _validate_operations(self):
        """Validate that operations list contains only ProtocolOperation instances"""

        for i, op in enumerate(self.operations):
            if isinstance(op, Condition):
                raise ValueError(
                    f"SuperOperation '{self.name}' contains a Condition at index {i}. "
                    f"Branching is not permitted in SuperOperations. "
                    f"Use regular branches in the protocol instead."
                )
            if not isinstance(op, ProtocolOperation):
                raise TypeError(
                    f"SuperOperation '{self.name}' contains invalid item at index {i}: {type(op)}. "
                    f"Only ProtocolOperation instances are allowed."
                )

    def execute(self) -> OperationStatus:
        """
        Execute all sub-operations in sequence and aggregate their reports.

        This method overrides ProtocolOperation.execute() to iterate through
        all sub-operations instead of calling measure/load_data/analyze.

        Returns:
            OperationStatus from the evaluate() method
        """
        # Validate operations before executing
        self._validate_operations()

        # Increment attempt counter
        self.current_attempt += 1
        self.total_attempts_made += 1

        # Add retry header if needed
        if self.current_attempt > 1:
            repetition_header = f"### ATTEMPT {self.current_attempt}\n\n"
            self.report_output.append(repetition_header)

        # Add SuperOperation header to report
        header = f"## {self.name}\n\n"
        self.report_output.append(header)

        # Execute each sub-operation
        for i, op in enumerate(self.operations):
            logger.info(
                f"  [{self.name}] Executing sub-operation {i + 1}/{len(self.operations)}: {op.name}"
            )

            # Execute the operation (measure -> load_data -> analyze -> evaluate)
            try:
                status = op.execute()
            except Exception as e:
                logger.error(
                    f"  [{self.name}] Exception in sub-operation {op.name}: {e}"
                )
                # If a sub-operation fails, the SuperOperation fails
                return OperationStatus.FAILURE

            # Aggregate the sub-operation's report output
            if op.report_output:
                # Add sub-operation section header
                self.report_output.append(f"### {op.name}\n\n")
                # Add all report items from the sub-operation
                self.report_output.extend(op.report_output)
                self.report_output.append("\n")

            # Check sub-operation status
            if status == OperationStatus.FAILURE:
                logger.error(
                    f"  [{self.name}] Sub-operation {op.name} failed critically"
                )
                return OperationStatus.FAILURE
            elif status == OperationStatus.RETRY:
                logger.warning(
                    f"  [{self.name}] Sub-operation {op.name} requested retry"
                )
                # Don't immediately fail - let evaluate() decide
                # But we could track this for evaluation logic
            elif status == OperationStatus.SUCCESS:
                logger.info(f"  [{self.name}] Sub-operation {op.name} succeeded")

            # Aggregate figure paths
            if hasattr(op, "figure_paths"):
                self.figure_paths.extend(op.figure_paths)

            # Aggregate improvements
            if hasattr(op, "improvements"):
                self.improvements.extend(op.improvements)

        # Call the subclass's evaluate() method to determine overall status
        status = self.evaluate()

        return status

    def measure(self):
        """Not used in SuperOperation - operations handle their own measurement"""
        raise NotImplementedError(
            f"measure() does not make sense for a SuperOperation. "
            f"Sub-operations in '{self.name}' handle their own measurement."
        )

    def load_data(self):
        """Not used in SuperOperation - operations handle their own data loading"""
        raise NotImplementedError(
            f"load_data() does not make sense for a SuperOperation. "
            f"Sub-operations in '{self.name}' handle their own data loading."
        )

    def analyze(self):
        """Not used in SuperOperation - operations handle their own analysis"""
        raise NotImplementedError(
            f"analyze() does not make sense for a SuperOperation. "
            f"Sub-operations in '{self.name}' handle their own analysis."
        )


class ProtocolBase:
    def __init__(self, report_path: Path = Path("")):

        self.name = self.__class__.__name__
        self.root_branch = None  # Required - must be set by subclass
        # True for successful protocol execution, False for failure at some operation, None for un-ran protocol
        self.success: bool | None = None

        self.report_path = report_path

        # Don't run if the user didn't select a platform
        if PLATFORMTYPE is None:
            raise ValueError("Please choose a platform")

    def _flatten_branch_for_execution(self, branch):
        """
        Recursively flatten a branch into a list of operations and conditions.

        Does NOT evaluate conditions - just collects them for runtime evaluation.
        Returns a list containing both ProtocolOperation and Condition instances.
        """
        items = []

        for item in branch.items:
            if isinstance(item, ProtocolOperation):
                items.append(item)
            elif isinstance(item, Condition):
                # Add the condition itself (will be evaluated during execution)
                items.append(item)
                # Don't traverse into branches yet - will be done at runtime

        return items

    def _collect_all_operations_from_branch(self, branch):
        """
        Recursively collect ALL operations from a branch tree (for parameter verification).

        Includes operations from all branches, not just taken ones.
        Also collects operations from inside SuperOperations.
        """

        operations = []

        for item in branch.items:
            if isinstance(item, SuperOperationBase):
                # Add the SuperOperation itself
                operations.append(item)
                # Also collect all sub-operations from the SuperOperation
                for sub_op in item.operations:
                    operations.append(sub_op)
            elif isinstance(item, ProtocolOperation):
                operations.append(item)
            elif isinstance(item, Condition):
                # Collect from BOTH branches
                operations.extend(
                    self._collect_all_operations_from_branch(item.true_branch)
                )
                operations.extend(
                    self._collect_all_operations_from_branch(item.false_branch)
                )

        return operations

    def verify_all_parameters(self) -> bool:
        """Verify parameters in all operations across all branches"""

        if self.root_branch is None:
            raise ValueError(
                f"Protocol {self.name} must set self.root_branch in __init__"
            )

        all_ops = self._collect_all_operations_from_branch(self.root_branch)

        failures = {}
        for op in all_ops:
            for param_name, param in op.input_params.items():
                try:
                    val = param()  # Use callable syntax to verify parameter access
                except Exception as e:
                    failures[param.name] = e

            for param_name, param in op.output_params.items():
                try:
                    val = param()  # Use callable syntax to verify parameter access
                except Exception as e:
                    failures[param.name] = e

        if failures:
            f_list = [f"{str(k)}: {str(v)}" for k, v in failures.items()]
            msg = f"The following parameters could not be verified: {f_list}"
            raise AttributeError(msg)

        return True

    def _assemble_report(self):
        """Generate HTML report from executed operations and conditions with embedded images"""
        # Create report directory structure
        report_dir = self.report_path / f"{self.name}_report"
        report_dir.mkdir(exist_ok=True)

        # Use the executed_items collected during execution
        if not hasattr(self, "executed_items"):
            logger.warning("No executed items found for report")
            executed_items = []
        else:
            executed_items = self.executed_items

        # Build table of contents and sections
        toc_entries = []
        sections = []

        for idx, item in enumerate(executed_items):
            section_id = f"section-{idx}"

            if isinstance(item, Condition):
                # Create section for condition
                section_title = f"Condition: {item.name}"
                toc_entries.append((section_id, section_title, "condition"))

                # Build section content
                section_content = []
                if item.report_output:
                    for report_item in item.report_output:
                        section_content.append(f"{report_item}\n")

                sections.append(
                    {
                        "id": section_id,
                        "title": section_title,
                        "type": "condition",
                        "content": "\n".join(section_content),
                    }
                )

            elif isinstance(item, ProtocolOperation):
                # Create section for operation
                section_title = f"Operation: {item.name}"
                toc_entries.append((section_id, section_title, "operation"))

                # Build section content
                section_content = []
                if item.report_output:
                    for report_item in item.report_output:
                        if isinstance(report_item, Path):
                            # Embed image as base64 data URI
                            try:
                                with open(report_item, "rb") as img_file:
                                    img_data = img_file.read()
                                    img_base64 = base64.b64encode(img_data).decode(
                                        "utf-8"
                                    )

                                # Determine MIME type from file extension
                                ext = report_item.suffix.lower()
                                mime_type = {
                                    ".png": "image/png",
                                    ".jpg": "image/jpeg",
                                    ".jpeg": "image/jpeg",
                                    ".gif": "image/gif",
                                    ".svg": "image/svg+xml",
                                    ".webp": "image/webp",
                                }.get(ext, "image/png")

                                # Add as markdown image with data URI
                                section_content.append(
                                    f"![Figure](data:{mime_type};base64,{img_base64})\n"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to embed image {report_item}: {e}"
                                )
                                section_content.append(
                                    f"![Figure - Error loading image]({report_item.name})\n"
                                )
                        else:
                            section_content.append(f"{report_item}\n")

                sections.append(
                    {
                        "id": section_id,
                        "title": section_title,
                        "type": "operation",
                        "content": "\n".join(section_content),
                    }
                )

        # Build HTML with table of contents and sections
        html_parts = []

        # Add title
        html_parts.append(f"<h1>{self.name} Report</h1>\n")

        # Add table of contents
        html_parts.append("<h2>Table of Contents</h2>\n")
        html_parts.append("<ul class='toc'>\n")
        for section_id, title, item_type in toc_entries:
            css_class = "toc-condition" if item_type == "condition" else "toc-operation"
            html_parts.append(
                f"  <li class='{css_class}'><a href='#{section_id}'>{title}</a></li>\n"
            )
        html_parts.append("</ul>\n")
        html_parts.append("<hr>\n")

        # Add sections
        for section in sections:
            html_parts.append(
                f"<section id='{section['id']}' class='report-section {section['type']}'>\n"
            )
            html_parts.append(
                f"<h2 class='section-header' onclick='toggleSection(\"{section['id']}-content\")'>\n"
            )
            html_parts.append(
                f"<span class='toggle-icon'>▼</span> {section['title']}\n"
            )
            html_parts.append("</h2>\n")

            # Wrappable content div
            html_parts.append(
                f"<div id='{section['id']}-content' class='section-content'>\n"
            )

            # Convert markdown content to HTML
            section_html = markdown.markdown(
                section["content"], extensions=["extra", "codehilite"]
            )
            html_parts.append(section_html)

            html_parts.append("<a href='#' class='back-to-top'>↑ Back to top</a>\n")
            html_parts.append("</div>\n")  # Close section-content
            html_parts.append("</section>\n")
            html_parts.append("<hr>\n")

        html_body = "\n".join(html_parts)

        # Create complete HTML with enhanced CSS
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.name} Report</title>
    <style>
        body {{
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            font-family: sans-serif;
            line-height: 1.6;
        }}

        h1 {{
            border-bottom: 3px solid #333;
            padding-bottom: 10px;
        }}

        h2 {{
            margin-top: 30px;
        }}

        /* Table of contents */
        .toc {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            list-style-position: inside;
        }}

        .toc li {{
            margin: 8px 0;
        }}

        .toc a {{
            text-decoration: none;
            color: #0066cc;
        }}

        .toc a:hover {{
            text-decoration: underline;
        }}

        .toc-condition {{
            font-style: italic;
            color: #666;
        }}

        .toc-operation {{
            font-weight: 500;
        }}

        /* Sections */
        .report-section {{
            margin: 40px 0;
            padding: 20px;
            background-color: #ffffff;
            border-left: 4px solid #ddd;
        }}

        .report-section.condition {{
            border-left-color: #ff9800;
            background-color: #fff8f0;
        }}

        .report-section.operation {{
            border-left-color: #2196f3;
            background-color: #f0f8ff;
        }}

        .back-to-top {{
            display: inline-block;
            margin-top: 20px;
            text-decoration: none;
            color: #666;
            font-size: 14px;
        }}

        .back-to-top:hover {{
            color: #000;
        }}

        /* Images and code */
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
        }}

        pre {{
            background-color: #f5f5f5;
            padding: 10px;
            overflow-x: auto;
            border-radius: 4px;
        }}

        hr {{
            border: none;
            border-top: 1px solid #e0e0e0;
            margin: 40px 0;
        }}

        /* Collapsible sections */
        .section-header {{
            cursor: pointer;
            user-select: none;
            margin-top: 0;
        }}

        .section-header:hover {{
            opacity: 0.8;
        }}

        .toggle-icon {{
            display: inline-block;
            transition: transform 0.3s ease;
            font-size: 0.8em;
        }}

        .toggle-icon.collapsed {{
            transform: rotate(-90deg);
        }}

        .section-content {{
            overflow: hidden;
            transition: max-height 0.3s ease, opacity 0.3s ease;
            max-height: 10000px;
            opacity: 1;
        }}

        .section-content.collapsed {{
            max-height: 0;
            opacity: 0;
        }}
    </style>
    <script>
        function toggleSection(contentId) {{
            const content = document.getElementById(contentId);
            const icon = content.previousElementSibling.querySelector('.toggle-icon');

            if (content.classList.contains('collapsed')) {{
                content.classList.remove('collapsed');
                icon.classList.remove('collapsed');
            }} else {{
                content.classList.add('collapsed');
                icon.classList.add('collapsed');
            }}
        }}
    </script>
</head>
<body>
{html_body}
</body>
</html>
"""

        # Save HTML file
        html_file = report_dir / f"{self.name}_report.html"
        html_file.write_text(html_template)

        logger.info(f"Report generated: {html_file}")

        return html_file

    def _execute_operation(self, op: ProtocolOperation) -> bool:
        """
        Execute a single operation with retry logic.

        The operation's execute() method handles the workflow, this method handles retries.

        Returns:
            True if operation succeeded, False if failed
        """
        max_attempts = op.max_attempts

        # Reset attempt counter for this operation
        op.current_attempt = 0

        while op.current_attempt < max_attempts:
            # Execute operation (it will increment current_attempt internally)
            try:
                status = op.execute()
            except Exception as e:
                logger.error(f"  Exception during {op.name}: {e}")
                return False

            # Handle status
            if status == OperationStatus.SUCCESS:
                logger.info(f"  SUCCESS: {op.name} succeeded")
                return True

            elif status == OperationStatus.RETRY:
                if op.current_attempt < max_attempts:
                    logger.warning(
                        f"  RETRY: {op.name} requesting retry (attempt {op.current_attempt}/{max_attempts})"
                    )
                    continue  # Retry
                else:
                    logger.error(
                        f"  FAILURE: {op.name} exhausted {max_attempts} attempts"
                    )
                    return False

            elif status == OperationStatus.FAILURE:
                logger.error(f"  FAILURE: {op.name} failed critically")
                return False

            else:
                logger.error(f"  Unknown status: {status}")
                return False

        # Should not reach here
        return False

    def _execute_branch(self, branch):
        """
        Recursively execute a branch, evaluating conditions at runtime.

        Returns list of executed items (operations and conditions) for reporting.
        """

        executed_items = []

        for item in branch.items:
            if isinstance(item, ProtocolOperation):
                logger.info(f"  Executing: {item.name}")
                success = self._execute_operation(item)
                executed_items.append(item)

                if not success:
                    return executed_items, False

            elif isinstance(item, Condition):
                taken_branch = item.evaluate()
                executed_items.append(item)

                sub_items, success = self._execute_branch(taken_branch)
                executed_items.extend(sub_items)

                if not success:
                    return executed_items, False

        return executed_items, True

    def execute(self):
        """Execute protocol by recursively executing branches"""
        logger.info(f"Starting protocol: {self.name}")

        if self.root_branch is None:
            raise ValueError(
                f"Protocol {self.name} must set self.root_branch in __init__"
            )

        # Verify all parameters before starting
        if not self.verify_all_parameters():
            logger.error("Parameter verification failed")
            self.success = False
            return

        # Execute the root branch (conditions evaluated at runtime)
        logger.info(f"Executing protocol with root branch: '{self.root_branch.name}'")
        self.executed_items, success = self._execute_branch(self.root_branch)

        if not success:
            logger.error("Protocol stopped due to operation failure")
            self.success = False
        else:
            logger.info("Protocol completed successfully")
            self.success = True

        self._assemble_report()


class BranchBase:
    """
    A named sequence of operations that can be executed.

    Branches can contain:
    - Operations (ProtocolOperation instances)
    - Conditions (conditional routing to other branches)

    Args:
        name: Display name for this branch

    Example:
        >>> main_branch = BranchBase("MainCalibration")
        >>> main_branch.append(ResonatorSpectroscopy(params))
        >>> main_branch.append(SaturationSpectroscopy(params))
    """

    def __init__(self, name: str = "Branch"):
        self.name = name
        self.items: list = []  # Will contain ProtocolOperation or Condition instances

    def append(self, item):
        """Add an operation or condition to this branch"""
        self.items.append(item)
        return self  # Allow chaining

    def extend(self, items: list):
        """Add multiple operations/conditions to this branch"""
        self.items.extend(items)
        return self  # Allow chaining

    def __repr__(self):
        return f"BranchBase(name='{self.name}', items={len(self.items)})"


class Condition:
    """
    A conditional decision point that routes execution to different branches.

    During protocol execution, the condition is evaluated and either
    true_branch or false_branch is executed.

    Args:
        condition: Callable returning bool to determine which branch
        true_branch: Branch to execute if condition is True
        false_branch: Branch to execute if condition is False
        name: Optional name for this condition (for logging/reporting)

    Example:
        >>> res_snr = params['resonator']['snr']
        >>>
        >>> high_snr_branch = BranchBase("HighSNR")
        >>> high_snr_branch.append(PiSpectroscopy(params))
        >>>
        >>> low_snr_branch = BranchBase("LowSNR")
        >>> low_snr_branch.append(PowerRabi(params))
        >>> low_snr_branch.append(PiSpectroscopy(params))
        >>>
        >>> condition = Condition(
        >>>     condition=lambda: res_snr() > 5.0,
        >>>     true_branch=high_snr_branch,
        >>>     false_branch=low_snr_branch,
        >>>     name="SNR Check"
        >>> )
    """

    def __init__(
        self,
        condition: Callable[[], bool],
        true_branch: BranchBase,
        false_branch: BranchBase,
        name: str = "Condition",
    ):
        self.condition_func = condition
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.name = name

        # Runtime state
        self.condition_result: bool | None = None
        self.taken_branch: BranchBase | None = None
        self.report_output: list[str] = []

    def evaluate(self) -> BranchBase:
        """Evaluate condition and return the branch to execute"""
        logger.info(f"[Condition] Evaluating condition: '{self.name}'")
        logger.debug(
            f"[Condition] Available branches: TRUE → '{self.true_branch.name}', FALSE → '{self.false_branch.name}'"
        )

        self.condition_result = self.condition_func()
        logger.info(f"[Condition] '{self.name}' evaluated to: {self.condition_result}")

        if self.condition_result:
            logger.info(
                f"[Condition] '{self.name}': TRUE → executing branch '{self.true_branch.name}'"
            )
            self.taken_branch = self.true_branch
            message = f"**Condition '{self.name}' has the result {self.condition_result} (TRUE branch). Choosing branch '{self.true_branch.name}'**"
            self.report_output.append(message)
            return self.true_branch
        else:
            logger.info(
                f"[Condition] '{self.name}': FALSE → executing branch '{self.false_branch.name}'"
            )
            self.taken_branch = self.false_branch
            message = f"**Condition '{self.name}' has the result {self.condition_result} (FALSE branch). Choosing branch '{self.false_branch.name}'**"
            self.report_output.append(message)
            return self.false_branch

    def __repr__(self):
        return f"Condition(name='{self.name}')"
