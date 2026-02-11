from dataclasses import dataclass, field
from typing import Optional, Iterable, List, Callable, Iterator, Tuple, Union, \
    Any, Dict
import inspect
from functools import update_wrapper
import copy
import collections
from enum import Enum
import logging

try:
    from qcodes import Parameter as QCParameter
    QCODES_PRESENT = True
except ImportError:
    QCParameter = None
    QCODES_PRESENT = False

from ..utils.misc import map_input_to_signature


logger = logging.getLogger(__name__)


class DataType(Enum):
    """Valid options for data types used in :class:`DataSpec`"""
    #: scalar (single-valued) data. typically numeric, but also bool, etc.
    scalar = 'scalar'
    #: multi-valued data. typically numpy-arrays.
    array = 'array'


@dataclass
class DataSpec:
    """Specification for data parameters to be recorded."""
    #: name of the parameter
    name: str
    #: dependencies. if ``None``, it is independent.
    depends_on: Union[None, List[str], Tuple[str]] = None
    #: information about data format
    type: Union[str, DataType] = 'scalar'
    #: physical unit of the data
    unit: str = ''

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = DataType(self.type)

    def copy(self) -> "DataSpec":
        """return a deep copy of the DataSpec instance."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        ret = self.name
        if self.depends_on is not None and len(self.depends_on) > 0:
            ret += f"({', '.join(list(self.depends_on))})"
        return ret


#: shorter notation for constructing DataSpec objects
ds = DataSpec
#: The type for creating a ds from a tuple (i.e., what can be passed to the
#: constructor of :class:`.DataSpec`)
DataSpecFromTupleType = Tuple[str, Union[None, List[str], Tuple[str]], str,
                              str]
#: The type for creating a ds from a dict (i.e., what can be passed to the
#: constructor of :class:`.DataSpec` as keywords)
DataSpecFromDictType = Dict[str, Union[str, Union[None, List[str], Tuple[str]]]]
#: The type from which we can create a DataSpec.
DataSpecCreationType = Union[str, DataSpecFromTupleType,
                             DataSpecFromDictType, DataSpec]


def data_specs_label(*dspecs: DataSpec) -> str:
    """Create a readable label for multiple data specs.

    Format:
        {data_name_1 (dep_1, dep_2), data_name_2 (dep_3), etc.}

    :param dspecs: data specs as positional arguments.
    :return: label as string.
    """
    return r"{" + f"{', '.join([d.__repr__() for d in dspecs])}" + r"}"


def make_data_spec(value: DataSpecCreationType) -> DataSpec:
    """Instantiate a DataSpec object.

    :param value:
        May be one of the following with the following behavior:

            - A string create a dependent with name given by the string
            - A tuple of values that can be used to pass to the constructor of :class:`.DataSpec`
            - A dictionary entries of which will be passed as keyword arguments to the constructor of :class:`.DataSpec`
            - A :class:`.DataSpec` instance

    """
    if isinstance(value, str):
        return dependent(value)
    elif isinstance(value, (tuple, list)):
        return DataSpec(*value)
    elif isinstance(value, dict):
        return DataSpec(**value)
    elif isinstance(value, DataSpec):
        return value
    else:
        raise TypeError(f"Cannot create DataSpec from {type(value)}")


def make_data_specs(*specs: DataSpecCreationType) -> Tuple[DataSpec, ...]:
    """Create a tuple of DataSpec instances.

    :param specs: will be passed individually to :func:`.make_data_spec`
    """
    ret = []
    for spec in specs:
        ret.append(make_data_spec(spec))
    ret = tuple(ret)
    return ret


def combine_data_specs(*specs: DataSpec) -> Tuple[DataSpec, ...]:
    """Create a tuple of DataSpecs from the inputs. Removes duplicates."""
    ret = []
    spec_names = []
    for s in specs:
        if s.name not in spec_names:
            ret.append(s)
            spec_names.append(s.name)

    return tuple(ret)


def independent(name: str, unit: str = '', type: str = 'scalar') -> DataSpec:
    """Create a the spec for an independent parameter.
    All arguments are forwarded to the :class:`.DataSpec` constructor.
    ``depends_on`` is set to ``None``."""
    return DataSpec(name, unit=unit, type=type, depends_on=None)


indep = independent


def dependent(name: str, depends_on: List[str] = [], unit: str = "",
              type: str = 'scalar'):
    """Create a the spec for a dependent parameter.
    All arguments are forwarded to the :class:`.DataSpec` constructor.
    ``depends_on`` may not be set to ``None``."""
    if depends_on is None:
        raise TypeError("'depends_on' may not be None for a dependent.")
    return DataSpec(name, unit=unit, type=type, depends_on=depends_on)


dep = dependent


def recording(*data_specs: DataSpecCreationType) -> Callable:
    """Returns a decorator that allows adding data parameter specs to a
    function.
    """
    def decorator(func):
        return FunctionToRecords(func, *make_data_specs(*data_specs))
    return decorator


def record_as(obj: Union[Callable, Iterable, Iterator],
              *specs: DataSpecCreationType):
    """Annotate produced data as records.

    :param obj: a function that returns data or an iterable/iterator that
        produces data at each iteration step
    :param specs: specs for the data produced (see :func:`.make_data_specs`)
    """
    specs = make_data_specs(*specs)
    if isinstance(obj, Callable):
        return recording(*specs)(obj)
    elif isinstance(obj, collections.abc.Iterable):
        return IteratorToRecords(obj, *specs)


def produces_record(obj: Any) -> bool:
    """Check if `obj` is annotated to generate records."""
    if hasattr(obj, 'get_data_specs'):
        return True
    else:
        return False


def _to_record(value: Union[Dict, Iterable],
               data_specs: Tuple[DataSpec]) -> Dict[str, Any]:
    """Convert data to a record using the provided DataSpecs"""
    ret = {}

    if isinstance(value, dict):
        for s in data_specs:
            ret[s.name] = value.get(s.name, None)

    elif isinstance(value, collections.abc.Iterator):
        ret = IteratorToRecords(value, *data_specs)

    else:
        if not isinstance(value, (tuple, list)):
            value = [value]
        for i, s in enumerate(data_specs):
            try:
                ret[s.name] = value[i]
            except IndexError:
                ret[s.name] = None

    return ret


class IteratorToRecords:
    """A wrapper that converts the iteration values to records."""

    def __init__(self, iterable: Iterable,
                 *data_specs: DataSpecCreationType):
        self.iterable = iterable
        self.data_specs = make_data_specs(*data_specs)

    def get_data_specs(self):
        return self.data_specs

    def __iter__(self):
        for val in self.iterable:
            yield _to_record(val, self.data_specs)

    def __repr__(self):
        from .sweep import CombineSweeps

        ret = self.iterable.__repr__()
        if not isinstance(self.iterable, CombineSweeps):
            dnames = data_specs_label(*self.get_data_specs())
            ret += f" as {dnames}"

        return ret


class FunctionToRecords:
    """A wrapper that converts a function return to a record."""

    def __init__(self, func, *data_specs):
        self.func = func
        self.func_sig = inspect.signature(self.func)
        self.data_specs = make_data_specs(*data_specs)
        update_wrapper(self, func)

        self._args: List[Any] = []
        self._kwargs: Dict[str, Any] = {}

    def get_data_specs(self):
        return self.data_specs

    def __call__(self, *args, **kwargs):
        args = tuple(self._args + list(args))
        kwargs.update(self._kwargs)
        func_args, func_kwargs = map_input_to_signature(self.func_sig,
                                                        *args, **kwargs)
        ret = self.func(*func_args, **func_kwargs)
        return _to_record(ret, self.get_data_specs())

    def __repr__(self):
        dnames = data_specs_label(*self.data_specs)
        ret = self.func.__name__ + str(self.func_sig)
        ret += f" as {dnames}"
        return ret

    def using(self, *args, **kwargs) -> "FunctionToRecords":
        """Set the default positional and keyword arguments that will be
        used when the function is called.

        :returns: a copy of the object. This is to allow setting different
            defaults to multiple uses of the function.
        """
        ret = copy.copy(self)
        ret._args = list(args)
        ret._kwargs = kwargs
        return ret


# TODO: support for qcodes parameters as actions. should automatically
#   inherit shapes, dependencies (for ParameterWithSetPoints, for example)
#   needs a function to make data specs from parameters (incl some user
#   customization, like setting to array for regular parameters)
def get_parameter(param: QCParameter):
    if not QCODES_PRESENT:
        raise RuntimeError("qcodes not found.")

    return record_as(param.get, dependent(param.name, unit=param.unit))
