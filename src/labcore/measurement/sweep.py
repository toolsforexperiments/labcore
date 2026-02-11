import itertools
import inspect
from typing import Iterable, Callable, Union, Tuple, Any, Optional, Dict, List, Generator
import collections
import logging
from functools import wraps, update_wrapper, partial
import copy

try:
    from qcodes import Parameter as QCParameter
    QCODES_PRESENT = True
except ImportError:
    QCParameter = None
    QCODES_PRESENT = False

from .record import produces_record, DataSpec, IteratorToRecords, \
    DataSpecFromTupleType, record_as, combine_data_specs, independent, \
    make_data_spec, data_specs_label, DataSpecCreationType, make_data_specs, \
    FunctionToRecords, map_input_to_signature
from ..utils.misc import indent_text


logger = logging.getLogger(__name__)


if QCODES_PRESENT:
    ParamSpecType = Union[str, QCParameter, DataSpecFromTupleType, DataSpec]
else:
    ParamSpecType = Union[str, DataSpecFromTupleType, DataSpec]


# Pointer tools
class PointerFunction(FunctionToRecords):
    """A class that allows using a generator function as a pointer."""

    def _iterator2records(self, *args, **kwargs):
        func_args, func_kwargs = map_input_to_signature(self.func_sig,
                                                        *args, **kwargs)
        ret = record_as(self.func(*func_args, **func_kwargs), *self.data_specs)
        return ret

    def __call__(self, *args, **kwargs):
        args = tuple(self._args + list(args))
        kwargs.update(self._kwargs)
        return self._iterator2records(*args, **kwargs)

    def __iter__(self):
        return iter(self._iterator2records(*self._args, **self._kwargs))

    def get_data_specs(self):
        return self.data_specs

    def using(self, *args, **kwargs) -> "PointerFunction":
        """Set the default positional and keyword arguments that will be
        used when the function is called.

        :returns: A copy of the object. This is to allow setting different
            defaults to multiple uses of the function.
        """
        ret = copy.copy(self)
        ret._args = list(args)
        ret._kwargs = kwargs
        return ret


def pointer(*data_specs: DataSpecCreationType) -> Callable:
    """Create a decorator for functions that return pointer generators."""
    def decorator(func: Callable) -> PointerFunction:
        return PointerFunction(func, *data_specs)
    return decorator


def as_pointer(fun: Callable, *data_specs: DataSpecCreationType) -> PointerFunction:
    """Convenient in-line creation of a pointer function."""
    return pointer(*data_specs)(fun)


null_pointer = [None]


# sweep tools
def once(action: Callable) -> "Sweep":
    """Return a sweep that executes the action once."""
    return Sweep(null_pointer, action)


def sweep_parameter(param: ParamSpecType, sweep_iterable: Iterable,
                    *actions: Callable) -> "Sweep":
    """Create a sweep over a parameter.

    :param param: One of:

        * A string: Generates an independent, scalar data parameter.
        * A tuple or list: will be passed to the constructor of :class:`.DataSpec`; see :func:`.make_data_spec`.
        * A :class:`.DataSpec` instance.
        * A qcodes parameter. In this case the parameter's ``set`` method is called for each value during the iteration.

    :param sweep_iterable: An iterable that generates the values the parameter
        will be set to.
    :param actions: An arbitrary number of action functions.
    """

    if isinstance(param, str):
        param_ds = independent(param)
    elif isinstance(param, (tuple, list)):
        param_ds = make_data_spec(*param)
    elif isinstance(param, DataSpec):
        param_ds = param
    elif QCODES_PRESENT and isinstance(param, QCParameter):
        param_ds = independent(param.name, unit=param.unit)

        def setfunc(*args, **kwargs):
            param.set(kwargs.get(param.name))

        actions = list(actions)
        actions.insert(0, setfunc)
    else:
        raise TypeError(f"Cannot make parameter from type {type(param)}")

    record_iterator = IteratorToRecords(sweep_iterable, param_ds)
    return Sweep(record_iterator, *actions)


def null_action():
    return None


class Sweep:
    """Base class for sweeps.

    Can be iterated over; for each pointer value the associated actions are
    executed. Each iteration step produces a record, containing all values
    produced by pointer and actions that have been annotated as such.
    (see: :func:`.record_as`)

    :param pointer: An iterable that defines the steps over which we iterate
    :param actions: A variable number of functions. Each will be called for
        each iteration step, with the pointer value(s) as arguments, provided
        the function can accept it.
    """

    # TODO: (MAYBE?) should we check if there are conflicting record parameters?
    # TODO: some way of passing meta-info around (about the sweep state)
    #   probably nice to have some info on benchmarking, current indices, etc.
    # TODO: need a way to look through all subsweeps, for example to find all
    #   kw arguments of all actions recursively.
    # TODO: support qcodes parameters as action directly.

    # TODO: these flags should maybe be passed on to 'child' sweeps?
    record_none = True
    pass_on_returns = True
    pass_on_none = False

    # TODO: Add the rules.
    @staticmethod
    def update_option_dict(src: Dict[str, Any], target: Dict[str, Any], level: int) -> None:
        """Rules: work in progress :).
        """
        if not isinstance(src, dict) or not isinstance(target, dict):
            raise ValueError('inputs need to be dictionaries.')

        for k, v in src.items():
            if k in target:
                if isinstance(v, dict) and level > 0:
                    Sweep.update_option_dict(src[k], target[k], level=level-1)
            else:
                target[k] = v

    @staticmethod
    def propagate_sweep_options(sweep: "Sweep"):

        try:
            first = sweep.pointer.iterable.first
            Sweep.copy_sweep_options(sweep, first)
        except AttributeError:
            pass

        try:
            second = sweep.pointer.iterable.second
            Sweep.copy_sweep_options(sweep, second)
        except AttributeError:
            pass

    @staticmethod
    def copy_sweep_options(src: "Sweep", target: Optional["Sweep"]):
        if src is target:
            return

        Sweep.update_option_dict(src._action_kwargs, target._action_kwargs, level=2)
        Sweep.propagate_sweep_options(target)

    @staticmethod
    def link_sweep_properties(src: "Sweep", target: "Sweep") -> None:
        """Share state properties between sweeps."""
        for p in ['_state', '_pass_kwargs']:
            if hasattr(src, p):
                setattr(target, p, getattr(src, p))
                iterable = getattr(target.pointer, 'iterable', None)
                if iterable is not None and hasattr(iterable, 'first'):
                    first = getattr(iterable, 'first')
                    setattr(first, p, getattr(src, p))
                if iterable is not None and hasattr(iterable, 'second'):
                    second = getattr(iterable, 'second')
                    setattr(second, p, getattr(src, p))

        Sweep.copy_sweep_options(src, target)

    def __init__(self, pointer: Optional[Iterable], *actions: Callable):
        """Constructor of :class:`.Sweep`."""
        self._state = {}
        self._pass_kwargs = {}
        self._action_kwargs = {}

        if pointer is None:
            self.pointer = null_pointer
        elif isinstance(pointer, (collections.abc.Iterable, Sweep)):
            self.pointer = pointer
        else:
            raise TypeError('pointer needs to be iterable.')

        self.actions = []
        for a in actions:
            self.append_action(a)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value: Dict[str, Any]):
        for k, v in value.items():
            self._state[k] = v

    @property
    def pass_kwargs(self):
        return self._pass_kwargs

    @pass_kwargs.setter
    def pass_kwargs(self, value: Dict[str, Any]):
        for k, v in value.items():
            self._pass_kwargs[k] = v

    @property
    def action_kwargs(self):
        return self._action_kwargs

    @action_kwargs.setter
    def action_kwargs(self, value: Dict[str, Any]):
        for k, v in value.items():
            self._action_kwargs[k] = v

    def __iter__(self):
        return self.run()

    def __add__(self, other: Union[Callable, "Sweep"]) -> "Sweep":
        if isinstance(other, Sweep):
            sweep2 = other
        elif callable(other):
            sweep2 = Sweep(None, other)
        else:
            raise TypeError(f'can only combine with Sweep or callable, '
                            f'not {type(other)}')

        Sweep.link_sweep_properties(self, sweep2)
        return append_sweeps(self, sweep2)

    def __mul__(self, other: Union[Callable, "Sweep"]) -> "Sweep":
        if isinstance(other, Sweep):
            sweep2 = other
        elif callable(other):
            sweep2 = Sweep(self.pointer, other)
        else:
            raise TypeError(f'can only combine with Sweep or callable, '
                            f'not {type(other)}')

        Sweep.link_sweep_properties(self, sweep2)
        return zip_sweeps(self, sweep2)

    def __matmul__(self, other: Union[Callable, "Sweep"]) -> "Sweep":
        if isinstance(other, Sweep):
            sweep2 = other
        elif callable(other):
            sweep2 = Sweep(None, other)
        else:
            raise TypeError(f'can only combine with Sweep or callable, '
                            f'not {type(other)}')

        Sweep.link_sweep_properties(self, sweep2)
        return nest_sweeps(self, sweep2)

    def append_action(self, action: Callable):
        """Add an action to the sweep."""
        if callable(action):
            if produces_record(action):
                self.actions.append(action)
            else:
                self.actions.append(record_as(action))
        else:
            raise TypeError('action must be a callable.')

    def run(self) -> "SweepIterator":
        """Create the iterator for the sweep."""
        return SweepIterator(
            self,
            state=self.state,
            pass_kwargs=self.pass_kwargs,
            action_kwargs=self.action_kwargs)

    # FIXME: currently this only works for actions -- should be used also
    #   for pointer funcs?
    def set_options(self, **action_kwargs: Dict[str, Any]):
        """Configure the sweep actions.

        :param action_kwargs: Keyword arguments to pass to action functions
            format: {'<action_name>': {'key': 'value'}
            <action_name> is what action_function.__name__ returns.
        """
        for func, val in action_kwargs.items():
            self.action_kwargs[func] = val
            Sweep.propagate_sweep_options(self)

    def get_data_specs(self) -> Tuple[DataSpec, ...]:
        """Return the data specs of the sweep."""
        specs = []
        pointer_specs = []
        if produces_record(self.pointer):
            pointer_specs = self.pointer.get_data_specs()
            specs = combine_data_specs(*(list(specs) + list(pointer_specs)))

        for a in self.actions:
            if produces_record(a):
                action_specs = a.get_data_specs()
                pointer_independents = [ds.name for ds in pointer_specs
                                        if ds.depends_on is None]
                for aspec in action_specs:
                    aspec_ = aspec.copy()
                    if aspec_.depends_on is not None:
                        aspec_.depends_on = pointer_independents + aspec_.depends_on

                    specs = combine_data_specs(*(list(specs) + [aspec_]))

        return tuple(specs)

    def __repr__(self):
        ret = self.pointer.__repr__()
        for a in self.actions:
            ret += f" >> {a.__repr__()}"
        ret += f"\n==> {data_specs_label(*self.get_data_specs())}"
        return ret


class SweepIterator:
    """Iterator for the :class:`.Sweep` class.

    Manages the actual iteration of the pointer, and the execution of action
    functions. Manages and updates the state of the sweep.
    """

    def __init__(self, sweep: Sweep,
                 state: Dict[str, Any],
                 pass_kwargs=Dict[str, Any],
                 action_kwargs=Dict[str, Dict[str, Any]]):

        self.sweep = sweep
        self.state = state
        self.pass_kwargs = pass_kwargs
        self.action_kwargs = action_kwargs

        if isinstance(self.sweep.pointer, Sweep):
            self.pointer = iter(self.sweep.pointer)
        elif isinstance(self.sweep.pointer, collections.abc.Iterator):
            self.pointer = self.sweep.pointer
        elif isinstance(self.sweep.pointer, collections.abc.Iterable):
            self.pointer = iter(self.sweep.pointer)
        else:
            raise TypeError('pointer needs to be iterable.')

    def __next__(self):
        ret = {}
        next_point = next(self.pointer)
        if produces_record(self.sweep.pointer):
            ret.update(next_point)

        pass_args = []
        if self.sweep.pass_on_returns:
            if isinstance(next_point, (tuple, list)):
                if not self.sweep.pass_on_none:
                    pass_args = [r for r in next_point if r is not None]
                else:
                    pass_args = list(next_point)
            elif isinstance(next_point, dict):
                if not self.sweep.pass_on_none:
                    self.pass_kwargs.update({k: v for k, v in next_point.items()
                                             if v is not None})
                else:
                    self.pass_kwargs.update(next_point)
            else:
                if self.sweep.pass_on_none or next_point is not None:
                    pass_args.append(next_point)

        for a in self.sweep.actions:
            this_action_kwargs = {}
            if self.sweep.pass_on_returns:
                this_action_kwargs.update(self.pass_kwargs)
            this_action_kwargs.update(
                self.action_kwargs.get(a.__name__, {}))

            action_return = a(*pass_args, **this_action_kwargs)
            if produces_record(a):
                ret.update(action_return)

            # actions always return records, so no need to worry about args
            if not self.sweep.pass_on_none:
                self.pass_kwargs.update({k: v for k, v in action_return.items()
                                         if v is not None})
            else:
                self.pass_kwargs.update(action_return)

        if self.sweep.record_none is False:
            for k in list(ret.keys()):
                if ret[k] is None:
                    ret.pop(k)

        return ret


def append_sweeps(first: Sweep, second: Sweep) -> Sweep:
    """Append two sweeps.

    Iteration over the combined sweep will first complete the first sweep, then
    the second sweep.
    """
    both = IteratorToRecords(
        AppendSweeps(first, second),
        *combine_data_specs(*(list(first.get_data_specs())
                            + list(second.get_data_specs())))
    )
    sweep = Sweep(both)
    Sweep.link_sweep_properties(first, sweep)
    return sweep


def zip_sweeps(first: Sweep, second: Sweep) -> Sweep:
    """Zip two sweeps.

    Iteration over the combined sweep will elementwise advance the two sweeps
    together.
    """
    both = IteratorToRecords(
        ZipSweeps(first, second),
        *combine_data_specs(*(list(first.get_data_specs())
                            + list(second.get_data_specs())))
    )
    sweep = Sweep(both)
    Sweep.link_sweep_properties(first, sweep)
    return sweep


def nest_sweeps(outer: Sweep, inner: Sweep) -> Sweep:
    """Nest two sweeps.

    Iteration over the combined sweep will execute the full inner sweep for each
    iteration step of the outer sweep.
    """
    outer_specs = outer.get_data_specs()
    outer_indeps = [s.name for s in outer_specs if s.depends_on is None]

    inner_specs = [s.copy() for s in inner.get_data_specs()]
    for s in inner_specs:
        if s.depends_on is not None:
            s.depends_on = outer_indeps + s.depends_on

    nested = IteratorToRecords(
        NestSweeps(outer, inner),
        *combine_data_specs(*(list(outer_specs) + inner_specs))
    )
    sweep = Sweep(nested)
    Sweep.link_sweep_properties(outer, sweep)
    return sweep


class CombineSweeps:

    _operator_symbol = None

    def __init__(self, first: Sweep, second: Sweep):
        self.first = first
        self.second = second

    def __iter__(self):
        raise NotImplementedError

    def __repr__(self):
        ret = self.__class__.__name__ + ":\n"
        ret += indent_text(self.first.__repr__(), 4) + '\n'
        sym = ''
        if self._operator_symbol is not None:
            sym += self._operator_symbol + ' '
        sec_text = indent_text(self.second.__repr__(), 2)
        sec_text = sym + sec_text[len(sym):]
        ret += indent_text(sec_text, 4) + '\n'
        ret = ret.rstrip()
        while ret[-1] == "\n" and ret[-2] == "\n":
            ret = ret[:-1]
        return ret

    def get_data_specs(self):
        specs = list(self.first.get_data_specs()) + \
                list(self.second.get_data_specs())
        return combine_data_specs(*specs)


class ZipSweeps(CombineSweeps):

    _operator_symbol = '*'

    def __iter__(self):
        for fd, sd in zip(self.first, self.second):
            ret = fd.copy()
            ret.update(sd)
            yield ret


class AppendSweeps(CombineSweeps):

    _operator_symbol = '+'

    def __iter__(self):
        for ret in itertools.chain(self.first, self.second):
            yield ret


class NestSweeps(CombineSweeps):

    _operator_symbol = '@'

    def __iter__(self):
        for outer in self.first:
            for inner in self.second:
                ret = outer.copy()
                ret.update(inner)
                yield ret


class AsyncRecord:
    """
    Base class decorator used to record asynchronous data from instrument.
    Use the decorator with create_background_sweep function to create Sweeps that collect asynchronous data from
    external devices running experiments independently of the measurement PC,
    e.i. the measuring happening is not being controlled by a Sweep but instead an external device (e.g. the OPX).
    Each instrument should have its own custom setup_wrapper (see setup_wrapper docstring for more info),
    and a custom collector.
    Auxiliary functions for the start_wrapper and collector should also be located in this class.

    :param specs: A list of the DataSpecs to record the data produced.
    """

    wrapped_setup: Callable

    def __init__(self, *specs):
        self.specs = specs
        self.communicator = {}

    def __call__(self, fun) -> Callable:
        """
        When the decorator is called the experiment function gets wrapped so that it returns an Sweep object composed
        of 2 different Sweeps, the setup sweep and the collector Sweep.
        """

        def sweep(collector_options={}, **setup_kwargs) -> Sweep:
            """
            Returns a Sweep comprised of 2 different Sweeps: start_sweep and collector_sweep.
            start_sweep should perform any setup actions as well as starting the actual experiment. This sweep is only
            executed once. collector_sweep is iterated multiple time to collect all the data generated from the
            instrument.

            :param collector_kwargs: Any arguments that the collector needs.
            """
            start_sweep = once(self.wrap_setup(fun))
            collector_sweep = Sweep(as_pointer(self.collect, *self.specs).using(**collector_options))
            ret = start_sweep + collector_sweep
            ret.set_options(**{fun.__name__: setup_kwargs})
            return ret

        return sweep

    def wrap_setup(self, fun: Callable, *args: Any, **kwargs: Any) -> Callable:
        """
        Wraps the start function. setup_wrapper should consist of another function inside of it decorated with @wraps
        with fun as its argument.
        In this case the wrapped function is setup.
        Setup should accept the \*args and \**kwargs of fun. It should also place any returns from fun
        in the communicator. setup_wrapper needs to return the wrapped function (setup).

        :param fun: The measurement function. In the case of the OPX this would be the function that returns the QUA
                    code with any arguments that it might use.
        """
        self.wrapped_setup = partial(self.setup, fun, *args, **kwargs)
        update_wrapper(self.wrapped_setup, fun)
        return self.wrapped_setup

    def setup(self, fun, *args, **kwargs):
        return fun(*args, **kwargs)

    def collect(self, *args, **kwargs) -> Generator[Dict, None, None]:
        yield {}
