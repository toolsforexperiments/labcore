"""misc.py

Various utility functions.
"""

import logging
from enum import Enum
from pathlib import Path
from importlib.metadata import distributions
from typing import List, Tuple, TypeVar, Optional, Sequence, Any, Callable, Union
import inspect

from git import Repo, InvalidGitRepositoryError

logger = logging.getLogger(__name__)

def reorder_indices(lst: Sequence[str], target: Sequence[str]) -> Tuple[int, ...]:
    """
    Determine how to bring a list with unique entries to a different order.

    Supports only lists of strings.

    :param lst: input list
    :param target: list in the desired order
    :return: the indices that will reorder the input to obtain the target.
    :raises: ``ValueError`` for invalid inputs.
    """
    if set([type(i) for i in lst]) != {str}:
        raise ValueError('Only lists of strings are supported')
    if len(set(lst)) < len(lst):
        raise ValueError('Input list elements are not unique.')
    if set(lst) != set(target) or len(lst) != len(target):
        raise ValueError('Contents of input and target do not match.')

    idxs = []
    for elt in target:
        idxs.append(lst.index(elt))

    return tuple(idxs)


def reorder_indices_from_new_positions(lst: List[str], **pos: int) \
        -> Tuple[int, ...]:
    """
    Determine how to bring a list with unique entries to a different order.

    :param lst: input list (of strings)
    :param pos: new positions in the format ``element = new_position``.
                non-specified elements will be adjusted automatically.
    :return: the indices that will reorder the input to obtain the target.
    :raises: ``ValueError`` for invalid inputs.
    """
    if set([type(i) for i in lst]) != {str}:
        raise ValueError('Only lists of strings are supported')
    if len(set(lst)) < len(lst):
        raise ValueError('Input list elements are not unique.')

    target = lst.copy()
    for item, newidx in pos.items():
        oldidx = target.index(item)
        del target[oldidx]
        target.insert(newidx, item)

    return reorder_indices(lst, target)


T = TypeVar('T')


def unwrap_optional(val: Optional[T]) -> T:
    """Covert a variable of type Optional[T] to T
    If the variable has value None a ValueError will be raised
    """
    if val is None:
        raise ValueError("Expected a not None value but got a None value.")
    return val


class AutoEnum(Enum):
    """Enum that with automatically incremented integer values.

    Allows to pass additional arguments in the class variables to the __init__
    method of the instances.
    See: https://stackoverflow.com/questions/19330460/how-do-i-put-docstrings-on-enums/19330461#19330461
    """

    def __new__(cls, *args: Any) -> "AutoEnum":
        """creating a new instance.

        :param args: will be passed to __init__.
        """
        value = len(cls) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj


class LabeledOptions(AutoEnum):
    """Enum with a label for each element. We can find the name from the label
    using :meth:`.fromLabel`.

    Example::

            >>> class Color(LabeledOptions):
            ...     red = 'Red'
            ...     blue = 'Blue'

    Here, ``Color.blue`` has value ``2`` and ``Color.fromLabel('Blue')`` returns
    ``Color.blue``.
    """

    def __init__(self, label: str) -> None:
        self.label = label

    @classmethod
    def fromLabel(cls, label: str) -> Optional["LabeledOptions"]:
        """Find enum element from label."""
        for k in cls:
            if k.label.lower() == label.lower():
                return k
        return None


    # FIXME: 'None' should never overrides a default!
def map_input_to_signature(func: Union[Callable, inspect.Signature],
                           *args: Any, **kwargs: Any):
    """Try to re-organize the positional arguments `args` and key word
    arguments `kwargs` such that `func` can be called with them.

    if `func` expects arguments that cannot be given, they will be given
    as ``None``.
    Surplus arguments are ignored if `func` does not accept variable positional
    and/or keyword arguments.
    Example::

        >>> def myfunc(x, y, z=1):
        ...     print(f"x={x}, y={y}, z={z}")
        ...
        ... args, kwargs = map_input_to_signature(myfunc, z=1, x=1, unused=4)
        ... myfunc(*args, **kwargs)
        x=1, y=None, z=1

    It is important to note that the position of positional arguments is not
    preserved, because input key words that match expected positional arguments
    are inserted as positional arguments at the right position. The order,
    however, is preserved. Example::

        >>> def myfunc(x, y, z):
        ...     print(f"x={x}, y={y}, z={z}")
        ...
        ... args, kwargs = map_input_to_signature(myfunc, 1, 2, x=5)
        ... myfunc(*args, **kwargs)
        x=5, y=1, z=2
    """
    args = list(args)
    func_args = []
    func_kwargs = {}

    if isinstance(func, inspect.Signature):
        sig = func
    else:
        sig = inspect.signature(func)

    # Logic:
    # for each param the function expects, we need to check if have
    # received a fitting one
    for idx, p in enumerate(sig.parameters):
        p_ = sig.parameters[p]

        # we treat anything that can be given positionally as positional.
        # first prio to keyword-given values, second to positionally given,
        # finally default if given in signature.
        if p_.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD,
                       inspect.Parameter.POSITIONAL_ONLY]:
            if p in kwargs:
                func_args.insert(idx, kwargs.pop(p))
            else:
                if len(args) > 0:
                    func_args.insert(idx, args.pop(0))
                elif p_.default is inspect.Parameter.empty:
                    func_args.insert(idx, None)
                else:
                    func_args.insert(idx, p_.default)

        elif p_.kind is inspect.Parameter.KEYWORD_ONLY:
            if p in kwargs:
                func_kwargs[p] = kwargs.pop(p)

        elif p_.kind is inspect.Parameter.VAR_POSITIONAL:
            for a in args:
                func_args.append(a)

        elif p_.kind is inspect.Parameter.VAR_KEYWORD:
            func_kwargs.update(kwargs)

    return func_args, func_kwargs


def indent_text(text: str, level: int = 0) -> str:
        """Indent each line of ``text`` by ``level`` spaces."""
        return "\n".join([" " * level + line for line in text.split('\n')])


def get_environment_packages():
    """
    Generates a dictionary with the names of the installed packages and their current version.
    It detects if a package was installed in development mode and places the current commit hash instead of the version.
    """
    packages = {}
    for dist in distributions():
        package_name = dist.metadata['Name']
        version = dist.version
        location = Path(dist.locate_file(''))

        try:
            repo = Repo(location, search_parent_directories=True)
            if repo.is_dirty():
                raise RuntimeError(f"There are uncommitted changes in tracked files in {location}.")
            commit = repo.head.commit.hexsha
            packages[package_name] = commit
        except (InvalidGitRepositoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError):
                logger.warning(f"The package {package_name} has uncommitted changes. Will not be tracked. Please fix")
                packages[package_name] = 'uncommitted-changes'
            elif isinstance(e, InvalidGitRepositoryError):
                # Editable packages might appear twice in the list of distributions. If the one pointing to the repo appears second, it will get overwritten.
                if package_name not in packages:
                    packages[package_name] = version

    return packages


def commit_changes_in_repo(current_dir: Path) -> Optional[str]:
    """
    Commits the changes in the repository at the given directory and returns the commit hash.
    If the directory is not a git repository, it returns None.
    """
    try:
        repo = Repo(current_dir, search_parent_directories=True)
        if repo.is_dirty(untracked_files=True):
            repo.git.add(A=True)
            repo.git.commit('-m', '[Auto-commit] Save changes before running measurement')
            commit_hash = repo.head.commit.hexsha
            return commit_hash
        commit_hash = repo.head.commit.hexsha
        return commit_hash
    except InvalidGitRepositoryError:
        return None


def add_end_number_to_repeated_file(path: Path) -> Path:
    """
    Checks if a file or directory with given path exists, if it does it add `(x)` where `x` is an increasing number,
    until a file or directory with that name does not exist.
    """
    i = 1
    new_path = path
    while new_path.exists():
        new_path = path.parent.joinpath(f"{path.stem}({i}){path.suffix}")
        i += 1
    return new_path
