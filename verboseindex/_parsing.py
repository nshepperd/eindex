
from collections import Counter
from typing import Iterable, List, Literal, Tuple

from . import VerboseIndexError

Aggregation = Literal["set", "min", "max", "sum", "mean", "std", "logsumexp"]


def _verify_axis_name(name: str) -> Tuple[bool, str]:
    if not str.isidentifier(name):
        return False, f"Axis name {name} is not a valid python identifier"
    elif name[0] == "_" or name[-1] == "_":
        return False, f"Axis name {name} should should not start or end with underscore"
    else:
        return True, ""


def _parse_space_separated_dimensions(dims: str) -> List[str]:
    if "[" in dims:
        raise VerboseIndexError(
            f"Symbol [ was used in a part '{dims}', which does not contain indexers, only space-separated axes"
        )
    if "]" in dims:
        raise VerboseIndexError(
            f"Symbol ] was used in a part '{dims}', which does not contain indexers, only space-separated axes"
        )
    axes_names = dims.split()

    for axis in axes_names:
        is_valid, reason = _verify_axis_name(axis)
        if not is_valid:
            raise VerboseIndexError(reason)
    if len(set(axes_names)) != len(axes_names):
        repeated_axes = [ax for ax, i in Counter(axes_names).items()]
        raise VerboseIndexError(f"Some axes were repeated: {repeated_axes}")

    return axes_names


def _parse_comma_separated_dimensions(dims: str) -> List[str]:
    dims = dims.strip()
    if len(dims) == 0:
        return []
    axes_names = [name.strip() for name in dims.split(",")]
    for axis in axes_names:
        if " " in axis:
            raise VerboseIndexError(f"Seems you forgot comma in indexing: '{axis}'")
        is_valid, reason = _verify_axis_name(axis)
        if not is_valid:
            raise VerboseIndexError(reason)
    return axes_names


def detect_duplicates(x: list[str]) -> list[str]:
    return [name for name, count in Counter(x).items() if count > 1]


def _parse_indexing_part(x: str, *, allow_duplicate_indexers: bool = False) -> Tuple[List[str], List[str]]:
    """
    Parses indexing part, e.g. '[ind_axis_1, ind_axis2] ind_other_axis1 ind_other_axis2'
    """
    x = x.strip()
    if not x.startswith("["):
        raise VerboseIndexError(f"Composition axis should go first in indexer, like '[h, w] i j k', not '{x}'")
    composition_start = 0  # x.index("[")
    composition_end = x.index("]")
    indexing_axes_names = _parse_comma_separated_dimensions(x[composition_start + 1 : composition_end])
    if not allow_duplicate_indexers:
        if duplicates := detect_duplicates(indexing_axes_names):
            raise VerboseIndexError(f"Axes {duplicates} present more than once in '{x}' ")
    indexer_other_axes_names = _parse_space_separated_dimensions(x[composition_end + 1 :])
    # did not check if there is an overlap between main and other axes
    return indexing_axes_names, indexer_other_axes_names


# presence reflects which parts of expression a particular axis participates in
# order is (in result, in array, in main indexing axes, in other indexing axes)
Presence = Tuple[bool, bool, bool, bool] 

class ParsedPattern:
    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        left, right = pattern.split("<-")
        arr_pattern, _, ind_pattern = right.partition(",")

        self.res_axes_names = _parse_space_separated_dimensions(left)
        self.arr_axes_names = _parse_space_separated_dimensions(arr_pattern)
        self.ind_axes_names, self.ind_other_axes_names = _parse_indexing_part(ind_pattern)


        for group_name, group, subpattern in [
            ("Result", self.res_axes_names, left),
            ("Array", self.arr_axes_names, arr_pattern),
            ("Indexer", self.ind_axes_names + self.ind_other_axes_names, ind_pattern),
        ]:
            if len(set(group)) != len(group):
                raise VerboseIndexError(f"{group_name} pattern ({subpattern}) contains a duplicated axis in {pattern}")


    def axis2presence(self) -> Iterable[Tuple[str, Presence]]:
        all_axes = {
            *self.res_axes_names,
            *self.arr_axes_names,
            *self.ind_axes_names,
            *self.ind_other_axes_names,
        }
        for axis in list(all_axes):
            yield axis, (
                axis in self.res_axes_names,
                axis in self.arr_axes_names,
                axis in self.ind_axes_names,
                axis in self.ind_other_axes_names,
            )