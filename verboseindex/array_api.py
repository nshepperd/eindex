from typing import Any, List, Protocol, TypeVar, Union

from ._core import IXP, ArgmaxFormula, ArgminFormula, ArgsortFormula, IndexFormula


class ArrayProtocol(Protocol):
    def __array_namespace__(self) -> Any:
        pass

    @property
    def device(self) -> Any:
        pass


class ArrayApiIXP(IXP):
    def __init__(self, xp) -> None:
        self.xp = xp

    def permute_dims(self, arr, permutation):
        return self.xp.permute_dims(arr, permutation)

    def arange_at_position(self, n_axes, axis, axis_len, array_to_copy_device_from: ArrayProtocol):
        xp = self.xp
        x = xp.arange(axis_len, dtype=xp.int64, device=array_to_copy_device_from.device)
        shape = [1] * n_axes
        shape[axis] = axis_len
        x = xp.reshape(x, shape)
        return x


Array = TypeVar("Array", bound=ArrayProtocol)


def argmax(tensor: Array, pattern: str, /) -> Array:
    formula = ArgmaxFormula(pattern)
    ixp = ArrayApiIXP(tensor.__array_namespace__())
    return formula.apply_to_ixp(ixp, tensor)


def argmin(tensor: Array, pattern: str, /) -> Array:
    formula = ArgminFormula(pattern)
    ixp = ArrayApiIXP(tensor.__array_namespace__())
    return formula.apply_to_ixp(ixp, tensor)


def argsort(tensor: Array, pattern: str, /) -> Array:
    formula = ArgsortFormula(pattern)
    ixp = ArrayApiIXP(tensor.__array_namespace__())
    return formula.apply_to_ixp(ixp, tensor)


def einindex(pattern: str, arr: Array, ind: Union[Array, List[Array]], /):
    formula = IndexFormula(pattern)
    ixp = ArrayApiIXP(arr.__array_namespace__())
    return formula.apply_to_array_api(ixp, arr, ind)


# current standard misses API for indexing
