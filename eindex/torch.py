from typing import Any, List, Optional, Protocol, TypeVar, Union

from . import _core
from ._core import Aggregation

import torch

__all__ = ["argmin", "argmax", "argsort", "gather"]


class ArrayProtocol(Protocol):
    @property
    def device(self) -> Any:
        pass

def np_take(input, indices, axis=None):
    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices, device=input.device)
    if axis is None:
        return input.flatten()[indices]
    if axis < 0:
        axis += input.ndim
    input = input.movedim(axis, 0)
    result = input[indices]
    if axis != 0:
        result = result.movedim(0, axis)
    return result

class _TorchXP(object):
    int64 = torch.int64
    float64 = torch.float64
    inf = torch.inf

    def reshape(self, x: torch.Tensor, shape: List[int]):
        return torch.reshape(x, shape)

    def full(self, shape: List[int], fill_value: float, dtype: torch.dtype = None):
        return torch.full(shape, fill_value, dtype=dtype)

    def permute(self, x: torch.Tensor, permutation: List[int]):
        return torch.permute(x, permutation)

    def arange(self, axis_len: int, dtype: torch.dtype = None, device: torch.device = None):
        return torch.arange(axis_len, dtype=dtype, device=device)

    def all(self, x: torch.Tensor):
        return torch.all(torch.as_tensor(x))
    
    def broadcast_to(self, x: torch.Tensor, shape: List[int]):
        return torch.broadcast_to(x, shape)

    def take(self, x: torch.Tensor, indices: torch.Tensor, axis: int):
        return np_take(x, indices, axis=axis)
    
    def stack(self, arrays: List[torch.Tensor], axis=0):
        return torch.stack(arrays, dim=axis)
    
    def argmax(self, array, axis=None):
        return torch.argmax(array, dim=axis)
    
    def argmin(self, array, axis=None):
        return torch.argmin(array, dim=axis)

    def argsort(self, array, axis=None):
        if axis is None:
            return torch.argsort(array.flatten())
        else:
            return torch.argsort(array, dim=axis)

    def permute_dims(self, array: torch.Tensor, dims: List[int]):
        return array.permute(dims)

    def sort(self, array: torch.Tensor, axis: int = -1):
        return torch.sort(array, dim=axis).values
    
    def sum(self, array: torch.Tensor, axis=None):
        if axis is None:
            return array.sum()
        else:
            return array.sum(dim=axis)
    
    def mean(self, array: torch.Tensor, axis=None):
        if axis is None:
            return array.mean()
        else:
            return array.mean(dim=axis)
            
    def astype(self, array: torch.Tensor, dtype: torch.dtype):
        return array.to(dtype)
    
    def max(self, array: torch.Tensor, axis=None):
        if axis is None:
            # Wtf? torch.max has a cryptic error about named dims if you pass None
            return array.max()
        elif isinstance(axis, int):
            return array.max(dim=axis).values
        elif isinstance(axis, (tuple, list)):
            for i in sorted(axis, reverse=True):
                array = array.max(dim=i).values
            return array

    def min(self, array: torch.Tensor, axis=None):
        if axis is None:
            return array.min()
        elif isinstance(axis, int):
            return array.min(dim=axis).values
        elif isinstance(axis, (tuple, list)):
            for i in sorted(axis, reverse=True):
                array = array.min(dim=i).values
            return array
            


class _TorchIXP(_core.IXP):
    def __init__(self) -> None:
        self.xp = _TorchXP()

    def permute_dims(self, arr, permutation):
        return self.xp.permute(arr, permutation)

    def arange_at_position(self, n_axes, axis, axis_len, array_to_copy_device_from: ArrayProtocol):
        xp = self.xp
        x = xp.arange(axis_len, dtype=xp.int64, device=array_to_copy_device_from.device)
        shape = [1] * n_axes
        shape[axis] = axis_len
        x = xp.reshape(x, shape)
        return x


Array = TypeVar("Array", bound=ArrayProtocol)


def argmax(tensor: Array, pattern: str, /) -> Array:
    formula = _core.ArgmaxFormula(pattern)
    ixp = _TorchIXP()
    return formula.apply_to_ixp(ixp, tensor)


def argmin(tensor: Array, pattern: str, /) -> Array:
    formula = _core.ArgminFormula(pattern)
    ixp = _TorchIXP()
    return formula.apply_to_ixp(ixp, tensor)


def argsort(tensor: Array, pattern: str, /, *, order_axis="order") -> Array:
    formula = _core.ArgsortFormula(pattern, order_axis=order_axis)
    ixp = _TorchIXP()
    return formula.apply_to_ixp(ixp, tensor)


def _einindex(arr: Array, ind: Union[Array, List[Array]], pattern: str, /):
    formula = _core.IndexFormula(pattern)
    ixp = _TorchIXP()
    return formula.apply_to_array_api(ixp, arr, ind)


def gather(arr: Array, ind: Union[Array, List[Array]], pattern: str, /, agg: Optional[Aggregation] = None):
    formula = _core.GatherFormula(pattern=pattern, agg=agg)
    ixp = _TorchIXP()
    return formula.apply_to_array_api(ixp, arr, ind)


# scatter and gather_scatter is not implemented - maybe do them later
