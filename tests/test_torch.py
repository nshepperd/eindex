from typing import TypeVar
import numpy as np
import torch
import unittest

from eindex._core import CompositionDecomposition, zip2
from eindex.torch import _TorchIXP, _einindex, argmax, argmin, argsort, gather

from .utils import (
    _enum_1d,
    compose_index,
    enumerate_indexer,
    flatten,
    generate_array,
    generate_indexer,
    pseudo_random_tensor,
    range_of_shape,
    to_flat_index,
)

T = TypeVar("T")


def test_composition_and_decomposition():
    ixp = _TorchIXP()
    xp = ixp.xp

    x = range_of_shape(2, 3, 5, 7, xp=xp)
    comp = CompositionDecomposition(
        decomposed_shape=["a", "b", "c", "d"],
        composed_shape=[["a", "b"], ["c", "d"]],
    )
    x_composed = comp.compose_ixp(ixp, x, known_axes_lengths={})
    assert x_composed.shape == (2 * 3, 5 * 7)
    assert xp.all(x_composed == xp.reshape(x, (2 * 3, 5 * 7)))

    y = CompositionDecomposition(
        decomposed_shape=["a", "b", "c", "d"],
        composed_shape=[["a", "b"], [], ["c", "d"], []],
    ).compose_ixp(ixp, x, {})
    assert y.shape == (2 * 3, 1, 5 * 7, 1)
    assert xp.all(xp.reshape(x, (-1,)) == xp.reshape(y, (-1,)))

    comp = CompositionDecomposition(
        decomposed_shape=["a", "b", "e", "c", "d"],
        composed_shape=[["e", "c"], ["b"], ["a", "d"]],
    )
    x = range_of_shape(2, 3, 5, 7, 3, xp=xp)

    axes = {}
    y = comp.compose_ixp(ixp, x, axes)
    assert y.shape == (5 * 7, 3, 2 * 3)
    y_manual = xp.reshape(xp.permute(x, (2, 3, 1, 0, 4)), y.shape)

    assert xp.all(y == y_manual)
    x2 = comp.decompose_ixp(ixp, y, axes)
    assert xp.all(x == x2)


def test_simple_indexing():
    ixp = _TorchIXP()
    xp = ixp.xp

    # simple 2d test
    arr = pseudo_random_tensor(xp, [5, 7])
    ind = xp.arange(7) % 5
    x = _einindex(arr, [ind], "i j, [i] j -> j")
    for j, i in _enum_1d(ind):
        assert arr[i, j] == x[j]

    y = _einindex(xp.permute(arr, (1, 0)), [ind], "j i, [i] j -> j")
    for j, i in _enum_1d(ind):
        assert arr[i, j] == y[j]


def test_multidimensional_indexing():
    ixp = _TorchIXP()
    xp = ixp.xp

    B, H, W, C, T = 2, 3, 5, 7, 11
    hindices_bt = pseudo_random_tensor(xp, [B, T]) % H
    windices_bt = pseudo_random_tensor(xp, [B, T]) % W
    _t = hindices_bt

    embedding_bhwc = (
        0
        + ixp.arange_at_position(4, 0, B, _t) * 1000
        + ixp.arange_at_position(4, 1, H, _t) * 100
        + ixp.arange_at_position(4, 2, W, _t) * 10
        + ixp.arange_at_position(4, 3, C, _t) * 1
    )

    # imagine that you have pairs of image <> sentence.
    # goal is to get most suitable token from image for every token in sentence
    # thus for every token in sentence you compute best H and vW

    result = _einindex(embedding_bhwc, [hindices_bt, windices_bt], "b H W c, [H, W] b t -> c t b")
    # example of using a single array for indexing multiple axes
    hw_indices_bt = xp.stack([hindices_bt, windices_bt])
    result2 = _einindex(embedding_bhwc, hw_indices_bt, "b H W c, [H, W] b t -> c t b")
    assert xp.all(result == result2)

    # check vs manual element computation
    result_manual = result * 0
    for b in range(B):
        for t in range(T):
            for c in range(C):
                h = int(hindices_bt[b, t])
                w = int(windices_bt[b, t])
                result_manual[c, t, b] = embedding_bhwc[b, h, w, c]

    assert xp.all(result == result_manual)


def test_reverse_indexing():
    ixp = _TorchIXP()
    xp = ixp.xp

    C, T, B = 2, 3, 5
    # G = GPU, batch-like varaible
    G = 4
    H = 7
    W = 9

    t_indices_gbhw = xp.reshape(xp.arange(G * B * H * W), (G, B, H, W)) % T
    _t = t_indices_gbhw

    arr_gtbc = (
        0
        + ixp.arange_at_position(4, 0, G, _t) * 1000
        + ixp.arange_at_position(4, 1, T, _t) * 100
        + ixp.arange_at_position(4, 2, B, _t) * 10
        + ixp.arange_at_position(4, 3, C, _t) * 1
    )

    result = _einindex(arr_gtbc, [t_indices_gbhw], "g t b c, [t] g b h w -> g b c h w")

    result_manual = result * 0
    for g in range(G):
        for b in range(B):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        t = int(t_indices_gbhw[g, b, h, w])
                        result_manual[g, b, c, h, w] = arr_gtbc[g, t, b, c]

    assert xp.all(result == result_manual)


def check_max_min(x, pattern: str):
    xp = _TorchIXP().xp
    assert xp.all(argmax(x, pattern) == argmin(-x, pattern))


def test_argmax_straight():
    ixp = _TorchIXP()
    xp = _TorchIXP().xp

    A, B, C, D = 2, 3, 5, 7
    x = pseudo_random_tensor(xp, [A, B, C, D])
    # set one maximum for every B, so argmax is unambiguous
    for b in range(B):
        x[1, b, b + 1, b + 2] = 2000 + b
    [a, b, c, d] = argmax(x, "a b c d -> [a, b, c, d]")
    assert x[a, b, c, d] == xp.max(x)
    cad = argmax(x, "a b c d -> [c, a, d] b")
    comp = CompositionDecomposition(composed_shape=[["c", "a", "d"], ["b"]], decomposed_shape=["a", "b", "c", "d"])
    reference = xp.argmax(comp.compose_ixp(ixp, x, {}), axis=0)
    assert xp.all(reference == compose_index(cad, [C, A, D]))


def test_argmax_by_indexing():
    ixp = _TorchIXP()
    xp = _TorchIXP().xp

    x = xp.reshape(xp.arange(3 * 4 * 5), (3, 4, 5))
    x[1, 2, 3] = 10000
    reference = xp.argmax(x, axis=0)

    assert xp.all(argmax(x, "i j k -> [i] j k")[0, ...] == reference)
    assert xp.all(argmax(x, "i j k -> [i] k j")[0, ...] == reference.T)

    ind = argmax(x, "i j k -> [i] j k")
    assert xp.all(_einindex(x, ind, "i j k, [i] j k -> j k") == xp.max(x, axis=0))

    ind = argmax(x, "i j k -> [i, j] k")
    assert xp.all(_einindex(x, ind, "i j k, [i, j] k -> k") == xp.max(x, axis=(0, 1)))

    ind = argmax(x, "i j k -> [j, i] k")
    assert xp.all(_einindex(x, ind, "i j k, [j, i] k -> k") == xp.max(x, axis=(0, 1)))

    ind = argmax(x, "i j k -> [i, k] j")
    assert xp.all(_einindex(x, ind, "i j k, [i, k] j -> j") == xp.max(x, axis=(0, 2)))

    ind = argmax(x, "i j k -> [k, i, j]")
    assert xp.all(_einindex(x, ind, "i j k, [k, i, j] -> ") == xp.max(x))

    check_max_min(x, "i j k -> [k, i, j]")
    check_max_min(x, "i j k -> [i, j] k")
    check_max_min(x, "i j k -> [j, i] k")
    check_max_min(x, "i j k -> [j] k i")


def test_argsort_against_numpy():
    ixp = _TorchIXP()
    xp = _TorchIXP().xp

    x = xp.reshape(xp.arange(3 * 4 * 5), (3, 4, 5))
    x[1, 2, 3] = 1000

    assert xp.all(argsort(x, "i j k -> [i] order j k")[0, ...] == xp.argsort(x, axis=0))
    right = xp.permute_dims(xp.argsort(x, axis=0), (2, 1, 0))
    assert xp.all(argsort(x, "i j k -> [i] k j order")[0, ...] == right)

    ind = argsort(x, "i j k -> [k, i, j] order")
    assert xp.all(_einindex(x, ind, "i j k, [k, i, j] order -> order") == xp.sort(xp.reshape(x, (-1,))))

    ind = argsort(x, "i j k -> [k, i] order j")
    reference = xp.permute_dims(x, (0, 2, 1))
    reference = xp.reshape(reference, (-1, reference.shape[-1]))
    assert xp.all(_einindex(x, ind, "i j k, [k, i] order j -> order j") == xp.sort(reference, axis=0))


def test_index():
    ixp = _TorchIXP()
    xp = _TorchIXP().xp

    sizes = dict(
        a=2,
        b=3,
        c=5,
        d=7,
        e=2,
        f=3,
        g=4,
        h=5,
    )

    array = generate_array(xp, "a b c d", sizes=sizes)
    indexer = generate_indexer(xp, "[a, c] d f g", sizes=sizes)
    result_einindex = _einindex(array, indexer, "a b c d, [a, c] d f g -> g f d b")
    result = gather(array, indexer, "a b c d, [a, c] d f g -> g f d b")
    print(result.shape, flatten(xp, result)[0])
    print(result_einindex.shape, flatten(xp, result_einindex)[0])
    indexer_as_dict = enumerate_indexer(ixp, "[a, c] d f g", indexer=indexer, sizes=sizes)

    for b in range(sizes["b"]):
        flat_index_arr = to_flat_index("a b c d", {**indexer_as_dict, "b": b}, sizes=sizes)
        flat_index_result = to_flat_index("g f d b", {**indexer_as_dict, "b": b}, sizes=sizes)

        array_flat = flatten(xp, array)
        result_flat = flatten(xp, result)

        for i, j in zip2(flat_index_arr, flat_index_result):
            assert array_flat[i] == result_flat[j], ("failed", i, j)

    assert xp.all(result_einindex == result), (result_einindex, result)


def test_gather():
    ixp = _TorchIXP()
    xp = _TorchIXP().xp

    sizes = dict(
        a=2,
        b=3,
        c=5,
        d=7,
        i1=3,
        i2=5,
        r=3,
    )

    final_pattern = "b c d"
    array_pattern = "b i1 i2 d r"
    index_pattern = "[i1, i2] c b a r"
    full_pattern = f"{array_pattern}, {index_pattern} -> {final_pattern}"
    array = generate_array(xp, array_pattern=array_pattern, sizes=sizes)
    indexer = generate_indexer(xp, index_pattern, sizes=sizes)
    result_gather = gather(array, indexer, full_pattern, agg="sum")

    indexer_as_dict = enumerate_indexer(ixp, index_pattern, indexer=indexer, sizes=sizes)

    array_flat = flatten(xp, array)
    result_flat = flatten(xp, result_gather)

    for d in range(sizes["d"]):
        flat_index_array = to_flat_index(array_pattern, {**indexer_as_dict, "d": d}, sizes=sizes)
        flat_index_final = to_flat_index(final_pattern, {**indexer_as_dict, "d": d}, sizes=sizes)

        for ia, ir in zip2(flat_index_array, flat_index_final):
            result_flat[ir] -= array_flat[ia]

    assert xp.max(abs(result_flat)) == 0

    # checking different aggregations
    array_flat_float = xp.astype(array_flat, xp.float64)

    for agg_name, agg_func, default_value in [
        ("sum", lambda a, b: a + b, 0.0),
        ("min", min, xp.inf),
        ("max", max, -xp.inf),
    ]:
        result_gather = gather(array, indexer, full_pattern, agg=agg_name)
        result_gather = xp.reshape(result_gather, (-1,))
        result_ref = xp.full(shape=tuple(sizes[d] for d in final_pattern.split()), fill_value=default_value)
        result_ref = xp.reshape(result_ref, (-1,))
        for d in range(sizes["d"]):
            flat_index_array = to_flat_index(array_pattern, {**indexer_as_dict, "d": d}, sizes=sizes)
            flat_index_final = to_flat_index(final_pattern, {**indexer_as_dict, "d": d}, sizes=sizes)

            for ia, ir in zip2(flat_index_array, flat_index_final):
                result_ref[ir] = agg_func(array_flat_float[ia], result_ref[ir])
        assert xp.all(xp.astype(result_ref, xp.int64) == result_gather)

    # checking mean aggregation on constant tensor
    result_mean_const = gather(xp.full(array.shape, fill_value=3.0), indexer, full_pattern, agg="mean")
    assert xp.all(result_mean_const == 3.0)

    # testing that ratio is constant, as number of elements averaged is the same for every result entry
    values = xp.astype(array**2, xp.float64) + 1.0
    result_mean = gather(values, indexer, full_pattern, agg="mean")
    result__sum = gather(values, indexer, full_pattern, agg="sum")
    ratio = result_mean / result__sum
    assert xp.min(ratio) * 0.99 < xp.max(ratio) < xp.min(ratio) * 1.01

class TestTorchXP(unittest.TestCase):
    """Basic tests for the TorchIXP implementation since we're using it in the above test cases."""

    def setUp(self):
        self.xp = _TorchIXP().xp
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
    def _to_numpy(self, tensor):
        return tensor.cpu().numpy()
    
    def assertArrayEqual(self, torch_result, numpy_result):
        if isinstance(torch_result, torch.Tensor):
            torch_result = self._to_numpy(torch_result)
        np.testing.assert_array_almost_equal(torch_result, numpy_result)

    def test_reshape(self):
        x = torch.randn(2, 3, 4)
        x_np = self._to_numpy(x)
        
        shapes = [(24,), (6, 4), (2, 3, 4), (2, -1)]
        for shape in shapes:
            torch_result = self.xp.reshape(x, shape)
            numpy_result = np.reshape(x_np, shape)
            self.assertArrayEqual(torch_result, numpy_result)

    def test_full(self):
        shapes = [(2, 3), (4,), (2, 3, 4)]
        values = [0.0, 1.0, -1.0, np.inf]
        dtypes = [torch.float32, torch.int64]
        
        for shape in shapes:
            for value in values:
                for dtype in dtypes:
                    if value is np.inf and dtype is torch.int64:
                        continue
                    torch_result = self.xp.full(shape, value, dtype)
                    numpy_result = np.full(shape, value, dtype=torch_to_numpy_dtype(dtype))
                    self.assertArrayEqual(torch_result, numpy_result)

    def test_permute(self):
        x = torch.randn(2, 3, 4)
        x_np = self._to_numpy(x)
        
        permutations = [(0, 2, 1), (2, 1, 0), (1, 0, 2)]
        for perm in permutations:
            torch_result = self.xp.permute(x, perm)
            numpy_result = np.transpose(x_np, perm)
            self.assertArrayEqual(torch_result, numpy_result)

    def test_arange(self):
        for length in [0, 1, 10, 100]:
            torch_result = self.xp.arange(length)
            numpy_result = np.arange(length)
            self.assertArrayEqual(torch_result, numpy_result)

    def test_all(self):
        test_cases = [
            torch.tensor([True, True, True]),
            torch.tensor([True, False, True]),
            torch.tensor([[True, True], [True, True]]),
            torch.tensor([[True, False], [True, True]]),
        ]
        
        for x in test_cases:
            torch_result = self.xp.all(x)
            numpy_result = np.all(self._to_numpy(x))
            self.assertEqual(bool(torch_result), numpy_result)

    def test_broadcast_to(self):
        x = torch.randn(3, 1)
        x_np = self._to_numpy(x)
        
        shapes = [(3, 4), (3, 5), (2, 3, 4)]
        for shape in shapes:
            try:
                torch_result = self.xp.broadcast_to(x, shape)
                numpy_result = np.broadcast_to(x_np, shape)
                self.assertArrayEqual(torch_result, numpy_result)
            except (RuntimeError, ValueError):
                # Ensure both raise errors for invalid broadcasts
                with self.assertRaises(Exception):
                    np.broadcast_to(x_np, shape)

    def test_stack(self):
        arrays = [torch.randn(2, 3) for _ in range(4)]
        arrays_np = [self._to_numpy(arr) for arr in arrays]
        
        for axis in range(3):
            torch_result = self.xp.stack(arrays, axis=axis)
            numpy_result = np.stack(arrays_np, axis=axis)
            self.assertArrayEqual(torch_result, numpy_result)

    def test_argmax_argmin(self):
        x = torch.randn(3, 4, 5)
        x_np = self._to_numpy(x)
        
        # Test with different axis values
        for axis in [None, 0, 1, 2]:
            torch_max = self.xp.argmax(x, axis=axis)
            numpy_max = np.argmax(x_np, axis=axis)
            self.assertArrayEqual(torch_max, numpy_max)
            
            torch_min = self.xp.argmin(x, axis=axis)
            numpy_min = np.argmin(x_np, axis=axis)
            self.assertArrayEqual(torch_min, numpy_min)

    def test_argsort(self):
        x = torch.randn(3, 4, 5)
        x_np = self._to_numpy(x)
        
        for axis in [None, 0, 1, 2]:
            torch_result = self.xp.argsort(x, axis=axis)
            numpy_result = np.argsort(x_np, axis=axis)
            print(axis)
            self.assertArrayEqual(torch_result, numpy_result)

    def test_sort(self):
        x = torch.randn(3, 4, 5)
        x_np = self._to_numpy(x)
        
        for axis in [-1, 0, 1, 2]:
            torch_result = self.xp.sort(x, axis=axis)
            numpy_result = np.sort(x_np, axis=axis)
            self.assertArrayEqual(torch_result, numpy_result)

    def test_sum_mean(self):
        x = torch.randn(3, 4, 5)
        x_np = self._to_numpy(x)
        
        # Test with different axis values and None
        for axis in [None, 0, 1, 2]:
            torch_sum = self.xp.sum(x, axis=axis)
            numpy_sum = np.sum(x_np, axis=axis)
            self.assertArrayEqual(torch_sum, numpy_sum)
            
            torch_mean = self.xp.mean(x, axis=axis)
            numpy_mean = np.mean(x_np, axis=axis)
            self.assertArrayEqual(torch_mean, numpy_mean)

    def test_max_min(self):
        x = torch.randn(3, 4, 5)
        x_np = self._to_numpy(x)
        
        # Test single axis
        for axis in [None, 0, 1, 2]:
            torch_max = self.xp.max(x, axis=axis)
            numpy_max = np.max(x_np, axis=axis)
            self.assertArrayEqual(torch_max, numpy_max)
            
            torch_min = self.xp.min(x, axis=axis)
            numpy_min = np.min(x_np, axis=axis)
            self.assertArrayEqual(torch_min, numpy_min)
        
        # Test multiple axes
        for axes in [(0, 1), (1, 2), (0, 2)]:
            torch_max = self.xp.max(x, axis=axes)
            numpy_max = np.max(x_np, axis=axes)
            self.assertArrayEqual(torch_max, numpy_max)
            
            torch_min = self.xp.min(x, axis=axes)
            numpy_min = np.min(x_np, axis=axes)
            self.assertArrayEqual(torch_min, numpy_min)

def torch_to_numpy_dtype(torch_dtype):
    """Helper function to convert torch dtype to numpy dtype"""
    if torch_dtype == torch.float32:
        return np.float32
    elif torch_dtype == torch.float64:
        return np.float64
    elif torch_dtype == torch.int64:
        return np.int64
    # Add more dtype conversions as needed
    raise ValueError(f"Unsupported dtype: {torch_dtype}")
