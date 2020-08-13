import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import array_shapes, arrays, floating_dtypes
from hypothesis.strategies import floats, sampled_from

from dict_minimize.core._scipy import SCIPY_DTYPE, _default_to_np
from dict_minimize.jax_api import from_np

to_np = _default_to_np

np_float_arrays = arrays(
    dtype=floating_dtypes(),
    shape=array_shapes(min_dims=0, max_dims=5, min_side=0, max_side=5),
    elements=floats(allow_nan=False, width=16),
)
jax_float_dtypes = sampled_from(["float16", "float32", "float64"])


@given(np_float_arrays, jax_float_dtypes)
def test_from_np(x_np, dtype_str):
    x_jax = from_np(x_np, dtype_str)
    x_np2 = to_np(x_jax)

    assert isinstance(x_np2, np.ndarray)
    x_np2.dtype.kind == "f"

    assert np.allclose(x_np, x_np2)

    x_np2 = x_np2.astype(SCIPY_DTYPE, copy=False)
    assert x_np2.dtype == SCIPY_DTYPE

    assert np.allclose(x_np, x_np2)
