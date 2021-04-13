from collections import OrderedDict

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis.extra.numpy import array_shapes, arrays, floating_dtypes
from hypothesis.strategies import dictionaries, floats, integers, lists, sampled_from, text, tuples
from jax.config import config
from jax.dtypes import _jax_types

from dict_minimize.core._scipy import SCIPY_DTYPE, _default_to_np, _pack, _unpack
from dict_minimize.jax_api import _from_np, _get_dtype, minimize

# The float64 tests will not work without this:
config.update("jax_enable_x64", True)

get_dtype = _get_dtype
from_np = _from_np
to_np = _default_to_np

np_float_arrays = arrays(
    dtype=floating_dtypes(),
    shape=array_shapes(min_dims=0, max_dims=5, min_side=0, max_side=5),
    elements=floats(allow_nan=False, width=16),
)
# See: https://jax.readthedocs.io/en/latest/_modules/jax/dtypes.html
jax_float_dtypes = sampled_from(["float16", "float32", "float64"])
grad_methods = sampled_from(["CG", "BFGS", "L-BFGS-B", "TNC", "SLSQP", "trust-constr"])
# These are the only optimizers that seem to always respect the bounds arguments, no matter what
always_respects_bounds = ("L-BFGS-B", "TNC", "SLSQP")


def prep3(v):
    return (3,) + tuple(v)


np_float_arrays3 = arrays(
    dtype=floating_dtypes(),
    shape=array_shapes(min_dims=0, max_dims=5, min_side=0, max_side=5).map(prep3),
    elements=floats(allow_nan=False, width=16),
    unique=True,
)


def to_np_validated(x):
    x_np = to_np(x)
    # These asserts could be made exceptions for user's sake
    assert isinstance(x_np, np.ndarray), "Numpy conversion must return ndarray."
    assert x_np.dtype.kind == "f", "Numpy conversion must return float ndarray."
    x_np = x_np.astype(SCIPY_DTYPE, copy=False)
    return x_np


@given(dictionaries(text(), tuples(np_float_arrays, jax_float_dtypes), min_size=1))
def test_pack_unpack(x_dict):
    x_dict = OrderedDict([(kk, from_np(vv, dd)) for kk, (vv, dd) in x_dict.items()])

    # Get dtypes as well
    x_dtypes = OrderedDict([(kk, get_dtype(vv)) for kk, vv in x_dict.items()])

    x_vec, shapes = _pack(x_dict, to_np_validated)

    # Check we get same thing when we pass in shapes
    x_vec_, shapes_ = _pack(x_dict, to_np_validated, shapes)
    assert np.all(x_vec == x_vec_)
    assert shapes is shapes_

    # Now test we get same thing back on round-trip to x_dict
    x_dict_ = _unpack(x_vec, from_np, shapes, x_dtypes)
    assert isinstance(x_dict_, OrderedDict)
    assert list(x_dict_.keys()) == list(x_dict.keys())
    for kk in x_dict.keys():
        assert type(x_dict[kk]) == type(x_dict_[kk])  # noqa: E721
        assert str(x_dict[kk].dtype) == str(x_dict_[kk].dtype)
        assert x_dict[kk].dtype == x_dict_[kk].dtype
        assert x_dict[kk].shape == x_dict_[kk].shape
        assert np.allclose(to_np(x_dict[kk]), to_np(x_dict_[kk]))

    # Now Let's try to round trip the other way
    x_vec_, shapes_ = _pack(x_dict_, to_np_validated, shapes)
    assert np.allclose(x_vec, x_vec_)
    assert shapes is shapes_


@given(np_float_arrays, sampled_from(_jax_types))
def test_get_dtype(x_np, dtype):
    x_jax = from_np(x_np, dtype)

    dtype2 = get_dtype(x_jax)

    assert dtype == dtype2
    # We only expect the str version to match if it is a np dtype
    assert str(np.dtype(dtype)) == str(dtype2)


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


def validate_solution(x0_dict, x_sol, lb=None, ub=None):
    assert isinstance(x_sol, OrderedDict)
    assert list(x_sol.keys()) == list(x0_dict.keys())
    for kk in x0_dict.keys():
        assert type(x0_dict[kk]) == type(x_sol[kk])  # noqa: E721
        assert str(x0_dict[kk].dtype) == str(x_sol[kk].dtype)
        assert x0_dict[kk].dtype == x_sol[kk].dtype
        assert x0_dict[kk].shape == x_sol[kk].shape
        assert (lb is None) or np.all(lb[kk] <= x_sol[kk])
        assert (ub is None) or np.all(x_sol[kk] <= ub[kk])


@settings(deadline=None)
@given(
    dictionaries(text(), tuples(np_float_arrays, jax_float_dtypes), min_size=1),
    lists(integers()),
    grad_methods,
    floats(0, 1),
)
def test_minimize(x0_dict, args, method, tol):
    total_el = sum(vv.size for vv, _ in x0_dict.values())
    assume(total_el > 0)

    x0_dict = OrderedDict([(kk, from_np(vv, dd)) for kk, (vv, dd) in x0_dict.items()])
    args = tuple(args)

    def dummy_f(xk, *args_):
        assert args == args_
        validate_solution(x0_dict, xk)
        # Pass back some arbitrary stuff
        v = sum(vv.sum() for vv in xk.values())
        dv = OrderedDict([kk, vv + 1] for kk, vv in xk.items())
        return v, dv

    def callback(xk):
        validate_solution(x0_dict, xk)

    x_sol = minimize(dummy_f, x0_dict, args=args, method=method, tol=tol, callback=callback, options={"maxiter": 10})
    validate_solution(x0_dict, x_sol)


@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow], deadline=None)
@given(
    dictionaries(text(), tuples(np_float_arrays3, jax_float_dtypes), min_size=1),
    lists(integers()),
    grad_methods,
    floats(0, 1),
)
def test_minimize_bounded(x0_dict_, args, method, tol):
    total_el = sum(vv.size for vv, _ in x0_dict_.values())
    assume(total_el > 0)

    args = tuple(args)

    check_bounds = method in always_respects_bounds

    x0_dict = OrderedDict()
    lb_dict = OrderedDict()
    ub_dict = OrderedDict()
    for kk, (vv, dd) in x0_dict_.items():
        vv = np.sort(vv, axis=0)
        lb, vv, ub = vv
        x0_dict[kk] = from_np(vv, dd)
        lb_dict[kk] = from_np(lb, dd)
        ub_dict[kk] = from_np(ub, dd)

    validate_solution(x0_dict, x0_dict, lb_dict, ub_dict)

    def dummy_f(xk, *args_):
        assert args == args_

        if check_bounds:
            validate_solution(x0_dict, xk, lb_dict, ub_dict)
        else:
            validate_solution(x0_dict, xk)

        # Pass back some arbitrary stuff
        v = sum(vv.sum() for vv in xk.values())
        dv = OrderedDict([kk, vv + 1] for kk, vv in xk.items())
        return v, dv

    def callback(xk):
        if check_bounds:
            validate_solution(x0_dict, xk, lb_dict, ub_dict)
        else:
            validate_solution(x0_dict, xk)

    x_sol = minimize(
        dummy_f,
        x0_dict,
        args=args,
        lb_dict=lb_dict,
        ub_dict=ub_dict,
        method=method,
        tol=tol,
        callback=callback,
        options={"maxiter": 10},
    )

    if check_bounds:
        validate_solution(x0_dict, x_sol, lb_dict, ub_dict)
    else:
        validate_solution(x0_dict, x_sol)
