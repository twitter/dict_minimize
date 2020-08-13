from collections import OrderedDict

import numpy as np
from hypothesis import assume, given, settings
from hypothesis.extra.numpy import array_shapes, arrays, floating_dtypes, scalar_dtypes
from hypothesis.strategies import booleans, dictionaries, floats, integers, lists, sampled_from, text, tuples

from dict_minimize.core._scipy import SCIPY_DTYPE, _default_to_np
from dict_minimize.numpy_api import from_np, get_dtype, minimize

to_np = _default_to_np

np_float_arrays = arrays(
    dtype=floating_dtypes(),
    shape=array_shapes(min_dims=0, max_dims=5, min_side=0, max_side=5),
    elements=floats(allow_nan=False, width=16),
)
grad_methods = sampled_from(["CG", "BFGS", "L-BFGS-B", "TNC", "SLSQP", "trust-constr"])


def prep3(v):
    return (3,) + tuple(v)


np_float_arrays3 = arrays(
    dtype=floating_dtypes(),
    shape=array_shapes(min_dims=0, max_dims=5, min_side=0, max_side=5).map(prep3),
    elements=floats(allow_nan=False, width=16),
)


@given(np_float_arrays, scalar_dtypes())
def test_get_dtype(x_np, dtype):
    x_ = from_np(x_np, dtype)
    dtype2 = get_dtype(x_)

    assert dtype == dtype2
    assert str(dtype) == str(dtype2)


@given(np_float_arrays, floating_dtypes())
def test_from_np(x_np, dtype):
    x_ = from_np(x_np, dtype)
    x_np2 = to_np(x_)

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


@settings(deadline=None)
@given(
    dictionaries(text(), tuples(np_float_arrays, floating_dtypes()), min_size=1),
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


@settings(deadline=None)
@given(
    dictionaries(text(), tuples(np_float_arrays3, floating_dtypes()), min_size=1),
    lists(integers()),
    grad_methods,
    floats(0, 1),
    booleans(),
    booleans(),
)
def test_minimize_bounded(x0_dict_, args, method, tol, ignore_lb, ignore_ub):
    total_el = sum(vv.size for vv, _ in x0_dict_.values())
    assume(total_el > 0)

    args = tuple(args)

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

    if ignore_lb:
        lb_dict = None
    if ignore_ub:
        ub_dict = None

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
    # TODO determine when we can check bounds
    validate_solution(x0_dict, x_sol)
