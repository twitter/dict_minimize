"""The dict minimize core routines.

Utility for minimization of a scalar function with a dictionary of variables as the input. It can
interface to functions written outside of `numpy` (e.g., `tensorflow`, `torch` or `jax`).

This is a wrapper around `scipy.optimize.minimize`.
"""

from collections import OrderedDict
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize as scipy_minimize

SCIPY_DTYPE = np.float64  # Some scipy optimizers prefer double

_default_to_np = np.asarray


def _pack(
    x_dict: OrderedDict, to_np_validated: Callable, shapes: Optional[OrderedDict] = None
) -> Tuple[np.ndarray, OrderedDict]:
    """Pack a dictionary of variables (of any shape) into a 1D `ndarray`."""
    assert isinstance(x_dict, OrderedDict)

    build_shapes = False
    if shapes is None:
        build_shapes = True
        shapes = OrderedDict()

    v = []
    for kk in x_dict:
        xx = to_np_validated(x_dict[kk])

        if build_shapes:
            shapes[kk] = xx.shape
        else:
            ss = shapes[kk]
            assert xx.shape == tuple(ss)

        v.append(xx.ravel(order="C"))
    x_vec = np.concatenate(v)
    assert x_vec.dtype == SCIPY_DTYPE

    total_size = np.sum([np.prod(ss, dtype=int) for ss in shapes.values()])
    assert x_vec.shape == (total_size,)
    assert list(x_dict.keys()) == list(shapes.keys())

    return x_vec, shapes


def _unpack(x_vec: np.ndarray, from_np: Callable, shapes: OrderedDict, dtypes: OrderedDict) -> OrderedDict:
    """Invert `_pack` and get the dictionary of variables (in their original shapes) back."""
    assert isinstance(shapes, OrderedDict)

    total_size = np.sum([np.prod(ss, dtype=int) for ss in shapes.values()])
    assert x_vec.shape == (total_size,)

    start = 0
    x_dict = OrderedDict()
    for kk in shapes:
        ss = shapes[kk]
        end = start + np.prod(ss, dtype=int)

        x_dict[kk] = from_np(np.reshape(x_vec[start:end], ss, order="C"), dtypes[kk])
        start = end
    assert start == total_size
    assert list(x_dict.keys()) == list(shapes.keys())
    return x_dict


def _minimize(
    fun: Callable,
    x0_dict: OrderedDict,
    *,
    from_np: Callable,
    get_dtype: Callable,
    lb_dict: Optional[OrderedDict] = None,
    ub_dict: Optional[OrderedDict] = None,
    to_np: Callable = _default_to_np,
    args: Sequence = (),
    method: Optional[str] = None,
    tol: Optional[float] = None,
    callback: Optional[Callable] = None,
    options: Optional[dict] = None,
) -> OrderedDict:
    """Minimization of a scalar function with a dictionary of variables as the input. It can interface to functions
    written outside of `numpy` (e.g., `tensorflow` or `torch`).

    This is a wrapper around `scipy.optimize.minimize`.

    Args:
      fun (callable): The objective function to be minimized, in the form of
          ``fun(x, *args) -> (float, OrderedDict)``
        where `x` is an `OrderedDict` in the format of `x0_dict`, and `args` is a tuple of the fixed
        parameters needed to completely specify the function. The second returned variable is the
        gradients. It should be an `OrderedDict` with the same keys and shapes as `x`. The values do
        *not* need to be `numpy` `ndarray` variables. The conversion to `ndarray` is taken care of by
        the `to_np` function.
      x0_dict (OrderedDict): Initial guess. Dictionary of variables from variable name to float
        array, not necessarily `numpy`.
      from_np (callable): Function to convert `numpy` `ndarray` into whatever the preferred type is
        for `fun`, e.g., `torch.Tensor` for `torch`.
      get_dtype (callable): Function get the dtype from each element of `x0_dict`. This will be
        different depending on the framework used.
      lb_dict (OrderedDict): Dictionary with same keys and shapes as `x0_dict` with lower bounds for
        each variable. Set to `None` in an unconstrained problem.
      ub_dict (OrderedDict): Dictionary with same keys and shapes as `x0_dict` with upper bounds for
        each variable. Set to `None` in an unconstrained problem.
      to_np (callable): Inverse of `from_np`. Convert variable from format for `fun` (e.g.,
        `torch.Tensor`) to a `numpy` `ndarray`.
      args (tuple): Extra arguments passed to the objective function.
      method (str): Type of solver. Should be one of
          - 'Nelder-Mead'
          - 'Powell'
          - 'CG'
          - 'BFGS'
          - 'Newton-CG'
          - 'L-BFGS-B'
          - 'TNC'
          - 'COBYLA'
          - 'SLSQP'
          - 'trust-constr'
          - 'dogleg'
          - 'trust-ncg'
          - 'trust-exact'
          - 'trust-krylov'
          - custom - a callable object
        If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``, depending if the problem
        has constraints or bounds.
      tol (float): Tolerance for termination. For detailed control, use solver-specific options.
      callback (callable): Called after each iteration. For 'trust-constr' it is a callable with the
        signature:
          ``callback(xk, state) -> bool``
        where `xk` is the current parameter vector. and `state` is an `OptimizeResult` object, with
        the same fields as the ones from the return. If callback returns `True` the algorithm
        execution is terminated. For all the other methods, the signature is:
          ``callback(xk)``
        where `xk is the current parameter vector.
      options (dict): A dictionary of solver options. All methods accept the following generic
        options:
          maxiter : int
            Maximum number of iterations to perform. Depending on the method each iteration may use
            several function evaluations.
          disp : bool
            Set to `True` to print convergence messages.

    Returns:
      x (OrderedDict): Final solution found by the optimizer. It has the same keys and shapes as
        `x0_dict`.
    """
    assert isinstance(x0_dict, OrderedDict)
    assert len(x0_dict) > 0, "Cannot optimize empty dictionary."

    def to_np_validated(x):
        x_np = to_np(x)
        # These asserts could be made exceptions for user's sake
        assert isinstance(x_np, np.ndarray), "Numpy conversion must return ndarray."
        assert x_np.dtype.kind == "f", "Numpy conversion must return float ndarray."
        x_np = x_np.astype(SCIPY_DTYPE, copy=False)
        return x_np

    def fun_wrap(x):
        # Put back in dict form for objective
        x_dict = _unpack(x, from_np, shapes, x0_dtypes)

        # Evaluate objective
        val, dx_dict = fun(x_dict, *args)

        # Re-pack into vec form
        val = to_np_validated(val)
        dval, _ = _pack(dx_dict, to_np_validated, shapes)

        # Put objective as native type to ensure as simple as possible for scipy
        # This assert could be made an exception
        assert val.shape == (), "Objective function must return scalar."
        val = val.item()

        # Validate all again
        assert isinstance(val, float)
        assert isinstance(dval, np.ndarray)
        assert dval.dtype == SCIPY_DTYPE
        return val, dval

    def callback_wrap(xk, *args_, **kwargs):
        x_dict = _unpack(xk, from_np, shapes, x0_dtypes)
        callback(x_dict)

    # Setup any callback wrapper
    callback_ = None if callback is None else callback_wrap

    # Get shapes and dtypes
    x0, shapes = _pack(x0_dict, to_np_validated)
    assert x0.size > 0, "Cannot optimize 0-dim space."
    x0_dtypes = OrderedDict([(kk, get_dtype(vv)) for kk, vv in x0_dict.items()])

    # Flatten bounds
    assert (lb_dict is None) == (ub_dict is None), "Either both or neither of lb_dict and lb_dict can be None."
    if (lb_dict is None) and (ub_dict is None):
        bounds = None
    else:
        lb, _ = _pack(lb_dict, to_np_validated, shapes)
        ub, _ = _pack(ub_dict, to_np_validated, shapes)
        bounds = list(zip(lb.tolist(), ub.tolist()))

    # args=() because used in wrapper directly, wrapper returns (val, dval) => use jac=True
    res = scipy_minimize(
        fun_wrap, x0, args=(), method=method, jac=True, bounds=bounds, tol=tol, callback=callback_, options=options
    )
    x = _unpack(res.x, from_np, shapes, x0_dtypes)
    assert list(x.keys()) == list(x0_dict.keys())
    return x
