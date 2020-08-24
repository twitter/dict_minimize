from collections import OrderedDict
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from dict_minimize.core._scipy import _minimize


def _get_dtype(X):
    dtype = X.dtype
    return dtype


def _from_np(X, dtype):
    assert X.dtype.kind == "f"

    dtype_ = X.dtype
    X = np.asarray(X)  # In case a scalar was given
    assert X.dtype == dtype_

    Xt = torch.from_numpy(X).type(dtype)
    # Do this weird way to avoid `UserWarning` from torch.
    Xt = Xt.clone().detach().requires_grad_(True)
    return Xt


def _to_np(X):
    Xn = X.detach().numpy()
    assert Xn.dtype.kind == "f"
    return Xn


def minimize(
    fun: Callable,
    x0_dict: OrderedDict,
    *,
    lb_dict: Optional[OrderedDict] = None,
    ub_dict: Optional[OrderedDict] = None,
    args: Sequence = (),
    method: Optional[str] = None,
    tol: Optional[float] = None,
    callback: Optional[Callable] = None,
    options: Optional[dict] = None,
) -> OrderedDict:
    """Minimization of a scalar function with a dictionary of variables as the input. It can interface to functions
    written for `torch`.

    This is a wrapper around `scipy.optimize.minimize`.

    Args:
      fun (callable): The objective function to be minimized, in the form of \
          ``fun(x, *args) -> (float, OrderedDict)`` \
        where `x` is an `OrderedDict` in the format of `x0_dict`, and `args` is a tuple of the fixed \
        parameters needed to completely specify the function. The second returned variable is the \
        gradients. It should be an `OrderedDict` with the same keys and shapes as `x`. The values \
        should be `torch` `Tensor`.
      x0_dict (OrderedDict): Initial guess. Dictionary of variables from variable name to \
        `torch` variables.
      lb_dict (OrderedDict): Dictionary with same keys and shapes as `x0_dict` with lower bounds for \
        each variable. Set to `None` in an unconstrained problem.
      ub_dict (OrderedDict): Dictionary with same keys and shapes as `x0_dict` with upper bounds for \
        each variable. Set to `None` in an unconstrained problem.
      args (tuple): Extra arguments passed to the objective function.
      method (str): Type of solver. Should be one of: ``CG``, ``BFGS``, ``L-BFGS-B``, ``TNC``, \
        ``SLSQP``, or ``trust-constr``. If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, \
        ``SLSQP``, depending if the problem has bounds. Note, only ``L-BFGS-B``, ``TNC``, ``SLSQP`` \
        seem to strictly respect the bounds ``lb_dict`` and ``ub_dict``.
      tol (float): Tolerance for termination. For detailed control, use solver-specific options.
      callback (callable): Called after each iteration. The signature is: \
          ``callback(xk)`` \
        where `xk` is the current parameter as an `OrderedDict` with the same form as the final \
        solution `x`.
      options (dict): A dictionary of solver options. All methods accept the following generic \
        options: \
          maxiter : int \
            Maximum number of iterations to perform. Depending on the method each iteration may use \
            several function evaluations. \
          disp : bool
            Set to `True` to print convergence messages.

    Returns:
      x (OrderedDict): Final solution found by the optimizer. It has the same keys and shapes as `x0_dict`.
    """
    x = _minimize(
        fun,
        x0_dict,
        from_np=_from_np,
        get_dtype=_get_dtype,
        to_np=_to_np,
        lb_dict=lb_dict,
        ub_dict=ub_dict,
        args=args,
        method=method,
        tol=tol,
        callback=callback,
        options=options,
    )
    return x
