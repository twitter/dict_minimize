import unittest
from collections import OrderedDict


class DictMinimizeTestCase(unittest.TestCase):
    def test_jax(self):
        from jax import random, value_and_grad
        import jax.numpy as np
        from dict_minimize.jax_api import minimize

        def rosen(x):
            r = np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0, axis=0)
            return r

        def rosen_obj(params, shift):
            val = rosen(params["x_half_a"] - shift) + rosen(params["x_half_b"] - shift)
            return val

        n_a = 3
        n_b = 5
        shift = -1.0

        # Jax makes it this simple:
        d_rosen_obj = value_and_grad(rosen_obj, argnums=0)

        # Setup randomness in JAX
        key = random.PRNGKey(0)
        key, subkey_a = random.split(key)
        key, subkey_b = random.split(key)

        params = OrderedDict(
            [("x_half_a", random.normal(subkey_a, shape=(n_a,))), ("x_half_b", random.normal(subkey_b, shape=(n_b,)))]
        )

        params = minimize(d_rosen_obj, params, args=(shift,), method="L-BFGS-B", options={"disp": False})

    def test_numpy(self):
        import numpy as np
        from scipy.optimize import rosen, rosen_der
        from dict_minimize.numpy_api import minimize

        def rosen_obj(params, shift):
            val = rosen(params["x_half_a"] - shift) + rosen(params["x_half_b"] - shift)

            dval = OrderedDict(
                [
                    ("x_half_a", rosen_der(params["x_half_a"] - shift)),
                    ("x_half_b", rosen_der(params["x_half_b"] - shift)),
                ]
            )
            return val, dval

        np.random.seed(0)

        n_a = 3
        n_b = 5
        shift = -1.0

        params = OrderedDict([("x_half_a", np.random.randn(n_a)), ("x_half_b", np.random.randn(n_b))])

        params = minimize(rosen_obj, params, args=(shift,), method="L-BFGS-B", options={"disp": False})

    def test_tensorflow(self):
        import tensorflow as tf
        from dict_minimize.tensorflow_api import minimize

        def rosen_obj(params, shift):
            """Based on augmented Rosenbrock from botorch."""
            X, Y = params["x_half_a"], params["x_half_b"]
            X = X - shift
            Y = Y - shift
            obj = 100 * (X[1] - X[0] ** 2) ** 2 + 100 * (Y[1] - Y[0] ** 2) ** 2
            return obj

        def d_rosen_obj(params, shift):
            with tf.GradientTape(persistent=True) as t:
                t.watch(params["x_half_a"])
                t.watch(params["x_half_b"])

                obj = rosen_obj(params, shift=shift)

            da = t.gradient(obj, params["x_half_a"])
            db = t.gradient(obj, params["x_half_b"])
            d_obj = OrderedDict([("x_half_a", da), ("x_half_b", db)])
            del t  # Explicitly drop the reference to the tape
            return obj, d_obj

        tf.random.set_seed(123)

        n_a = 2
        n_b = 2
        shift = -1.0

        params = OrderedDict([("x_half_a", tf.random.normal((n_a,))), ("x_half_b", tf.random.normal((n_b,)))])

        params = minimize(d_rosen_obj, params, args=(shift,), method="L-BFGS-B", options={"disp": False})

    def test_torch(self):
        import torch
        from dict_minimize.torch_api import minimize

        def rosen_obj(params, shift):
            """Based on augmented Rosenbrock from botorch."""
            X, Y = params["x_half_a"], params["x_half_b"]
            X = X - shift
            Y = Y - shift
            obj = 100 * (X[1] - X[0] ** 2) ** 2 + 100 * (Y[1] - Y[0] ** 2) ** 2
            return obj

        def d_rosen_obj(params, shift):
            obj = rosen_obj(params, shift=shift)
            da, db = torch.autograd.grad(obj, [params["x_half_a"], params["x_half_b"]])
            d_obj = OrderedDict([("x_half_a", da), ("x_half_b", db)])
            return obj, d_obj

        torch.manual_seed(123)

        n_a = 2
        n_b = 2
        shift = -1.0

        params = OrderedDict([("x_half_a", torch.randn((n_a,))), ("x_half_b", torch.randn((n_b,)))])

        params = minimize(d_rosen_obj, params, args=(shift,), method="L-BFGS-B", options={"disp": False})
