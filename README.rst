.. figure:: https://user-images.githubusercontent.com/28273671/90945755-c2935580-e3db-11ea-9ba9-dbb054834b02.png
   :alt: Rosenbrock

*************************
The Dict Minimize Package
*************************

Access ``scipy`` optimizers from your favorite deep learning framework.

Installation
============

Only ``Python>=3.6`` is officially supported, but older versions of Python likely work as well.

The core package itself can be installed with:

.. code-block:: bash

   pip install dict_minimize

To also get the dependencies for all the supported frameworks (torch, JAX, tensorflow) in the README install with

.. code-block:: bash

   pip install dict_minimize[framework]

See the `GitHub <https://github.com/twitter/dict_minimize>`_, `PyPI <https://pypi.org>`_, and `Read the Docs <https://readthedocs.org>`_.

Example Usage
=============

In these examples we optimize a modified `Rosenbrock <https://en.wikipedia.org/wiki/Rosenbrock_function>`_ function.
However, the arguments have been split into two chunks and stored as two entries in a dictionary.
This is to illustrate how this package optimizes *dictionaries* of (tensor) parameters rather then vectors.
We also pass in an extra ``shift`` argument to demonstrate how ``minimize`` allows extra constant arguments to be passed into the objective.

PyTorch
-------

.. code-block:: python

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

    params = minimize(d_rosen_obj, params, args=(shift,), method="L-BFGS-B", options={"disp": True})

TensorFlow
----------

.. code-block:: python

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

    params = minimize(d_rosen_obj, params, args=(shift,), method="L-BFGS-B", options={"disp": True})

NumPy
-----

.. code-block:: python

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

    params = minimize(rosen_obj, params, args=(shift,), method="L-BFGS-B", options={"disp": True})

JAX
---

.. code-block:: python

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

    params = minimize(d_rosen_obj, params, args=(shift,), method="L-BFGS-B", options={"disp": True})

Contributing
============

The following instructions have been tested with Python 3.7.4 on Mac OS (10.14.6).

Install in editable mode
------------------------

First, define the variables for the paths we will use:

.. code-block:: bash

   GIT=/path/to/where/you/put/repos
   ENVS=/path/to/where/you/put/virtualenvs

Then clone the repo in your git directory ``$GIT``:

.. code-block:: bash

   cd $GIT
   git clone https://github.com/twitter/dict_minimize.git

Inside your virtual environments folder ``$ENVS``, make the environment:

.. code-block:: bash

   cd $ENVS
   virtualenv dict_minimize --python=python3.7
   source $ENVS/dict_minimize/bin/activate

Now we can install the pip dependencies. Move back into your git directory and run

.. code-block:: bash

   cd $GIT/dict_minimize
   pip install -r requirements/base.txt
   pip install -e .  # Install the package itself

Contributor tools
-----------------

First, we need to setup some needed tools:

.. code-block:: bash

   cd $ENVS
   virtualenv dict_minimize_tools --python=python3.7
   source $ENVS/dict_minimize_tools/bin/activate
   pip install -r $GIT/dict_minimize/requirements/tools.txt

To install the pre-commit hooks for contributing run (in the ``dict_minimize_tools`` environment):

.. code-block:: bash

   cd $GIT/dict_minimize
   pre-commit install

To rebuild the requirements, we can run:

.. code-block:: bash

   cd $GIT/dict_minimize

   # Check if there any discrepancies in the .in files
   pipreqs dict_minimize/core/ --diff requirements/base.in
   pipreqs dict_minimize/ --diff requirements/frameworks.in
   pipreqs tests/ --diff requirements/tests.in
   pipreqs docs/ --diff requirements/docs.in

   # Regenerate the .txt files from .in files
   pip-compile-multi --no-upgrade

Generating the documentation
----------------------------

First setup the environment for building with ``Sphinx``:

.. code-block:: bash

   cd $ENVS
   virtualenv dict_minimize_docs --python=python3.7
   source $ENVS/dict_minimize_docs/bin/activate
   pip install -r $GIT/dict_minimize/requirements/docs.txt

Then we can do the build:

.. code-block:: bash

   cd $GIT/dict_minimize/docs
   make all
   open _build/html/index.html

Documentation will be available in all formats in ``Makefile``. Use ``make html`` to only generate the HTML documentation.

Running the tests
-----------------

The tests for this package can be run with:

.. code-block:: bash

   cd $GIT/dict_minimize
   ./local_test.sh

The script creates an environment using the requirements found in ``requirements/test.txt``.
A code coverage report will also be produced in ``$GIT/dict_minimize/htmlcov/index.html``.

Deployment
----------

The wheel (tar ball) for deployment as a pip installable package can be built using the script:

.. code-block:: bash

   cd $GIT/dict_minimize/
   ./build_wheel.sh

This script will only run if the git repo is clean, i.e., first run ``git clean -x -ff -d``.

Links
=====

The `source <https://github.com/twitter/dict_minimize>`_ is hosted on GitHub.

The `documentation <https://readthedocs.org>`_ is hosted at Read the Docs.

Installable from `PyPI <https://pypi.org>`_.

License
=======

This project is licensed under the Apache 2 License - see the LICENSE file for details.
