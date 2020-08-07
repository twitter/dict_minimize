*************************
The Dict Minimize Package
*************************

Access ``scipy`` optimizers from your favorite deep learning framework.

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
   git clone https://github.com/twitter/dict-minimize.git

Inside your virtual environments folder ``$ENVS``, make the environment:

.. code-block:: bash

   cd $ENVS
   virtualenv dict_minimize --python=python3.7
   source $ENVS/dict-minimize/bin/activate

Now we can install the pip dependencies. Move back into your git directory and run

.. code-block:: bash

   cd $GIT/dict-minimize
   pip install -r requirements/base.txt
   pip install -e .  # Install the package itself

Contributor tools
-----------------

First, we need to setup some needed tools:

.. code-block:: bash

   cd $ENVS
   virtualenv dict_minimize_tools --python=python3.7
   source $ENVS/dict-minimize_tools/bin/activate
   pip install -r $GIT/dict-minimize/requirements/tools.txt

To install the pre-commit hooks for contributing run (in the ``dict_minimize_tools`` environment):

.. code-block:: bash

   cd $GIT/dict-minimize
   pre-commit install

To rebuild the requirements, we can run:

.. code-block:: bash

   cd $GIT/dict-minimize

   # Check if there any discrepancies in the .in files
   pipreqs dict_minimize/core/ --diff requirements/base.in
   pipreqs dict_minimize/ --diff requirements/demos.in
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
   source $ENVS/dict-minimize_docs/bin/activate
   pip install -r $GIT/dict-minimize/requirements/docs.txt

Then we can do the build:

.. code-block:: bash

   cd $GIT/dict-minimize/docs
   make all
   open _build/html/index.html

Documentation will be available in all formats in ``Makefile``. Use ``make html`` to only generate the HTML documentation.

Running the tests
-----------------

The tests for this package can be run with:

.. code-block:: bash

   cd $GIT/dict-minimize
   ./local_test.sh

The script creates an environment using the requirements found in ``requirements/test.txt``.
A code coverage report will also be produced in ``$GIT/dict-minimize/htmlcov/index.html``.

Deployment
----------

The wheel (tar ball) for deployment as a pip installable package can be built using the script:

.. code-block:: bash

   cd $GIT/dict-minimize/
   ./build_wheel.sh

Links
=====

The `source <https://github.com/twitter/dict-minimize>`_ is hosted on GitHub.

The `documentation <>`_ is hosted at Read the Docs.

Installable from `PyPI <>`_.

License
=======

This project is licensed under the Apache 2 License - see the LICENSE file for details.
