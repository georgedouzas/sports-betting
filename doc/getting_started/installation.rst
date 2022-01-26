.. _installation:

#############
 Installation
#############

There are different ways of installing the `sports-betting` package,
depending on its usage but the simplest way is the following command:

.. code::

   pip install sports-betting

The sections below more information about the installation procedure.

***************
 Python version
***************

The package is compatible with any Python version greater or equal than
3.8. You may check your Python version by running the following command:

.. code::

   python --version

The appropriate Python version can be installed in different ways.
Besides the `Python official website
<https://www.python.org/downloads/>`_ where you can download and install
any Python version, `pyenv <https://github.com/pyenv/pyenv>`_ and
`Anaconda <https://www.anaconda.com/products/individual>`_ are also
useful Python version management tools.

********************
 Virtual Environment
********************

An optional but recommended step is to create a Python virtual
environment before installing `sports-betting`. More information can be
found `here
<https://www.freecodecamp.org/news/python-virtual-environments-explained-with-examples>`_.
In order to create a Python virtual environment, you may use the
procedure described below.

Initially, run the following command:

.. code::

   python -m venv .venv

Finally, activate the virtual environment:

.. code::

   . .venv/bin/activate

*****
 PyPi
*****

As it was explained above, you can install `sports-betting` by using
`pip` package manager since it is currently available on the PyPi:

.. code::

   pip install sports-betting

******
 Conda
******

Another option is using `conda` package manager, as the package is
released also in Anaconda Cloud platform:

.. code::

   conda install -c gdouzas sports-betting
