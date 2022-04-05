.. _python: https://www.python.org/downloads
.. _pyenv: https://github.com/pyenv/pyenv
.. _anaconda: https://www.anaconda.com/products/individual

.. _installation:

############
Installation
############

There are different ways of installing the `sports-betting` package,
depending on its usage, but once all requirements are met 
the simplest way is the following command::

   $ pip install sports-betting

The sections below provide more information about the installation procedure.

**************
Python version
**************

The package is compatible with any Python version greater or equal than
3.8. You may check your Python version by running the following command::

   $ python --version

The appropriate Python version can be installed in different ways.
Besides the Python_ official website where you can download and install
any Python version, pyenv_ and Anaconda_ are also useful Python 
version management tools.

********************
 Virtual environment
********************

An optional but recommended step is to create a Python virtual environment 
before installing `sports-betting`. You may use the procedure described 
below or for more information refer `here
<https://www.freecodecamp.org/news/python-virtual-environments-explained-with-examples>`_.

In order to create a Python virtual environment first install a supported `Python version`_ 
and then run the following command::

   $ python -m venv .venv

Finally, activate the virtual environment::

   $ source .venv/bin/activate

****
PyPi
****

As it was mentioned beofre, you can install `sports-betting` by using
the `pip` package manager, since it is available on the PyPi::

   $ pip install sports-betting

