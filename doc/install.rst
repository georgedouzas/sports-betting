.. _installation:

============
Installation
============

There are different ways of installing the `sports-betting` package, depending on its usage 
but the simplest way is the following command::

  pip install sports-betting

The package is compatible with any Python version greater or equal than 3.8. You may check 
your Python version by running the following command::

  python --version
 
The appropriate Python version can be installed in different ways. Besides the `Python official
website <https://www.python.org/downloads/>`_ where you can download and install any Python 
version, `pyenv <https://github.com/pyenv/pyenv>`_ and 
`Anaconda <https://www.anaconda.com/products/individual>`_ are also useful version management tools.

Users
-----

This is the most basic installation for the users of `sports-betting`. An optional but recommended 
step is to create a Python virtual environment before installing `sports-betting`. More information 
can be found `here <https://www.freecodecamp.org/news/python-virtual-environments-explained-with-examples>`_.

Python virtual environment
**************************

In order to create a Python virtual environment, you may use the procedure described below. 

Initially, run the following command::

  python -m venv .venv

Activate the new virtual environment::

  . .venv/bin/activate
  
PyPi
****

As it was explained above, you can install `sports-betting` by using `pip` package manager since it is 
currently available on the PyPi::

  pip install sports-betting

Conda
*****

Another option is using `conda` package manager, as the package is released also in Anaconda Cloud platform::

  conda install -c gdouzas sports-betting

Developers
----------

You can clone it and run the setup.py file. Use the following
commands to get a copy from Github and install it in editable mode::

  git clone https://github.com/georgedouzas/sports-betting.git
  cd sports-betting
  pip install -e .

In case you would like to contribute to the project then fork the
repository and substitute the first command by cloning your own 
repository::

  git clone https://github.com/yourrepo/sports-betting.git
  cd sports-betting
  pip install -e .

Then you make any code changes, make sure that they pass all the tests 
and open a Pull Request.

Main dependencies
*****************

The `sports-betting` package requires the following dependencies:

* pandas (>=1.0.0)
* scikit-learn (>=1.0.0)
* cloudpickle (>=2.0.0)
* beautifulsoup4 (>=4.0.0)
* rich (>=4.28)

They can be installed via the following command::

  pip install -r requirements.txt

Testing dependencies
********************

Additionally, you can install the testing dependencies via the 
following command::

  pip install -r requirements.test.txt

Documentation dependencies
**************************

Finally, you can install the dependencies for building 
the documentation via the following command::

  pip install -r requirements.docs.txt
