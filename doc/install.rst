.. _installation:

============
Installation
============

There are different types of installation of `sports-betting` package and its 
requirements depending on its usage.

Installation for users
----------------------

`sports-betting` is currently available on the PyPi's repositories and you can
install it via `pip`::

  pip install -U sports-betting

The package is released also in Anaconda Cloud platform::

  conda install -c gdouzas sports-betting

Installation for developers
---------------------------

You can clone it and run the setup.py file. Use the following
commands to get a copy from Github and install it in editable mode::

  git clone https://github.com/georgedouzas/sports-betting.git
  cd sports-betting
  pip install -e .

Dependencies
------------

The `sports-betting` package requires the following dependencies:

* pandas (>=1.0.0)
* scikit-learn (>=1.0.0)
* cloudpickle (>=2.0.0)
* beautifulsoup4 (>=4.0.0)
* rich (>=4.28)

They can be installed via the following command::

  pip install -r requirements.txt

Additionally, you can install the testing dependencies via the 
following command::

  pip install -r requirements.test.txt

Finally, you can install the dependencies for building 
the documentation via the following command::

  pip install -r requirements.docs.txt
