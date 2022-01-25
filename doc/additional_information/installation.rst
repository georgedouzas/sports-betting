##############
 Installation
##############

************
 Developers
************

You may clone the `sports-betting` repo:

.. code::

   git clone https://github.com/georgedouzas/sports-betting.git

In case you would like to contribute to the project then fork the
repository and clone your own repo:

.. code::

   git clone https://github.com/yourrepo/sports-betting.git

In both cases, you may install the project in editable mode by running
the following commands:

.. code::

   cd sports-betting
   pip install -e .

Then make any code changes, make sure that they pass all the tests and
open a Pull Request.

Main dependencies
=================

The `sports-betting` package requires the following dependencies:

-  pandas (>=1.0.0)
-  scikit-learn (>=1.0.0)
-  cloudpickle (>=2.0.0)
-  beautifulsoup4 (>=4.0.0)
-  rich (>=4.28)

They can be installed via the following command:

.. code::

   pip install -r requirements.txt

Testing dependencies
====================

Additionally, you can install the testing dependencies via the following
command:

.. code::

   pip install -r requirements.test.txt

Documentation dependencies
==========================

Finally, you can install the dependencies for building the documentation
via the following command:

.. code::

   pip install -r requirements.docs.txt
