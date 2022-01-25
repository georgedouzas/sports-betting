#########
 Testing
#########

Various types of tests are included. Please refer to :ref:`installation`
section for the installation of test requirements.

******
 Code
******

Testing the code:

.. code::

   $ make test-code

You can also use `pytest`:

.. code::

   $ pytest sportsbet -v

**********
 Coverage
**********

Test the coverage of the code:

.. code::

   $ make test-coverage

***************
 Documentation
***************

Test the documentation examples:

.. code::

   $ make test-doc

*****
 All
*****

Run all tests:

.. code::

   $ make test
