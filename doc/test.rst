=======
Testing
=======

Various types of tests are included. Please refer to :ref:`installation` section for 
the installation of test requirements.

Code
----

Testing the code::

    $ make test-code

You can also use `pytest`::

    $ pytest sportsbet -v

Coverage
--------

Test the coverage of the code::

    $ make test-coverage

Documentation
-------------

Test the documentation examples::

    $ make test-doc

All
---

Run all tests::

    $ make test
