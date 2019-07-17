.. _imbalanced-learn: https://imbalanced-learn.readthedocs.io/en/stable/introduction.html#problem-statement-regarding-imbalanced-data-sets

.. _introduction:

============
Introduction
============

.. _api_gsmote:

API
---

Geometric SMOTE over-sampler follows the imbalanced-learn API using the base
over-sampler functionality. More specifically:

It implements a ``fit`` method to learn from data::

      oversampler = object.fit(data, targets)

it implements a ``fit_resample`` method to resample data sets::

      data_resampled, targets_resampled = object.fit_resample(data, targets)

Geometric SMOTE over-sampler accepts the following inputs:

* ``data``: array-like (2-D list, pandas.Dataframe, numpy.array) or sparse
  matrices;
* ``targets``: array-like (1-D list, pandas.Series, numpy.array).


Imbalanced learning problem
---------------------------

Classification of imbalanced datasets is a challenging task for standard
algorithms. Although many methods exist to address this problem in different
ways, generating artificial data for the minority class is a more general
approach compared to algorithmic modifications. For a visual representation,
the reader is referred to imbalanced-learn_.

Data generation mechanism
-------------------------

SMOTE algorithm, as well as any other over-sampling method based on the SMOTE
mechanism, generates synthetic samples along line segments that join minority
class instances. Geometric SMOTE (G-SMOTE) is an enhancement of the SMOTE data
generation mechanism. G-SMOTE generates synthetic samples in a geometric region
of the input space, around each selected minority instance. While in the basic
configuration this region is a hyper-sphere, G-SMOTE allows its deformation
to a hyper-spheroid.
