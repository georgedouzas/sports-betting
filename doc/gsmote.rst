.. _gsmote:

===============
Geometric SMOTE
===============

.. currentmodule:: gsmote

A practical guide
-----------------

One way to fight the imbalanced learning problem is to generate
new samples in the classes which are under-represented. Douzas
and Bacao (2019) proposed Geometric SMOTE [DB2019]_. The
:class:`GeometricSMOTE` class is an implementation of the proposed
over-sampling strategy. It offers such scheme::

   >>> from collections import Counter
   >>> from sklearn.datasets import make_classification
   >>> X, y = make_classification(n_classes=3, weights=[0.10, 0.10, 0.80], random_state=0, n_informative=10)
   >>> from gsmote import GeometricSMOTE
   >>> geometric_smote = GeometricSMOTE()
   >>> X_resampled, y_resampled = geometric_smote.fit_resample(X, y)
   >>> from collections import Counter
   >>> print(sorted(Counter(y_resampled).items()))
   [(0, 80), (1, 80), (2, 80)]

The augmented data set should be used instead of the original data set
to train a classifier::

   >>> from sklearn.tree import DecisionTreeClassifier
   >>> clf = DecisionTreeClassifier()
   >>> clf.fit(X_resampled, y_resampled) # doctest : +ELLIPSIS
   DecisionTreeClassifier(...)

Sample generation
-----------------

Geometric SMOTE represents a modification of the original SMOTE algorithm
[CBHK2002]_ on the data generation mechanism. Considering a sample :math:`x_i`,
a new sample :math:`x_{new}` will be generated using its k-nearest neighbours
(corresponding to ``k_neighbors``). Contrary to SMOTE, instead of generating an
artificial observation within a segment between :math:`x_i` and one of its
k-nearest neighbours, G-SMOTE will randomly generate the new artificial
observation within a geometry, whose shape is determined by the
``deformation_factor`` and ``truncation_factor``. For more information the
reader is referred to [DB2019]_.

Multi-class management
----------------------

:class:`GeometricSMOTE` can be used with multiple classes as well as binary
classes classification. It uses a one-vs-rest approach by selecting each
targeted class and computing the necessary statistics against the rest of the
data set which are grouped in a single class.

.. topic:: References

   .. [DB2019] Douzas, G., & Bacao, F. (2019). "Geometric SMOTE: a
      geometrically enhanced drop-in replacement for SMOTE",
      Information Sciences, 501, 118–135.
      https://doi.org/10.1016/J.INS.2019.06.007

   .. [CBHK2002] N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer, "SMOTE:
      synthetic minority over-sampling technique", Journal of Artificial
      Intelligence Research, vol. 16, pp. 321-357, 2002.
