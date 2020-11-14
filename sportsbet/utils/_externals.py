"""
Includes extensions of external packages.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from joblib import delayed, Parallel
import numpy as np
from sklearn.base import is_classifier, ClassifierMixin
from sklearn.multiclass import check_classification_targets
from sklearn.multioutput import _fit_estimator, _partial_fit_estimator
from sklearn.utils import check_array
from sklearn.ensemble._base import _BaseHeterogeneousEnsemble
from sklearn.utils.validation import has_fit_parameter, check_is_fitted, _check_fit_params


class MultiOutputClassifiers(ClassifierMixin, _BaseHeterogeneousEnsemble):
    """Multi target classification.

    Ιτ consists of fitting one classifier per target for a list of named classifiers.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples of ``n_output`` length
        Invoking the ``fit`` method on the ``MultiOutputClassifiers`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.estimators_``. An estimator can be set to ``'drop'``
        using ``set_params``.

    n_jobs : int or None, default=None
        The number of jobs to use for the computation.
        It does each target variable in y in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        
    Attributes
    ----------
    classes_ : array, shape = (n_classes,)
        Class labels.

    estimators_ : list of ``n_output`` estimators
        Estimators used for predictions.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_multilabel_classification
    >>> from sklearn.multioutput import MultiOutputClassifiers
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> X, y = make_multilabel_classification(n_classes=3, random_state=0)
    >>> clf = MultiOutputClassifiers(KNeighborsClassifier()).fit(X, y)
    >>> clf.predict(X[-2:])
    array([[1, 1, 0], [1, 1, 1]])
    """

    def __init__(self, estimators, n_jobs=None):
        self.estimators = estimators
        self.n_jobs = n_jobs

    def partial_fit(self, X, Y, classes=None, sample_weight=None):
        """Incrementally fit the model to data.
        Fit a separate model for each output variable.

        Parameters
        ----------
        X : (sparse) array-like, shape (n_samples, n_features)
            The input data.

        Y : (sparse) array-like, shape (n_samples, n_outputs)
            Multi-output targets.

        classes : list of numpy arrays, shape (n_outputs)
            Each array contains the unique classes for one output. 
            Can be obtained by via ``[np.unique(Y[:, i]) for i in
            range(Y.shape[1])]``, 
            where ``Y`` is the target matrix of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that ``Y`` doesn't need to contain all labels in ``classes``.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If ``None``, then samples are equally weighted.
            Only supported if the underlying classifier supports sample
            weights.

        Returns
        -------
        self : object
        """
        self._validate_estimators()

        for _, est in self.estimators:
            if not hasattr(est, 'partial_fit'):
                raise AttributeError('Every base estimator should implement a partial_fit method.')

        X, Y = self._validate_data(X, Y, multi_output=True, accept_sparse=True)

        if Y.ndim == 1:
            raise ValueError('Output Y must have at least two dimensions for multi-output classification but has only one.')
        
        if sample_weight is not None and any([not has_fit_parameter(clf, 'sample_weight') for _, clf in self.estimators]):
            raise ValueError('One of base estimators does not support sample weights.')

        first_time = not hasattr(self, 'estimators_')

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_partial_fit_estimator)(
                self.estimators_[i] if not first_time else clf, X, Y[:, i], 
                classes[i] if classes is not None else None, 
                sample_weight, first_time) 
                for i, (_, clf) in zip(range(Y.shape[1]), self.estimators))
        
        return self

    def fit(self, X, Y, sample_weight=None, **fit_params):
        """Fit the model to data matrix X and targets Y.

        Parameters
        ----------
        X : (sparse) array-like, shape (n_samples, n_features)
            input data.

        Y : (sparse) array-like, shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.

        sample_weight : array-like of shape (n_samples,) or None
            Sample weights. If None, then samples are equally weighted.
            Only supported if the underlying classifier supports sample
            weights.

        **fit_params : dict of string -> object
            Parameters passed to the ``estimator.fit`` method of each step.

        Returns
        -------
        self : object
        """
        self._validate_estimators()

        for _, est in self.estimators:
            if not hasattr(est, 'fit'):
                raise AttributeError('Every base estimator should implement a fit method.')

        X, Y = self._validate_data(X, Y, multi_output=True, accept_sparse=True)

        if is_classifier(self):
            check_classification_targets(Y)

        if Y.ndim == 1:
            raise ValueError('Output Y must have at least two dimensions for multi-output classification but has only one.')

        if sample_weight is not None and any([not has_fit_parameter(clf, 'sample_weight') for _, clf in self.estimators]):
            raise ValueError('One of base estimators does not support sample weights.')

        fit_params_validated = _check_fit_params(X, fit_params)
        
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(
                clf, X, Y[:, i], sample_weight, **fit_params_validated) 
            for i, (_, clf) in zip(range(Y.shape[1]), self.estimators))
        
        self.classes_ = [est.classes_ for est in self.estimators_]
        
        return self

    def predict(self, X):
        """Predict multi-output variable using a model
         trained for each target variable.

        Parameters
        ----------
        X : (sparse) array-like, shape (n_samples, n_features)
            input data.

        Returns
        -------
        Y : (sparse) array-like, shape (n_samples, n_outputs)
            Multi-output targets predicted across multiple predictors.
            Note: Separate models are generated for each predictor.
        """
        
        check_is_fitted(self)

        for _, est in self.estimators:
            if not hasattr(est, 'predict'):
                raise AttributeError('Every base estimator should implement a predict method')

        X = check_array(X, accept_sparse=True)

        Y = Parallel(n_jobs=self.n_jobs)(delayed(est.predict)(X) for est in self.estimators_)

        return np.asarray(Y).T

    def predict_proba(self, X):
        """Probability estimates.

        Returns prediction probabilities for each class of each output.
        This method will raise a ``ValueError`` if any of the
        estimators do not have ``predict_proba``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) input data.

        Returns
        -------
        p : list of length ``n_outputs`` that contains arrays.
            
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.                
        """
        
        check_is_fitted(self)

        for _, clf in self.estimators:
            if not hasattr(clf, 'predict_proba'):
                raise AttributeError('Every base should implement predict_proba method')

        X = check_array(X, accept_sparse=True)

        p = Parallel(n_jobs=self.n_jobs)(delayed(est.predict_proba)(X) for est in self.estimators_)

        return p
    
    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Test samples

        Y : array-like, shape [n_samples, n_outputs]
            True values for X

        Returns
        -------
        scores : float
            accuracy_score of self.predict(X) versus y
        """
        check_is_fitted(self)
        n_outputs_ = len(self.estimators_)
        if y.ndim == 1:
            raise ValueError("y must have at least two dimensions for "
                             "multi target classification but has only one")
        if y.shape[1] != n_outputs_:
            raise ValueError("The number of outputs of Y for fit {0} and"
                             " score {1} should be same".
                             format(n_outputs_, y.shape[1]))
        y_pred = self.predict(X)
        return np.mean(np.all(y == y_pred, axis=1))
    
    def _more_tags(self):
        return {'multioutput_only': True}
