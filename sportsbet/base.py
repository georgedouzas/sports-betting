"""
Contains various base classes.
"""

from datetime import datetime as dt

import numpy as np
from sklearn.utils import Parallel, delayed
from sklearn.base import ClassifierMixin, clone
from sklearn.preprocessing import label_binarize
from sklearn.utils.metaestimators import _BaseComposition


class BaseDataSource:
    """Base class of a data source."""
    
    def download(self):
        """Download the data source."""    
        self.content_ = []
        return self

    def transform(self):
        """Transform the data source."""
    
        return self.content_.copy()

    def download_transform(self):
        """Download and transform the data source."""
        return self.download().transform()


class BaseDataLoader:
    """Base class of a data loader"""

    @property
    def training_data(self):
        """Generate the training data."""
        self.training_time_stamp_ = dt.now().ctime()

    @property
    def fixtures_data(self):
        """Generate the fixtures data."""
        self.fixtures_time_stamp_ = dt.now().ctime()


class MultiClassifier(_BaseComposition, ClassifierMixin):

    def __init__(self, classifiers):
        self.classifiers = classifiers

    @staticmethod
    def _fit_binary_classifier(X, y, classifier):
        """Fit binary classifier."""
        classifier.fit(X, y)
        return classifier

    def fit(self, X, y):
        """Parallel fit of classifiers."""
        self.classes_ = np.unique(y)
        y_bin = label_binarize(y, classes=self.classes_)
        self.classifiers_ = Parallel(n_jobs=-1)(delayed(self._fit_binary_classifier)(X, y, clone(classifier)) 
                                                for classifier, y in zip(self.classifiers, y_bin.T.tolist()))
        return self

    def predict_proba(self, X):
        """Predict the class probabilities."""
        probs = np.concatenate([clf.predict_proba(X)[:, 1].reshape(-1, 1) for clf in self.classifiers_], axis=1)
        return probs

    def predict(self, X):
        """Predict the results."""
        classes = np.array([self.classes_[ind] for ind in self.predict_proba(X).argmax(axis=1)])
        return classes
