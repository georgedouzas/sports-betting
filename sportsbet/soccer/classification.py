import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import NotFittedError, check_array
from sportsbet.soccer import ODDS_FEATURES


class OddsEstimator(BaseEstimator, ClassifierMixin):
    """Predict the result based on the odds given by betting agents."""

    def fit(self, X, y):
        """No actual fitting occurs."""
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.odds_features_ = range(len(ODDS_FEATURES))
        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        C : array, shape = [n_samples]
            Predicted class label per sample.
        """

        predictions = self.predict_proba(X).argmax(axis=1)
        return predictions

    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        if not hasattr(self, "classes_"):
            raise NotFittedError("Call fit before prediction")
        X = check_array(X)
        odds = X[:, self.odds_features_].reshape(X.shape[0], -1, 3)
        probabilities = 1 / odds
        probabilities = (probabilities / probabilities.sum(axis=2)[:, :, None]).mean(axis=1)
        return probabilities

