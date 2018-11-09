from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

from .utils import import_custom_classifiers

DEFAULT_CLASSIFIERS = {
    'random': (DummyClassifier(), {}),
    'baseline': (make_pipeline(SMOTE(), LogisticRegression(solver='lbfgs')), {})
}
CLASSIFIERS = import_custom_classifiers(DEFAULT_CLASSIFIERS)