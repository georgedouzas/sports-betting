"""Build a betting model from a scikit-learn expression or a reference to your own."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.utils import all_estimators

from ..core import BuildError
from ..core._reference import _load_object
from ._base import BaseBettor
from ._classifier import ClassifierBettor
from ._model_selection import BettorGridSearchCV
from ._rules import OddsComparisonBettor


def _bettor_namespace() -> dict[str, object]:
    """Return the estimators an inline model expression may name."""
    namespace: dict[str, object] = dict(all_estimators())
    namespace['make_pipeline'] = make_pipeline
    namespace['make_column_transformer'] = make_column_transformer
    namespace['ClassifierBettor'] = ClassifierBettor
    namespace['OddsComparisonBettor'] = OddsComparisonBettor
    namespace['BettorGridSearchCV'] = BettorGridSearchCV
    return namespace


def build_bettor(model: str) -> BaseBettor:
    """Build a betting model from a scikit-learn expression or a reference to your own.

    Args:
        model:
            A scikit-learn estimator written as a Python expression, with the library's bettors and every
            scikit-learn estimator already in scope, as in `ClassifierBettor(LogisticRegression(C=1.0))`; or a
            bettor you built in a file, named by where it lives, as in `models.py:BETTOR`.

    Returns:
        bettor:
            The betting model, ready to fit.

    Raises:
        BuildError:
            When the expression or the reference does not describe a bettor.
    """
    if ':' in model and '(' not in model:
        built = _load_object(model)
    else:
        try:
            built = eval(model, _bettor_namespace())  # noqa: S307  # nosec B307
        except Exception as error:
            msg = (
                f'`{model}` is not a model. Write it as a scikit-learn expression, as in '
                '`OddsComparisonBettor(alpha=0.05)`, or point to one with `models.py:BETTOR`.'
            )
            raise BuildError(msg) from error
    if not isinstance(built, BaseBettor):
        msg = f'`{model}` is not a bettor.'
        raise BuildError(msg)
    return built
