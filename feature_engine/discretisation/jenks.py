# Authors: Shea Lambert <shea.maddock@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd

from feature_engine._docstrings.fit_attributes import (
    _binner_dict_docstring,
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.discretisers import (
    _precision_docstring,
    _return_boundaries_docstring,
    _return_object_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_not_learn_docstring,
    _fit_transform_docstring,
    _transform_discretiser_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.discretisation.base_discretiser import BaseDiscretiser

from feature_engine.variable_handling._init_parameter_checks import (
    _check_init_parameter_variables,
)

from jenkspy import JenksNaturalBreaks


# TODO: double-check all this documentation logic
@Substitution(
    return_object=_return_object_docstring,
    return_boundaries=_return_boundaries_docstring,
    precision=_precision_docstring,
    binner_dict_=_binner_dict_docstring,
    transform=_transform_discretiser_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    fit_transform=_fit_transform_docstring,
)
class JenksDiscretiser(BaseDiscretiser):
    """
    The JenksDiscretiser() divides numerical variables into intervals using the Jenks
    natural breaks algorithm.
    """

    def __init__(
        self,
        bins: int = 10,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_object: bool = False,
        return_boundaries: bool = False,
        precision: int = 3,
    ) -> None:

        if not isinstance(return_object, bool):
            raise ValueError(
                "return_object must be True or False. " f"Got {return_object} instead."
            )

        if not isinstance(return_boundaries, bool):
            raise ValueError(
                "return_boundaries must be True or False. "
                f"Got {return_boundaries} instead."
            )

        if not isinstance(precision, int) or precision < 1:
            raise ValueError(
                "precision must be a positive integer. " f"Got {precision} instead."
            )

        self.return_object = return_object
        self.return_boundaries = return_boundaries
        self.precision = precision
        if not isinstance(bins, int):
            raise ValueError(
                "bins must be an integer. " f"Got {bins} instead." f""
            )
        self.bins = bins
        self.variables = _check_init_parameter_variables(variables)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):  # type: ignore
        """
        Fit the JenksDiscretiser to the train set X.

        Parameters
        ----------

        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset. Can be the entire dataframe, not just the
            variables to be transformed.

        y: pandas series.
            Target variable - not required.
        """
        # check input dataframe
        X = super().fit(X)

        self.binner_dict_ = {}
        jnb = JenksNaturalBreaks(self.bins)

        for feature in self.variables_:
            jnb.fit(X[feature])
            self.binner_dict_[feature] = jnb.breaks_

        return self
