# Authors: Shea Lambert <shea.maddock@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd
from jenkspy import JenksNaturalBreaks

from feature_engine._docstrings.fit_attributes import (
    _binner_dict_docstring, _feature_names_in_docstring,
    _n_features_in_docstring, _variables_attribute_docstring)
from feature_engine._docstrings.init_parameters.discretisers import (
    _precision_docstring, _return_boundaries_docstring,
    _return_object_docstring)
from feature_engine._docstrings.methods import (
    _fit_not_learn_docstring, _fit_transform_docstring,
    _transform_discretiser_docstring)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.discretisation.base_discretiser import BaseDiscretiser
from feature_engine.variable_handling._init_parameter_checks import \
    _check_init_parameter_variables


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
    The JenksDiscretiser() divides continuous numerical variables into intervals or bins
    using the Jenks Natural Breaks algorithm. This algorithm aims to minimize the
    variance within each interval while maximizing the variance between intervals.

    The Jenks Natural Breaks algorithm iteratively finds the best break points that
    minimize the sum of squared deviations from the class means. It starts by
    initializing the break points at equal intervals, and then optimizes their
    positions to achieve the desired number of intervals (bins).

    The `JenksDiscretiser()` works well when the distribution of the variable exhibits
    natural groupings or thresholds.

    Note: The number of bins (intervals) must be specified using the `bins` parameter,
    and there must be at least as many unique values in every variable to be transformed
    as there are bins.

    The :class:`JenksDiscretiser()` works only with numerical variables. You can
    indicate a list of variables to discretize, or the discretiser will automatically
    select all numerical variables in the training set.

    Parameters
    ----------
    {variables_}

    bins: int, default=10
        Desired number of intervals/bins.

    {return_object}

    {return_boundaries}

    {precision}

    Attributes
    ----------
    {binner_dict_}

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    {transform}

    References
    ----------
    .. [1] Jenks Natural Breaks Classification Method
       https://en.wikipedia.org/wiki/Jenks_natural_breaks_optimization

    .. [2] Jenks, G. F. (1977). "Optimal Data Classification for Choropleth Maps".
       https://www.georgefisher.com/research/jenks.pdf
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

        if not isinstance(bins, int) or bins < 1:
            raise ValueError(
                "bins must be a positive integer. " f"Got {bins} instead." f""
            )

        self.return_object = return_object
        self.return_boundaries = return_boundaries
        self.precision = precision
        self.bins = bins
        self.variables = _check_init_parameter_variables(variables)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
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
