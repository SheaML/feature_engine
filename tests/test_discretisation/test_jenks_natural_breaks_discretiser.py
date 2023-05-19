import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.discretisation import JenksDiscretiser


# test init params
@pytest.mark.parametrize("param", [0.1, "hola", (True, False), {"a": True}, 2])
def test_raises_error_when_return_object_not_bool(param):
    with pytest.raises(ValueError):
        JenksDiscretiser(return_object=param)


@pytest.mark.parametrize("param", [0.1, "hola", (True, False), {"a": True}, 2])
def test_raises_error_when_return_boundaries_not_bool(param):
    with pytest.raises(ValueError):
        JenksDiscretiser(return_boundaries=param)


@pytest.mark.parametrize("param", [0.1, "hola", (True, False), {"a": True}, 0, -1])
def test_raises_error_when_precision_not_int(param):
    with pytest.raises(ValueError):
        JenksDiscretiser(precision=param)


@pytest.mark.parametrize("param", [0.1, "hola", (True, False), {"a": True}])
def test_raises_error_when_bins_not_int(param):
    with pytest.raises(ValueError):
        JenksDiscretiser(bins=param)


@pytest.mark.parametrize("params", [(False, 1), (True, 10)])
def test_correct_param_assignment_at_init(params):
    param1, param2 = params
    t = JenksDiscretiser(
        return_object=param1, return_boundaries=param1, precision=param2, bins=param2
    )
    assert t.return_object is param1
    assert t.return_boundaries is param1
    assert t.precision == param2
    assert t.bins == param2


def test_fit_and_transform_methods():
    transformer = JenksDiscretiser(bins=4, variables=None, return_object=False)
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    test_df = pd.DataFrame({"var": x})

    X = transformer.fit_transform(test_df)

    breaks = [0.0, 2.0, 5.0, 8.0, 11.0]
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]

    assert (X["var"] == labels).all()
    assert transformer.binner_dict_["var"] == breaks
    assert transformer.variables_ == ["var"]


def test_automatically_find_variables_and_return_as_object(df_normal_dist):
    transformer = JenksDiscretiser(bins=5, variables=None, return_object=True)
    X = transformer.fit_transform(df_normal_dist)
    assert X["var"].dtypes == "O"


def test_error_if_input_df_contains_na_in_fit(df_na):
    # test case 3: when dataset contains na, fit method
    transformer = JenksDiscretiser()
    with pytest.raises(ValueError):
        transformer.fit(df_na)


def test_error_if_input_df_contains_na_in_transform(df_vartypes, df_na):
    # test case 4: when dataset contains na, transform method
    transformer = JenksDiscretiser(bins=3)
    transformer.fit(df_vartypes)
    with pytest.raises(ValueError):
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_non_fitted_error(df_vartypes):
    transformer = JenksDiscretiser()
    with pytest.raises(NotFittedError):
        transformer.transform(df_vartypes)
