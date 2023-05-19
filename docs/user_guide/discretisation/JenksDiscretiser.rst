.. _jenks_discretiser:

.. currentmodule:: feature_engine.discretisation

JenksDiscretiser
=========================

The :class:`JenksDiscretiser()` divides continuous numerical variables into
intervals determined by the Fisher-Jenks algorithm.

Note: The number of bins requested must not exceed the number of unique values in
the variable. If this is the case, an error will be raised.

Note: The width of some bins might be very small. Thus, to allow this transformer
to work properly, it might help to increase the precision value, that is,
the number of decimal values allowed to define each bin. If the variable has a
narrow range or you are sorting into several bins, allow greater precision
(i.e., if precision = 3, then 0.001; if precision = 7, then 0.0001).

The :class:`JenksDiscretiser()` works only with numerical variables. A list of
variables to discretise can be indicated, or the discretiser will automatically select
all numerical variables in the train set.
