"""Module containing the OnlineMinMaxScaler class implementing
the online data scaler for the data preparation service."""
import numpy


class OnlineMinMaxScaler:
    """An online scaler supporting min-max scaling.

    Attributes:
        _min (ndarray): The Numpy array containing the current minimum values.
            It has shape (1, number_of_features).
        _max (ndarray): The Numpy array containing the current maximum values.
            It has shape (1, number_of_features).
    """

    def fit_transform(self, features):
        """Scale the given features using online min-max scaling.
        The minimum and maximum values are kept and updated at each transformation.

        Args:
            features (ndarray): Numpy array containing the features to scale.

        Returns:
            ndarray: Numpy array containing the scaled features.
        """
        if self._min is None:
            self._min = numpy.min(features, axis=0, keepdims=True)
        else:
            self._min = numpy.minimum(self._min, numpy.min(features, axis=0, keepdims=True))

        if self._max is None:
            self._max = numpy.max(features, axis=0, keepdims=True)
        else:
            self._max = numpy.maximum(self._max, numpy.max(features, axis=0, keepdims=True))

        max_min_difference = self._max - self._min
        max_min_difference[max_min_difference == 0] = 1

        return (features - self._min)/max_min_difference

    def __init__(self):
        """Initializer."""
        self._min = None
        self._max = None
