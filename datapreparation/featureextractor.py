"""Module containing the FeatureExtractor class implementing
the feature extraction for the data preparation step."""
import numpy
import torch
import scipy.stats


def generate_labels(number_of_extracted_subseries):
    """Compute and return the labels denoting the extracted features.

    Args:
        number_of_extracted_subseries (int): Number of subseries effectively
            extracted for each time series.

    Returns:
        list: Labels denoting the extracted features.
    """
    labels = ["timestamp", "ANOMALOUS", "25th percentile", "50th percentile",
              "75th percentile", "90th percentile", "Maximum value", "Median",
              "Mean absolute deviation", "Skewness"]

    if number_of_extracted_subseries == 0:
        return labels

    for subseries_index in range(1, number_of_extracted_subseries + 1):
        labels.extend([
            "Mean absolute deviation difference -- subseries " + str(subseries_index),
            "Median difference -- subseries " + str(subseries_index),
            "Samples greater than 25% of global max -- subseries " + str(subseries_index),
            "Samples greater than 50% of global max -- subseries " + str(subseries_index)])

    return labels


class FeatureExtractor:
    """Class representing the feature extractor for the data preparation step.

    Attributes:
        _time_series (ndarray): The Numpy array containing the time series.
        _number_of_subseries (int): The number of subseries to extract for each time series.
        _subseries_overlap (int): The overlap of each subseries.
    """

    def _compute_percentiles(self):
        """Compute the 25th, 50th, 75th and 90th percentile of each time series.

        Returns:
            ndarray: Numpy array containing the percentiles of each time series.
        """
        return numpy.percentile(self._time_series, [25, 50, 75, 90], axis=1).transpose()

    def _compute_max_value(self):
        """Compute the maximum value of each time series.

        Returns:
            ndarray: Numpy array containing the maximum value of each time series.
        """
        return numpy.amax(self._time_series, axis=1, keepdims=True)

    def _compute_median(self):
        """Compute the median of each time series.

        Returns:
            ndarray: Numpy array containing the median of each time series.
        """
        return numpy.median(self._time_series, axis=1, keepdims=True)

    def _compute_mean_absolute_deviation(self):
        """Compute the mean absolute deviation of each time series.

        Returns:
            ndarray: Numpy array containing the mean absolute deviation
                of each time series.
        """
        deviation = self._time_series - numpy.mean(self._time_series, axis=1, keepdims=True)
        return numpy.mean(numpy.absolute(deviation), axis=1, keepdims=True)

    def _compute_skewness(self):
        """Compute the skewness of each time series.

        Returns:
            ndarray: Numpy array containing the skewness of each time series.
        """
        skewness = scipy.stats.skew(self._time_series, axis=1)
        return skewness.reshape(skewness.shape[0], 1)

    def _extract_subseries(self):
        """Extract the subseries of each time series, given a configured
        number of subseries and an overlap.

        Returns:
            ndarray/None: Numpy array containing the subseries of each time series
                if the configured parameters allow an extraction, None otherwise.
        """
        subseries_length = int(self._time_series.shape[1] /
                               (1 + (self._number_of_subseries - 1)*(1 - self._subseries_overlap)))
        step = int(subseries_length*(1 - self._subseries_overlap))

        if subseries_length == 0 or step == 0:
            return None

        return torch.from_numpy(self._time_series) \
            .unfold(dimension=1, size=subseries_length, step=step) \
            .numpy()

    def _compute_mean_absolute_deviation_difference(self, subseries):
        """For each subseries of a time series, compute the difference between
        the mean absolute deviation of the time series and the mean absolute deviation
        of the subseries.

        Args:
            subseries (ndarray): Numpy array containing the extracted
                subseries of each time series.

        Returns:
            ndarray: Numpy array containing the computed difference, for each subseries.
        """
        global_mean_absolute_deviation = self._compute_mean_absolute_deviation()

        subseries_deviation = subseries - numpy.mean(subseries, axis=2, keepdims=True)
        subseries_mean_absolute_deviation = numpy.mean(numpy.absolute(subseries_deviation), axis=2)

        return global_mean_absolute_deviation - subseries_mean_absolute_deviation

    def _compute_median_difference(self, subseries):
        """For each subseries of a time series, compute the difference between
        the median of the time series and the median of the subseries.

        Args:
            subseries (ndarray): Numpy array containing the extracted
                subseries of each time series.

        Returns:
            ndarray: Numpy array containing the computed difference, for each subseries.
        """
        global_median = self._compute_median()
        subseries_median = numpy.median(subseries, axis=2)

        return global_median - subseries_median

    def _compute_samples_greater_than_global_max(self, subseries):
        """For each subseries of a time series, compute the number of samples greater
        than 25% and 50% of the maximum value of the time series.

        Args:
            subseries (ndarray): Numpy array containing the extracted
                subseries of each time series.

        Returns:
            ndarray: Numpy array containing the computed numbers, for each subseries.
        """
        global_max = self._compute_max_value()
        global_max_25_percent = (global_max*0.25)[:, :, None]  # To allow Numpy broadcasting.
        global_max_50_percent = (global_max*0.50)[:, :, None]  # To allow Numpy broadcasting.

        samples_greater_25 = numpy.count_nonzero((subseries - global_max_25_percent)[:, :] > 0,
                                                 axis=2)
        samples_greater_50 = numpy.count_nonzero((subseries - global_max_50_percent)[:, :] > 0,
                                                 axis=2)

        return numpy.concatenate((samples_greater_25, samples_greater_50), axis=1)

    def _compute_global_features(self):
        """Call the methods used to extract features related to the time series
        and concatenate their results.

        Returns:
            ndarray: Numpy array containing the features of each time series.
        """
        return numpy.concatenate((self._compute_percentiles(),
                                  self._compute_max_value(),
                                  self._compute_median(),
                                  self._compute_mean_absolute_deviation(),
                                  self._compute_skewness()),
                                 axis=1)

    def _compute_subseries_features(self):
        """Call the methods used to extract features related to the subseries
        and concatenate their results.

        Returns:
            ndarray/None: Numpy array containing the features of each subseries
                if at least one subseries is extracted, None otherwise.
            int: Number of extracted subseries for each time series.
        """
        subseries = self._extract_subseries()

        if subseries is None:
            return None, 0

        return numpy.concatenate((self._compute_mean_absolute_deviation_difference(subseries),
                                  self._compute_median_difference(subseries),
                                  self._compute_samples_greater_than_global_max(subseries)),
                                 axis=1), subseries.shape[1]

    def extract(self):
        """Extract the features associated to subseries and time series.

        Returns:
            ndarray: Numpy array containing the extracted features.
        """
        global_features = self._compute_global_features()
        subseries_features, number_of_extracted_subseries = self._compute_subseries_features()

        labels = generate_labels(number_of_extracted_subseries)

        if number_of_extracted_subseries == 0:
            return global_features, labels

        return numpy.concatenate((global_features, subseries_features), axis=1), labels

    def __init__(self, time_series):
        """Initializer.

        Args:
            time_series (ndarray): Numpy array containing the time series.
        """
        self._time_series = time_series
        self._number_of_subseries = 8
        self._subseries_overlap = 0.5
