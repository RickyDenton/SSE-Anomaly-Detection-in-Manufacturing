"""Module containing the DataSegregationService class
implementing the data segregation step of the pipeline."""
from math import pi
import sys
import logging
import matplotlib.pyplot
import seaborn
import pandas
import numpy
import coloredlogs
from sklearn.model_selection import train_test_split
from pkg_resources import resource_filename
from configuration import Configuration


class DataSegregationService:
    """Class representing the data segregation step of the pipeline.

    Attributes:
        _configuration (Configuration): The configuration of the service.
        _features_path (str): The absolute path of the file containing the prepared features.
        _class_report_path (str): The absolute path of the file that will contain the report
            about class balancing.
        _feature_space_anomalous_report_path (str): The absolute path of the file that will contain
            the report about the feature space composition of the data classified as anomalous.
        _feature_space_regular_report_path (str): The absolute path of the file that will contain
            the report about the feature space composition of the data classified as regular.
        _training_set_path (str): The absolute path of the file that will contain
            the training set.
        _validation_set_path (str): The absolute path of the file that will contain
            the validation set.
        _test_set_path (str): The absolute path of the file that will contain the test set.
        _features (DataFrame): The Pandas dataframe representing the features read
            in the current step.
        _logger (Logger): The logger used in the module.
    """

    def _initialize_logger(self):
        """Initialize the logger."""
        self._logger = logging.getLogger(__name__)

        handler = logging.StreamHandler(sys.stdout)
        formatter = coloredlogs.ColoredFormatter("%(asctime)s %(name)s"
                                                 " %(levelname)s %(message)s",
                                                 "%Y-%m-%d %H:%M:%S")

        handler.setFormatter(formatter)

        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def _retrieve_features(self):
        """Retrieve the prepared features, if available, and save them in self._ingested_data.
        Features associated to unclassified time series are discarded. A classified time series is
        either marked with 0 (regular shift) or with 1 (anomalous shift).
        """
        try:
            # Overall features as a Pandas dataframe.
            self._features = pandas.read_csv(self._features_path)

            # Discard timestamp column.
            self._features = self._features.loc[:, self._features.columns != "timestamp"]

            # Discard unclassified features.
            self._features = self._features[(self._features["ANOMALOUS"] == 0) |
                                            (self._features["ANOMALOUS"] == 1)]

        except OSError:
            # File not found or file cannot be opened.
            self._features = pandas.DataFrame()

    def _generate_class_balancing_report(self):
        """Generate and save a report about the class composition of the available data."""
        anomalous_count = numpy.count_nonzero(self._features["ANOMALOUS"])
        regular_count = self._features["ANOMALOUS"].shape[0] - anomalous_count

        report_data = pandas.DataFrame([{"Type of shift": "Anomalous",
                                         "Number of samples": anomalous_count},
                                        {"Type of shift": "Regular",
                                         "Number of samples": regular_count}])

        threshold = max(anomalous_count, regular_count) *\
                       (self._configuration.field["classBalanceThreshold"]/100)

        figure = matplotlib.pyplot.figure(figsize=(19.20, 10.80))
        barplot = seaborn.barplot(x="Type of shift", y="Number of samples", data=report_data,
                                  palette=["firebrick", "cornflowerblue"])
        barplot.axhline(threshold, linestyle="--", color="black", label="Class balance threshold")
        barplot.legend(loc="upper left", prop={'size': 14})

        matplotlib.pyplot.savefig(self._class_report_path, format="png",
                                  dpi=300, bbox_inches='tight')
        matplotlib.pyplot.close(figure)

    def _generate_feature_space_report(self, classification):
        """Generate and save a report about the feature space composition of the data
        classified either as regular or as anomalous.

        Args:
            classification (str): String representing the class of data described by the report.
                It must be either "regular" or "anomalous".
        """
        if classification == "regular":
            features = self._features[self._features["ANOMALOUS"] == 0]
            title = "Feature space - Regular"
            path = self._feature_space_regular_report_path
        elif classification == "anomalous":
            features = self._features[self._features["ANOMALOUS"] == 1]
            title = "Feature space - Anomalous"
            path = self._feature_space_anomalous_report_path
        else:
            return

        # Drop the classification column.
        features = features.loc[:, features.columns != "ANOMALOUS"]

        # Main elements of the radar chart.
        labels = features.columns.tolist()
        radius = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]

        figure = matplotlib.pyplot.figure(figsize=(20.48, 10.80))
        axes = figure.add_subplot(111, projection="polar")

        for i, label in enumerate(labels):
            values = numpy.array(features[label])
            current_radius = numpy.full(shape=values.shape[0], fill_value=radius[i])
            axes.plot(current_radius, values, "s", marker=".", label=label)

        matplotlib.pyplot.xticks(radius, labels, color='black', size=3)
        axes.tick_params(pad=35)

        matplotlib.pyplot.title(title)
        matplotlib.pyplot.savefig(path, format="png", dpi=300, bbox_inches='tight')
        matplotlib.pyplot.close(figure)

    def _segregate_sets(self):
        """Extract training, validation and test sets given the available features.

        Returns:
            DataFrame/None: DataFrame representing the training set
                if the extraction succeeded, None otherwise.
            DataFrame/None: DataFrame representing the validation set
                if the extraction succeeded, None otherwise.
            DataFrame/None: DataFrame representing the test set
                if the extraction succeeded, None otherwise.
            bool: True if the extraction succeeded, False otherwise.
        """
        training_ratio = self._configuration.field["trainingSetSize"] / 100
        validation_ratio = self._configuration.field["validationSetSize"] / 100
        test_ratio = self._configuration.field["testSetSize"] / 100

        try:
            training_set, intermediate_set, training_set_target, intermediate_set_target = \
                train_test_split(self._features.loc[:, self._features.columns != "ANOMALOUS"],
                                 self._features["ANOMALOUS"],
                                 test_size=(1 - training_ratio),
                                 stratify=self._features["ANOMALOUS"])

            validation_set, test_set, validation_set_target, test_set_target = \
                train_test_split(intermediate_set, intermediate_set_target,
                                 test_size=test_ratio / (test_ratio + validation_ratio),
                                 stratify=intermediate_set_target)

            return pandas.concat([training_set_target, training_set], axis=1),\
                pandas.concat([validation_set_target, validation_set], axis=1),\
                pandas.concat([test_set_target, test_set], axis=1),\
                True

        except ValueError:
            self._logger.error("The number of features or the specified set "
                               "composition is such that at least one set is empty.\n"
                               "Training, validation and test sets will not be extracted.\n"
                               "Please wait for more features to be available or change "
                               "the configuration.")
            return None, None, None, False

    def _save_segregated_sets(self, training_set, validation_set, test_set):
        """Save training, validation and test sets to file.

        Args:
            training_set (DataFrame): The Pandas Dataframe representing the training set.
            validation_set (DataFrame): The Pandas Dataframe representing the validation set.
            test_set (DataFrame): The Pandas Dataframe representing the test set.
        """
        training_set.to_csv(self._training_set_path, index=False)
        validation_set.to_csv(self._validation_set_path, index=False)
        test_set.to_csv(self._test_set_path, index=False)

    def _delete_segregated_features(self):
        """Delete the previously read features that have been segregated."""
        original_columns = self._features.columns.tolist()
        original_columns.insert(0, "timestamp")

        empty_dataframe = pandas.DataFrame(columns=original_columns)
        empty_dataframe.to_csv(self._features_path, index=False)

    def start(self):
        """Start the data segregation.

        Returns:
            bool: True if the segregation succeeds, False if it does not.
                The segregation is successful if the number of available features is enough
                and a confirmation to proceed is given via console.
        """
        self._logger.info("Retrieving available features.")
        self._retrieve_features()

        if self._features.shape[0] < self._configuration.field["minimumNumberOfFeatures"]:
            self._logger.info("The available features are not enough. "
                              "Stopping the segregation.\n"
                              "Available features: %s"
                              "\nRequired number of features: at least %s",
                              str(self._features.shape[0]),
                              str(self._configuration.field["minimumNumberOfFeatures"]))
            return False

        self._logger.info("The available features are enough. Generating the reports about "
                          "the class balancing and the feature space composition.")
        self._generate_class_balancing_report()
        self._generate_feature_space_report("regular")
        self._generate_feature_space_report("anomalous")

        answer = input("DATA SEGREGATION: the reports, based on " + str(self._features.shape[0]) +
                       " features, are available. "
                       "Do you want to proceed with the segregation? [y/n] ")

        if answer.lower() == 'y' or answer.lower() == "yes":
            training_set, validation_set, test_set, success = self._segregate_sets()

            if success is True:
                self._save_segregated_sets(training_set, validation_set, test_set)
                self._delete_segregated_features()
                self._logger.info("New segregated sets are available. "
                                  "Stopping the segregation.")
                return True

        self._logger.info("Stopping the segregation.")
        return False

    def __init__(self):
        """Initializer.

        Raises:
            OSError: If the configuration or schema file cannot be opened.
            JSONDecodeError: If the JSON configuration or schema cannot be decoded correctly.
            ValidationError: If the JSON configuration is invalid.
            SchemaError: If the JSON schema is invalid.
            ValueError: If the sum of training, validation and test set sizes
                in the configuration is not equal to 100%.
        """
        configuration_path = resource_filename(__name__, "configuration/configuration.json")
        schema_path = resource_filename(__name__, "configuration/configuration.schema.json")

        self._configuration = Configuration(configuration_path, schema_path)
        self._features_path = resource_filename(self._configuration.field["preparationPackage"],
                                                self._configuration.field["preparedFeaturesPath"])

        self._class_report_path = resource_filename(
            __name__, self._configuration.field["classBalancingReport"])
        self._feature_space_anomalous_report_path = resource_filename(
            __name__, self._configuration.field["featureSpaceAnomalousReport"])
        self._feature_space_regular_report_path = resource_filename(
            __name__, self._configuration.field["featureSpaceRegularReport"])

        self._training_set_path = resource_filename(
            __name__, self._configuration.field["trainingSetPath"])
        self._validation_set_path = resource_filename(
            __name__, self._configuration.field["validationSetPath"])
        self._test_set_path = resource_filename(
            __name__, self._configuration.field["testSetPath"])

        self._features = None
        self._initialize_logger()

        # Check that the sum of the set sizes in the configuration is equal to 100%.
        if self._configuration.field["trainingSetSize"] + \
                self._configuration.field["validationSetSize"] + \
                self._configuration.field["testSetSize"] != 100:
            raise ValueError("The sum of training, validation and test set sizes "
                             "in the configuration is not equal to 100%.")
