"""Module containing the DataPreparationService class
implementing the data preparation step of the pipeline."""
import os
import sys
import json
import logging
import numpy
import pandas
import coloredlogs
from pkg_resources import resource_filename
from configuration import Configuration
from .featureextractor import FeatureExtractor
from .onlinescaler import OnlineMinMaxScaler


class DataPreparationService:
    """Class representing the data preparation step of the pipeline.

    Attributes:
        _configuration (Configuration): The configuration of the service.
        _labeled_ingested_data_path (str): The absolute path of the file containing
            the ingested data to be used for training (valid classification column).
        _unlabeled_ingested_data_path (str): The absolute path of the file containing
            the ingested data to be classified (invalid classification column).
        _extracted_features_path (str): The absolute path of the file that will contain
            the extracted features.
        _cursor_path (str): The absolute path of the file containing the cursors used
            to read the ingested data.
        _ingested_data (DataFrame): The Pandas dataframe representing the ingested data
            read in the current step.
        _cursor_labeled (int): The actual value of the cursor used to read the classified data
            for training.
        _cursor_unlabeled (int): The actual value of the cursor used to read the data
            to be classified.
        _online_scaler (OnlineMinMaxScaler): The scaler used to scale the extracted features.
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

    def _retrieve_cursors(self):
        """Retrieve the previously saved cursors from file and save them
        in the relative attributes. If the file cannot be opened or does not exist,
        the cursors are set so that they point to the beginning of the file to read.
        """
        try:
            with open(self._cursor_path, 'r') as cursor_file:
                cursors = json.load(cursor_file)
                self._cursor_labeled = cursors["cursorLabeled"]
                self._cursor_unlabeled = cursors["cursorUnlabeled"]

                if self._cursor_labeled < 0:
                    self._cursor_labeled = 0

                if self._cursor_unlabeled < 0:
                    self._cursor_unlabeled = 0

        except (OSError, json.JSONDecodeError):
            # File not found, file cannot be opened correctly or JSON decode error.
            self._logger.error("Impossible to retrieve the saved cursors. "
                               "Data will be read from the beginning of the file.")
            self._cursor_labeled = 0
            self._cursor_unlabeled = 0

    def _save_cursor(self):
        """Save the current value of the cursors to file."""
        try:
            with open(self._cursor_path, 'w') as cursor_file:
                cursors = {"cursorLabeled": self._cursor_labeled,
                           "cursorUnlabeled": self._cursor_unlabeled}

                json.dump(cursors, cursor_file, indent=2)

        except OSError:
            # Cursors cannot be saved.
            self._logger.error("Impossible to save the cursors.")

    def _retrieve_ingested_data(self, training_phase):
        """Retrieve new ingested data, if available, and save them in self._ingested_data.

        Args:
            training_phase (bool): true if the preparation must be done to support
                the training cycle of the pipeline, false if it must be done
                to support the classification cycle.
        """
        try:
            # Overall ingested data as a Pandas dataframe. Skip the previously read data.
            if training_phase:
                self._ingested_data = pandas.read_csv(self._labeled_ingested_data_path,
                                                      skiprows=range(1, self._cursor_labeled + 1))
                self._cursor_labeled += self._ingested_data.shape[0]
            else:
                self._ingested_data = pandas.read_csv(self._unlabeled_ingested_data_path,
                                                      skiprows=range(1, self._cursor_unlabeled + 1))
                self._cursor_unlabeled += self._ingested_data.shape[0]

        except OSError:
            # File not found or file cannot be opened.
            self._ingested_data = pandas.DataFrame()

    def _save_features(self, features, labels):
        """Save the extracted features to file, eventually with append.

        Args:
            features (ndarray): Numpy array containing the features to save.
            labels (list): Labels denoting the features.
        """
        feature_dataframe = pandas.DataFrame(data=features, columns=labels)

        # If file does not exist, write data and header.
        if not os.path.isfile(self._extracted_features_path):
            feature_dataframe.to_csv(self._extracted_features_path, index=False)
        else:  # If file exists, append data without header.
            feature_dataframe.to_csv(self._extracted_features_path, mode='a',
                                     index=False, header=False)

    def start(self, training_phase):
        """Start the data preparation.

        Args:
            training_phase (bool): true if the preparation must be done to support
                the training cycle of the pipeline, false if it must be done
                to support the classification cycle.

        Returns:
            bool: True if the preparation succeeds, False if it does not.
                The preparation is successful if new ingested data are available
                and they are correctly prepared.
        """
        self._logger.info("Retrieving new ingested data.")
        self._retrieve_ingested_data(training_phase)

        if self._ingested_data.shape[0] == 0:
            self._logger.info("New ingested data are not available. "
                              "Stopping the preparation.")
            return False

        self._logger.info("New ingested data are available. Starting the preparation.")

        timestamp = self._ingested_data["timestamp"].to_numpy().reshape(-1, 1)
        classification = self._ingested_data["ANOMALOUS"].to_numpy().reshape(-1, 1)
        time_series = self._ingested_data.loc[:,
                                              (self._ingested_data.columns != "ANOMALOUS") &
                                              (self._ingested_data.columns != "timestamp")]\
            .to_numpy(copy=True)

        features, labels = FeatureExtractor(time_series).extract()
        scaled_features = self._online_scaler.fit_transform(features)

        self._save_features(numpy.concatenate((timestamp, classification, scaled_features), axis=1),
                            labels)
        self._save_cursor()

        self._logger.info("New features are available. Stopping the preparation.")
        return True

    def __init__(self):
        """Initializer.

        Raises:
            OSError: If the configuration or schema file cannot be opened.
            JSONDecodeError: If the JSON configuration or schema cannot be decoded correctly.
            ValidationError: If the JSON configuration is invalid.
            SchemaError: If the JSON schema is invalid.
        """
        configuration_path = resource_filename(__name__, "configuration/configuration.json")
        schema_path = resource_filename(__name__, "configuration/configuration.schema.json")

        self._configuration = Configuration(configuration_path, schema_path)

        self._labeled_ingested_data_path = \
            resource_filename(self._configuration.field["ingestionPackage"],
                              self._configuration.field["labeledIngestedDataPath"])

        self._unlabeled_ingested_data_path =  \
            resource_filename(self._configuration.field["ingestionPackage"],
                              self._configuration.field["unlabeledIngestedDataPath"])

        self._extracted_features_path = \
            resource_filename(__name__, self._configuration.field["extractedFeaturesPath"])

        self._cursor_path = resource_filename(__name__, "cursor/cursor.json")

        self._ingested_data = None
        self._cursor_labeled = None
        self._cursor_unlabeled = None
        self._online_scaler = OnlineMinMaxScaler()

        self._initialize_logger()
        self._retrieve_cursors()
