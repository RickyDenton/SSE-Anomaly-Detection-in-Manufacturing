"""Module containing the EvaluationService class
implementing the data Evaluation step of the pipeline."""
import logging
import os
import pickle
import sys
import matplotlib
import numpy
import pandas
import seaborn
import coloredlogs
from pkg_resources import resource_filename
from pandas import read_csv
from matplotlib import pyplot
import matplotlib.dates as mdates
from configuration import Configuration


class AnonalyDetectionAndPerformanceService:
    """Class representing the deploy and performance monitor step of the pipeline.

            Attributes:
                _configuration (Configuration): The configuration of the service.
                _features_data_path (str): The absolute path of the file containing
                    the prepared features.
                _model_data_path (str): The absolute path of the file containing the model.
                _model (str): The model.
                _features (DataFrame): The Numpy Array representing the features
                    read in the current step.
                _anomalies_detected_count_data_path: The absolute path of the file containing
                    the count of detected anomalies that it must be write
                _anomalies_for_weeks_data_path:The absolute path of the graph containing the
                    count of detected anomalies for weeks that it must be write
                _detected_anomalies_data_path The absolute path of the file containing all
                    the anomalies detected that it must be write
                _logger (Logger): The logger used in the module.
                _input_path:input file to performance monitoring
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

    def _clean_features(self):
        """Delete the previously read features.
            Returns
                True if it have problems
        """
        try:
            empty_dataframe = pandas.DataFrame(columns=self._features.columns)
            empty_dataframe.to_csv(self._features_data_path, index=False)
        except OSError:
            self._logger.error("EVALUATION SERVICE : Error to get features")
            return True
        return False

    def _save_anomalies(self, anomalies):
        """Generate and save a file csv of anomalies detected.
                    Returns
                            True if it have problems
                """

        anomalies_for_csv = pandas.DataFrame(
            {'Time': self._data_sets['TIMESTAMP'], 'Value': anomalies})
        try:
            if not os.path.exists(self._detected_anomalies_data_path):
                empty_dataframe = pandas.DataFrame(columns=anomalies_for_csv.columns)
                empty_dataframe.to_csv(self._detected_anomalies_data_path, index=False)
            anomalies_for_csv.to_csv(self._detected_anomalies_data_path, mode='a', index=False,
                                     header=False)
        except OSError:
            self._logger.error(
                "EVALUATION SERVICE : Error to write the detected anomalies on: %s",
                self._detected_anomalies_data_path)
            return True
        return False

    def _generate_report(self, anomalies):
        """Generate and save a report about the anomalies detected.
            Returns
                    True if it have problems
        """
        anomalous_count = numpy.count_nonzero(anomalies)
        regular_count = len(anomalies) - anomalous_count
        if regular_count < 0:
            self._logger.error("EVALUATION SERVICE : Regular count is <0")
            return True

        report_data = pandas.DataFrame([{"Type of shift": "Anomalous",
                                         "Number of samples": anomalous_count},
                                        {"Type of shift": "Regular",
                                         "Number of samples": regular_count}])

        matplotlib.pyplot.figure(figsize=(19.20, 10.80))
        barplot = seaborn.barplot(x="Type of shift", y="Number of samples", data=report_data,
                                  palette=["firebrick", "cornflowerblue"])
        barplot.legend(loc="upper left", prop={'size': 14})

        try:
            matplotlib.pyplot.savefig(self._anomalies_detected_count_data_path, format="png",
                                      dpi=300,
                                      bbox_inches='tight')
        except OSError:
            self._logger.error(
                "EVALUATION SERVICE : Error to save graph on: %s",
                self._anomalies_detected_count_data_path)
            return True
        return False

    def _show_anomalies_for_weeks(self):
        """ Genrate a graph with X=Weeks,Y=N° anomalies
                STEPS:
                    Get the detectedAnomalies.csv
                    Plot the graph
                Returns
                    True if it have problems
        """
        try:
            series = read_csv(self._detected_anomalies_data_path)
        except OSError:
            self._logger.error("EVALUATION SERVICE : csv file: %s", self._configuration.field[
                "detectedAnomaliesDataPath"] + " don't found")
            return True

        data = series.set_index('Time')
        data.index = pandas.to_datetime(data.index)
        anomalies_for_weeks = data.resample(rule='W').sum()

        fig = pyplot.figure()
        ax_fig = fig.add_subplot(111)
        ax_fig.plot(anomalies_for_weeks)

        ax_fig.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

        pyplot.title('N° of Anomalies for weeks')
        pyplot.ylabel('N° of Anomalies')
        pyplot.xlabel('Weeks')
        try:
            pyplot.savefig(self._anomalies_for_weeks_data_path, dpi=300, bbox_inches='tight')
        except OSError:
            self._logger.error(
                "EVALUATION SERVICE : Error to write the file on: %s",
                self._anomalies_for_weeks_data_path)

        return False

    def _retrieve_features(self):
        """Retrieve the prepared features, if available, and save them in self._ingested_data.
            Returns
                    True if it have problems
        """
        try:
            # Overall features as a Pandas dataframe.
            self._features = pandas.read_csv(self._features_data_path)

            self._data_sets = {
                "X": numpy.array(
                    (self._features.loc[:, self._features.columns != 'ANOMALOUS']).drop('timestamp',
                                                                                        axis=1)),
                "Y": numpy.array(self._features['ANOMALOUS']),
                "TIMESTAMP": numpy.array(self._features['timestamp'])
            }

        except OSError:
            # File not found or file cannot be opened.
            self._logger.error("EVALUATION SERVICE : Feature don't found or not valid")
            return True
        return False

    def _retrieve_model(self):
        """Retrieve the prepared model, if available, and save them in self._model.
            Returns
                    True if it have problems
        """
        try:
            # Overall features as a Pandas dataframe.
            self._model = pickle.load(open(self._model_data_path, 'rb'))
        except OSError:
            # File not found or file cannot be opened.
            self._logger.error("EVALUATION SERVICE : Model don't found")
            return True

        return False

    def _detect_anomalies(self):
        """Valuate anomalies in the system.
                                Returns:
                                    vector of 0,1 (0 = ok, 1 = anomaly)
                                    Or False if it has an error
        """
        try:
            anomalies = self._model.predict(self._data_sets['X'])
        except OSError:
            self._logger.error(
                "EVALUATION SERVICE : Error to get anomalies by the model and data sets")
            return None
        if 1 in anomalies:
            print("ANOMALY DETECTED")
        return anomalies

    def _performance_monitoring(self):
        """Valuate performace.
                        Returns:
                            float: Percentage of accuracy
                            Or False if it has an error
        """
        real_data = None
        csv_file = None

        req = input("EVALUATION SERVICE : Do you want evaluate performance monitoring? [y/n] ")
        if req.lower() == 'n' or req.lower() == "no":
            return 0

        # Create dummy file to demonstration.
        # The file is a sequence of 0,1 with column ANOMALIES
        # This file must be create by the technical expert that control the system
        # --------
        count = self._data_sets['Y'].shape[0]
        dummy_array = numpy.random.randint(2, size=count)
        pandas.DataFrame({'ANOMALIES': dummy_array}).to_csv(
            os.path.join(self._input_path, 'realDataDummy.csv'),
            index=False)
        # ----------

        # Get file created by technical expert
        try:
            csv_file = [file for file in os.listdir(self._input_path) if file.endswith('.csv')]
            real_data = numpy.array(pandas.read_csv(os.path.join(self._input_path, csv_file[0])))

            score = self._model.score(self._data_sets['X'], real_data)

        except OSError:
            logging.error("real anomalies file don't found or not valid")
            return -1

        if score < 0 or score > 1:
            self._logger.error("EVALUATION SERVICE : Score cannot be < 0 or > 100")
            return -1
        return score

    def start(self):
        """Start the data segregation.
                Returns:
                    bool: True if the anomalydetectionandperformance succeeds, False if it does not.
        """

        self._logger.info("EVALUATION SERVICE : Retrieving available model.")
        if self._retrieve_model():
            self._logger.error("Retrieve Model Error")
            return False

        self._logger.info("EVALUATION SERVICE : Retrieving available features.")
        if self._retrieve_features():
            self._logger.error("EVALUATION SERVICE : Retrieve Features Error")
            return False

        self._logger.info("EVALUATION SERVICE : Valuate anomalies")
        anomalies = self._detect_anomalies()
        if anomalies is None:
            self._logger.error("Valuate anomalies Error")
            return False

        self._logger.info("EVALUATION SERVICE : Save anomalies on csv detectedAnomalies.csv")
        if self._save_anomalies(anomalies):
            self._logger.error("Save anomalies Error")
            return False

        self._logger.info("EVALUATION SERVICE : Generate report anomaliesDetectedCount.png")
        if self._generate_report(anomalies):
            self._logger.error("Generate Report Error")
            return False

        self._logger.info("EVALUATION SERVICE : Generate report anomalies for weeks")
        if self._show_anomalies_for_weeks():
            self._logger.error('Show anomalies error')
            return False

        self._logger.info("EVALUATION SERVICE : Valuate model accuracy")
        performance_evaluation_score = self._performance_monitoring()
        if performance_evaluation_score == 0:
            self._logger.info(
                "EVALUATION SERVICE : Performance anomalydetectionandperformance not done")
        elif performance_evaluation_score == -1:
            self._logger.error(
                "EVALUATION SERVICE : Performance anomalydetectionandperformance Error")
            return False
        else:
            self._logger.info("EVALUATION SERVICE : score model is %s",
                              str(performance_evaluation_score))
            if performance_evaluation_score < self._configuration.field["recalibrateModelScore"]:
                self._logger.info(
                    "EVALUATION SERVICE : Insufficient accuracy level,"
                    " model is not optimal, it's better to ri-calibrate")
                req = input("Do you want recalibrate the model? [Y/N]")
                if req.lower() == 'y' or req.lower() == "yes":
                    return True

        if self._clean_features():
            self._logger.error("EVALUATION SERVICE : Error to clean features")
            return False
        return None

    def __init__(self):
        """Initializer.

                Raises:
                    OSError: If the configuration or schema file cannot be opened.
                    JSONDecodeError: If the JSON configuration or schema cannot be decoded
                        correctly.
                    ValidationError: If the JSON configuration is invalid.
                    SchemaError: If the JSON schema is invalid.
                    NotImplementedError: If a scaling method different than standard or min-max
                        is allowed by configuration, but not implemented yet.
        """
        # Create a logger object.
        self._initialize_logger()

        configuration_path = resource_filename(__name__, "configuration/configuration.json")
        schema_path = resource_filename(__name__, "configuration/configuration.schema.json")

        self._configuration = Configuration(configuration_path, schema_path)
        self._features_data_path = resource_filename(
            self._configuration.field["dataPreparationPackage"],
            self._configuration.field["featureDataPath"])
        self._model_data_path = resource_filename(
            self._configuration.field["candidateModelPackage"],
            self._configuration.field["candidateModelDataPath"])
        self._anomalies_detected_count_data_path = \
            resource_filename(__name__, self._configuration.field["anomaliesDetectedCountDataPath"])
        self._anomalies_for_weeks_data_path = resource_filename(__name__,
                                                                self._configuration.field[
                                                                    "anomaliesForWeekDataPath"])
        self._detected_anomalies_data_path = resource_filename(__name__,
                                                               self._configuration.field[
                                                                   "detectedAnomaliesDataPath"])
        self._input_path = resource_filename(__name__, self._configuration.field["inputPath"])

        self._features = None
        self._model = None
        self._data_sets = None
