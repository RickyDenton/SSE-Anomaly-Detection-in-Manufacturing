""" module to perform tests on the pipeline. To not change the formal pipeline
the module uses a dummy replica 'testing_pipeline' which is built to easily perform
all the tests without changing the original behavior"""
import json
import os
import time
import coloredlogs
from pkg_resources import resource_filename
from dataingestion import DataIngestionTester
from datapreparation import DataPreparationService
from datasegregation import DataSegregationService
from candidatemodelevaluation import CandidateModelEvaluation
from anomalydetectionandperformance import AnonalyDetectionAndPerformanceService

coloredlogs.DEFAULT_LEVEL_STYLES = {"info": {"color": 250}, "error": {"color": 203}}
coloredlogs.DEFAULT_FIELD_STYLES = {"asctime": {"color": "green"},
                                    "levelname": {"color": "white"},
                                    "name": {"color": 25}}


def reset_cursor():
    """data preparation doesn't not have the ability to run sequentially
    it is necessary to work around its behavior by forcibly resetting
    the cursor which defines its execution[TO BE REMOVED]
    """
    cursor_path = resource_filename("datapreparation", "cursor/cursor.json")
    with open(cursor_path, 'w') as cursor_file:
        json.dump({"cursorLabeled": 0, "cursorUnlabeled": 0}, cursor_file, indent=2)
    return True


def reset_temp_files():
    """clean features.csv, unlabeledSeries.csv and labeledSeries.csv
    """
    delete_file_ingestion_labeled = resource_filename("dataingestion",
                                                      "datastore/labeledSeries/labeledSeries.csv")
    delete_file_preparation = resource_filename("datapreparation", "output/features.csv")
    delete_file_ingestion_unlabeled = resource_filename("dataingestion",
                                                        "datastore/unlabeledSeries/"
                                                        "unlabeledSeries.csv")
    try:
        os.remove(delete_file_preparation)
    except OSError:
        pass

    try:
        os.remove(delete_file_ingestion_labeled)
    except OSError:
        pass

    try:
        os.remove(delete_file_ingestion_unlabeled)
    except OSError:
        pass


def static_pipeline(_, size, data_ingestion_service) -> list:
    """Testing execution of the static pipeline.

    Args:
        data_ingestion_service: Data Ingestion Service pre-loaded
        size: number of time series to be ingested
        _: propagate the inputs to the module from the testing file

    Returns: a list containing the times of each phase and a total time or an empty list
             in case of error
    """

    reset_cursor()
    reset_temp_files()

    data_ingestion_gen_test = DataIngestionTester()
    data_preparation = DataPreparationService()
    data_segregation = DataSegregationService()
    candidate_model_evaluation = CandidateModelEvaluation()

    data_ingestion_total = 0
    data_preparation_total = 0
    data_segregation_total = 0

    while True:
        data_ingestion_gen_test.simulate_labeled_series(size)
        data_ingestion_time = time.time()
        data_ingestion_service.start_ingest_labeled_series(size)
        data_ingestion_time = time.time() - data_ingestion_time
        data_ingestion_total += data_ingestion_time

        data_preparation_time = time.time()
        if data_preparation.start(training_phase=True) is False:
            return []
        data_preparation_time = time.time() - data_preparation_time
        data_preparation_total += data_preparation_time

        data_segregation_time = time.time()
        result = data_segregation.start()
        data_segregation_time = time.time() - data_segregation_time
        data_segregation_total += data_segregation_time
        if result is True:
            break

    model_evaluation_time = time.time()
    if candidate_model_evaluation.start() is False:
        return []
    model_evaluation_time = time.time() - model_evaluation_time

    total_time = data_ingestion_total + data_preparation_total \
        + data_segregation_total + model_evaluation_time

    data_ingestion_gen_test.close()

    return [data_ingestion_total, data_preparation_total, data_segregation_total,
            model_evaluation_time,
            total_time]


def dynamic_pipeline(_, size, data_ingestion_service):
    """Testing execution of the dynamic pipeline.

    Args:
        data_ingestion_service: Data Ingestion Service pre-loaded
        size: number of time series to be ingested
        _: propagate the inputs to the module from the testing file

    Returns: a list containing the times of each phase and a total time or an empty list
             in case of error
    """

    reset_cursor()
    reset_temp_files()

    data_ingestion_gen_test = DataIngestionTester()
    data_preparation = DataPreparationService()
    evaluation_anomalies = AnonalyDetectionAndPerformanceService()

    data_ingestion_gen_test.simulate_unlabeled_series(size)
    data_ingestion_time = time.time()
    data_ingestion_service.start_ingest_unlabeled_series(size)
    data_ingestion_time = time.time() - data_ingestion_time

    data_preparation_time = time.time()
    if data_preparation.start(training_phase=False) is False:
        return []
    data_preparation_time = time.time() - data_preparation_time

    evaluation_anomalies_time = time.time()
    if evaluation_anomalies.start() is False:
        return []
    evaluation_anomalies_time = time.time() - evaluation_anomalies_time

    total_time = data_ingestion_time + data_preparation_time + evaluation_anomalies_time

    data_ingestion_gen_test.close()

    return [data_ingestion_time, data_preparation_time, evaluation_anomalies_time, total_time]
