"""Module containing the starting point of the pipeline."""
import atexit
import argparse
import coloredlogs
from dataingestion import DataIngestionService
from datapreparation import DataPreparationService
from datasegregation import DataSegregationService
from candidatemodelevaluation import CandidateModelEvaluation
from anomalydetectionandperformance import AnonalyDetectionAndPerformanceService

coloredlogs.DEFAULT_LEVEL_STYLES = {"info": {"color": 250}, "error": {"color": 203}}
coloredlogs.DEFAULT_FIELD_STYLES = {"asctime": {"color": "green"},
                                    "levelname": {"color": "white"},
                                    "name": {"color": 25}}


def parse_arguments():
    """Parse the command line arguments.

    Returns:
        Namespace: the object containing the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Pipeline for anomaly detection in manufacturing")

    parser.add_argument("--m", "--mode", action="store", dest="pipeline_mode",
                        type=str, choices=["training", "classification"],
                        help="Execution mode of the pipeline")

    return parser.parse_args()


def cleanup(data_ingestion):
    """Close the data ingestion service when the execution of the pipeline is stopped.

    Args:
        data_ingestion (DataIngestionService): the object representing the data ingestion service.
    """
    print("Closing the data ingestion module.")
    data_ingestion.close()


def main():
    """Start the pipeline either in training or classification mode."""
    arguments = parse_arguments()

    if arguments.pipeline_mode is None:
        print("Please specify the execution mode of the pipeline.")
        return

    data_ingestion = DataIngestionService()
    data_preparation = DataPreparationService()
    data_segregation = DataSegregationService()
    candidate_model_evaluation = CandidateModelEvaluation()
    evaluation_anomalies = AnonalyDetectionAndPerformanceService()

    atexit.register(cleanup, data_ingestion)

    if arguments.pipeline_mode == "training":
        while True:
            if not data_ingestion.start_ingest_labeled_series():
                continue

            if not data_preparation.start(training_phase=True):
                continue

            if not data_segregation.start():
                continue

            if not candidate_model_evaluation.start():
                continue

            print("Training completed.")
            return

    if arguments.pipeline_mode == "classification":
        while True:
            if not data_ingestion.start_ingest_unlabeled_series():
                continue

            if not data_preparation.start(training_phase=False):
                continue

            recalibration = evaluation_anomalies.start()

            if recalibration:
                print("Recalibration requested.")
                return


if __name__ == "__main__":
    main()
