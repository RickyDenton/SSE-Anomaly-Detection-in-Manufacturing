"""Module containing testing methods on the testing_pipelines"""
import json
import math
import random
from test import pipeline_for_testing
from test import diagramgenerator as gen_diagram
import pytest
import progressbar
import pandas
import scipy.stats
from dataingestion import DataIngestionService



@pytest.mark.parametrize(
    "input_size, data_preparation_prob, candidate_selection_prob, iteration, accuracy, seed",
    [[(100, 1), 0.3, 0.4, 30, 0.95, 1]])
def test_non_responsiveness(monkeypatch, input_size, data_preparation_prob,
                            candidate_selection_prob, iteration, accuracy, seed):
    """Function linked to the testing_pipeline to perform
    the non_responsiveness diagrams which describe the application

    Args:
        monkeypatch: links the pipeline module to enable auto input

    Returns:
        Generates the diagrams for the non_responsiveness of the system
        Throw a false assert in case of errors

    """
    static_threshold = 500
    dynamic_threshold = 10
    random.seed(seed)
    static_time = []
    data_ingestion_service = DataIngestionService()

    bartab = progressbar.ProgressBar(max_value=iteration,
                                     widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                              progressbar.Percentage(), '\n'])
    bartab.start()
    for i in range(iteration):
        responses = []

        while random.uniform(0, 1) < data_preparation_prob:
            responses.append("n")
        responses.append("y")

        while random.uniform(0, 1) < candidate_selection_prob:
            responses.append("y")
        responses.append("n")

        monkeypatch.setattr("builtins.input", lambda _: next(responses))

        time_results = pipeline_for_testing.static_pipeline(monkeypatch, input_size[0],
                                                            data_ingestion_service)
        if len(time_results) == 0:
            assert False
        static_time.append(time_results)
        bartab.update(i + 1)
    bartab.finish()

    gen_diagram.make_non_responsiveness_diagram(
        static_time,
        ["Data\nIngestion", "Data\nPreparation", "Data\nSegregation",
         "Training and \nCandidate Selection"],
        static_threshold,
        accuracy,
        "Static Non Responsiveness Results",
        "non_responsiveness_static.png"
    )

    dynamic_time = []
    bartab = progressbar.ProgressBar(max_value=iteration,
                                     widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                              progressbar.Percentage(), '\n'])
    bartab.start()
    for i in range(iteration):
        responses = iter(["y", "n"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        time_results = pipeline_for_testing.dynamic_pipeline(monkeypatch, input_size[1],
                                                             data_ingestion_service)
        if len(time_results) == 0:
            assert False
        dynamic_time.append(time_results)
        bartab.update(i + 1)
    bartab.finish()

    gen_diagram.make_non_responsiveness_diagram(
        dynamic_time,
        ["Data\nIngestion", "Data\nPreparation", "Performance\nEvaluation"],
        dynamic_threshold,
        accuracy,
        "Dynamic Non Responsiveness Results",
        "non_responsiveness_dynamic.png"
    )

    assert True


@pytest.mark.parametrize(
    "input_size, iteration, accuracy, seed",
    [[(50, 250, 500, 750, 1000), 30, 0.95, 1]])
def test_elasticity(monkeypatch, input_size, iteration, accuracy, seed):
    """Function linked to testing_pipeline to perform the elasticity test by
    executing the pipeline several times with different input loads and evaluating
    the needed time to complete the tasks

    Args:
        monkeypatch: links the pipeline module to enable auto input

    Returns:
        Generates the diagrams for the elasticity of the system
        Throw a false assert in case of errors

    """

    phase_responses = ['y', 'n']
    random.seed(seed)

    static_times = []
    dynamic_times = []
    data_ingestion_service = DataIngestionService()

    bartab = progressbar.ProgressBar(max_value=iteration * len(input_size),
                                     widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                              progressbar.Percentage(), '\n'])
    bartab.start()
    for size in input_size:
        phase_time = []
        for i in range(iteration):
            responses = iter(phase_responses)
            monkeypatch.setattr('builtins.input', lambda _: next(responses))
            result = pipeline_for_testing.static_pipeline(monkeypatch, size, data_ingestion_service)
            if len(result) == 0:
                assert False
            phase_time.append(result[4])
            bartab.update(i + 1)
        static_times.append(phase_time)
        # TODO Remove it
        with open("static_elasticity_" + str(size), 'w') as save_file:
            json.dump({"results": phase_time}, save_file, indent=2)
        bartab.finish()

    data = pandas.DataFrame(static_times)
    static_times = [
        data.iloc[i].mean() + (scipy.stats.norm.ppf(accuracy) * data.iloc[i].std()
                               / math.sqrt(iteration))
        for i in range(data.shape[0])
    ]

    gen_diagram.make_elasticity_diagram(
        input_size,
        static_times,
        accuracy,
        iteration,
        "Static Elasticity Diagram",
        "static_elasticity.png"

    )

    bartab = progressbar.ProgressBar(max_value=iteration * len(input_size),
                                     widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                              progressbar.Percentage(), '\n'])

    for size in input_size:
        phase_time = []
        bartab.start()
        for i in range(iteration):
            responses = iter(phase_responses)
            monkeypatch.setattr('builtins.input', lambda _: next(responses))
            result = pipeline_for_testing.dynamic_pipeline(monkeypatch, size, data_ingestion_service)
            if len(result) == 0:
                assert False
            phase_time.append(result[3])
            bartab.update(i + 1)
        bartab.finish()
        dynamic_times.append(phase_time)
        # TODO Remove it
        with open("dynamic_elasticity_" + str(size), 'w') as save_file:
            json.dump({"results": phase_time}, save_file, indent=2)

    data = pandas.DataFrame(dynamic_times)
    dynamic_times = [
        data.iloc[i].mean() + (scipy.stats.norm.ppf(accuracy) * data.iloc[i].std()
                               / math.sqrt(iteration))
        for i in range(data.shape[0])
    ]

    gen_diagram.make_elasticity_diagram(
        input_size,
        dynamic_times,
        accuracy,
        iteration,
        "Dynamic Elasticity Diagram",
        "dynamic_elasticity.png"

    )
    assert True
