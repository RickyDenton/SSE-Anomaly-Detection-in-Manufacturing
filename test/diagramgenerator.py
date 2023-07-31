"""The module contains the diagrams that will be created during the testing phase
    - non-responsiveness diagram
    - elasticity diagram
"""
import json
import math
import pandas
import scipy.stats
import sklearn.metrics
from matplotlib import pyplot as plt
import matplotlib.patches as mpl_patch


def make_non_responsiveness_diagram(data, labels, threshold, accuracy, title, position):
    """Function to create the non_responsiveness diagrams

    Args:
        data: data to be plotted formatted to be a list of lists [[],[],[]]
        labels: list of labels to identify the phases results
        threshold: threshold value of the module responsiveness
        accuracy: accuracy used to analyze data
        title: title to be plotted on the diagram
        position: name of the file where the diagram will be saved

    Returns:
        Generates the diagrams for the responsiveness of the static and dynamic system
        Throw a false assert in case of errors

    """
    data = pandas.DataFrame(data)
    data_mean = [
        [
            round(data[i].mean(), 2) for i in range(data.shape[1])
        ], [
            round(scipy.stats.norm.ppf(accuracy) * data[i].std() / math.sqrt(data[i].shape[0]), 2)
            for i in range(data.shape[1])]
    ]

    # TODO Remove it
    with open(position[:-4] + "_results.txt", 'w') as save_file:
        json.dump({"results": data_mean}, save_file, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=[18, 8])
    fig.suptitle(title, fontsize=20, weight="bold")
    global_info = mpl_patch.Patch(
        label='Accuracy: ' + str(accuracy) + "\nIterations: " + str(data[0].shape[0]),
        color="orange")
    fig.legend(loc='upper left', frameon=False, handles=[global_info], fontsize=14)

    ax1.tick_params(axis="x", labelsize=14)
    ax1.tick_params(axis="y", labelsize=14)
    ax1.barh(labels,
             data_mean[0][0:data.shape[1] - 1],
             xerr=data_mean[1][0:data.shape[1] - 1], linewidth=3,
             capsize=14, align='center',
             color=["springgreen", "mediumseagreen", "seagreen", "green"], edgecolor='darkgreen')

    ax1.set_xlim([0, max(data_mean[0][0:data.shape[1]-1]) * 1.4])
    rects = ax1.patches
    labels = data_mean[0]
    for rect, label in zip(rects, labels):
        ax1.text(rect.get_x() + rect.get_width() + max(data_mean[0][0:data.shape[1]-1])*0.1,
                 rect.get_y() + rect.get_height() * 0.4, label,
                 ha='center', va='bottom', weight="bold", fontsize=18, color="darkgreen")

    ax2.tick_params(axis="x", labelsize=14)
    ax2.tick_params(axis="y", labelsize=14)
    ax2.set_xticklabels(["Total"], fontsize=14)

    ax2.bar(["Total"], [data_mean[0][data.shape[1] - 1]], width=0.5,
            color='orangered', edgecolor='darkred', linewidth=3,
            yerr=data_mean[1][data.shape[1] - 1], capsize=14, label='poacee')
    ax2.axhline(threshold, 0, 1)

    rects = ax2.patches
    labels = [data_mean[0][data.shape[1] - 1]]
    for rect, label in zip(rects, labels):
        ax2.text(rect.get_x() + rect.get_width() / 2, rect.get_height() / 3, label,
                 ha='center', va='bottom', weight="bold", fontsize=18, color="darkred")

    static_threshold_value = mpl_patch.Patch(label='Threshold: ' + str(threshold))
    ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=1, mode="expand", borderaxespad=0.,
               handles=[static_threshold_value], fontsize=14)

    plt.savefig(position, bbox_inches="tight")
    plt.clf()


def make_elasticity_diagram(input_sizes, exec_times, accuracy, iteration, title, position):
    """The function identifies the elasticity values of the static and dynamic
    components of our pipeline. It also generates a diagram to report the
    trend found and the elasticity value identified

    Args:
        exec_times:  recorder times for execute the elasticity test
        input_sizes: input sizes given into the elasticity test
        exec_times: times recordered during the elasticity test
        accuracy: accuracy used during the result analysis
        iteration: number of iteration used during the test
        title: title to give to the diagram
        position: name of the file for save it

    Returns:
        The test creates a elasticity diagram

    """
    mean_square_sel = None
    values = [0, 0.5] + list(range(1, 20))

    elasticity_value = 0
    for i in values:
        exp_values = [exec_times[0] * math.pow(input_sizes[a] / input_sizes[0], i)
                      for a in range(1, len(input_sizes))]

        mean_square_error = sklearn.metrics.mean_squared_error(exec_times[1:], exp_values)

        if mean_square_sel is not None and mean_square_sel < mean_square_error:
            break

        mean_square_sel = mean_square_error
        elasticity_value = i

    plt.figure(figsize=(12, 8))
    plt.title(title, pad=50, fontsize=20, weight="bold")

    el_value = mpl_patch.Patch(label='Elasticity Value: ' + str(elasticity_value), color="crimson")
    el_acc = mpl_patch.Patch(label='Accuracy: ' + str(accuracy), color="darkorange")
    el_iter = mpl_patch.Patch(label='Iterations: ' + str(iteration), color="darkslateblue")

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=3, mode="expand", borderaxespad=0.,
               handles=[el_value, el_acc, el_iter], fontsize=14)

    plt.xlim(min(input_sizes) - (max(input_sizes) - min(input_sizes)) * .4,
             max(input_sizes) + (max(input_sizes) - min(input_sizes)) * .4)
    plt.xlabel("number of samples", fontsize=14)

    plt.ylim(min(exec_times) - (max(exec_times) - min(exec_times)) * .4,
             max(exec_times) + (max(exec_times) - min(exec_times)) * .4)
    plt.ylabel("time(s)", fontsize=14)

    plt.plot(input_sizes, exec_times, ".--", markersize=30, color="deepskyblue",
             markeredgecolor="cornflowerblue", markerfacecolor="cornflowerblue")

    plt.savefig(position, bbox_inches="tight")
    plt.clf()
