import olvf
import olsf
import time
import copy
import random
import preprocess2
import miscMethods
import numpy as np
import matplotlib.pyplot as plt


def preprocess_data(data, mode):  # shuffle, remove and fill with 0's
    random.seed(10)
    copydata = copy.deepcopy(data)
    random.shuffle(copydata)
    if mode == 'trapezoidal':
        dataset = preprocess2.remove_features_trapezoidal(copydata)
    elif mode == 'variable':
        dataset = preprocess2.remove_features_random(copydata)
    else:
        dataset = copydata

    all_keys = set().union(*(d.keys() for d in dataset))

    x, y = [], []

    for row in dataset:
        for key in all_keys:
            if key not in row.keys():
                row[key] = 0
        y.append(row['class_label'])
        del row['class_label']
    if 0 not in row.keys():
        start = 1
    if 0 in row.keys():
        start = 0
    for row in dataset:
        X_row = []
        for i in range(start, len(row)):
            X_row.append(row[i])
        x.append(X_row)
    return x, y


def experiment_stream(data, dataset_name):
    plot_list = [("OLvf", olvf.OLVF("stream").fit, 1), ("OLsf", olsf.OLSF("stream").fit, 1)]
    plot_list2 = []

    x = list(range(len(data)))
    X, y = preprocess_data(data)  # data is being shuffled inside

    for triple in plot_list:
        if triple[2] == 1:
            plot_list2.append((triple[1](X, y), triple[0]))

    for i in range(len(plot_list2)):  # plot 10 by 10
        plt.plot(x[1:][0::int(len(y)/10)], plot_list2[i][0][1:][0::int(len(y)/10)], label=plot_list2[i][1], marker='o', linestyle='--')

    plt.legend()
    plt.xlabel("Instance")
    plt.ylabel("Test Error Rate %")
    plt.title(dataset_name)
    plt.savefig('./figures/'+'stream'+str("_")+time.strftime("%H%M%S")+'.png')
    plt.grid()
    plt.show()

    feature_summary = [np.count_nonzero(row) for row in X]
    miscMethods.plotFeatures(feature_summary, "")


def experiment_uci_stream():
    # olsf_stream(preprocess2.readUrlNormalized(1), "URL")
    # olsf_stream(preprocess2.readWbcNormalized(), "WBC")
    experiment_stream(preprocess2.read_german(), "German")
    # olsf_stream(preprocess2.readIonosphereNormalized(), "Ionosphere")
    # olsf_stream(preprocess2.readMagicNormalized(), "Magic")
    # olsf_stream(preprocess2.readSpambaseNormalized(), "Spambase")
    # olsf_stream(preprocess2.readWdbcNormalized(), "WDBC")
    # olsf_stream(preprocess2.readWpbcNormalized(), "WPBC")
    # olsf_stream(preprocess2.readA8ANormalized(), "A8A")
    #olsf_stream(preprocess2.read_svmguide3(), "svmguide3")


experiment_uci_stream()
