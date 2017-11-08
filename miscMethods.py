import numpy as np
import random
import parameters
import matplotlib.pyplot as plt
import time

random.seed(parameters.seed)


def find_common_keys(classifier, row):  # find the common keys of two dictionaries
    return set(classifier.keys()) & set(row.keys())


def find_different_keys(dict1, dict2):
    return set(dict1.keys())-set(dict2.keys())


def subset_dictionary(dictionary, intersection):  # extract subset of key-val pairs if in
    return dict((value, dictionary[value]) for value in intersection)


def dict_to_numpy_array(dictionary):
    return np.array(list(dictionary.values()))


def dict_to_unit_numpy_matrix(dictionary):
    matrix = np.identity(len(dictionary))
    values = list(dictionary.values())
    for i in range(0, len(dictionary)):
        matrix[i][i] = values[i]
    return matrix


def tr(matrix):
    return np.transpose(matrix)


def dot(value_list):
    return_value = value_list[0]
    for i in range(1, len(value_list)):
        return_value = np.dot(return_value, value_list[i])
    return return_value


def numpy_array_to_dict(numpy_array, labels):
    return dict((labels[i], numpy_array[i]) for i in range(0, len(labels)))


def numpy_matrix_to_dict(numpy_matrix, labels):
    return dict((labels[i], numpy_matrix[i][i]) for i in range(0, len(labels)))


def generate_training_test(dataset, training_test_ratio):
        random.shuffle(dataset)
        training_length = int(len(dataset)*training_test_ratio)
        test_dataset = dataset[:training_length]
        training_dataset = dataset[training_length:]
        return training_dataset, test_dataset


def generate_training_test_trapezoidal(dataset, heldout):  # dont shuffle for the order
        training_test_ratio = 1-heldout
        training_length = int(len(dataset)*training_test_ratio)
        test_dataset = dataset[training_length:]
        training_dataset = dataset[:training_length]
        return training_dataset, test_dataset


# Plotting Methods
def plot_error(error_vector, dataset_name):
    # xx = 1. - np.array(parameters.heldout) #heldouts
    xx = list(range(len(error_vector)-1))
    yy = (np.array(error_vector[1:])*100)
    plt.plot(xx, yy, label=dataset_name+", Remaining features: "+str(parameters.removing_percentage))
    plt.legend(loc="upper right")
    plt.xlabel("Proportion Train")
    plt.ylabel("Test Error Rate %")
    plt.grid()
    plt.savefig('./figures/'+dataset_name+'_'+str(1-parameters.removing_percentage)+'_'+time.strftime("%Y%m%d-%H%M%S")+'.png')
    plt.show()
    plt.clf()


def plot_features(feature_summary, dataset_name):
    xx = np.array(range(0, len(feature_summary)))
    yy = np.array(feature_summary)
    plt.plot(xx, yy, label=dataset_name, marker='o', linestyle='--')
    plt.legend(loc="upper right")
    plt.xlabel("Training examples")
    plt.ylabel("Number of features")
    plt.grid()
    plt.savefig('./figures/'+dataset_name+time.strftime("%Y%m%d-%H%M%S")+'features.png')
    plt.show()
    plt.clf()


def plot_classifier_dimension(classifier_summary, dataset_name):
    xx = np.array(range(0, len(classifier_summary)))
    yy = np.array(classifier_summary)
    plt.plot(xx, yy, label=dataset_name+", B: "+str(parameters.sparsity_B))
    plt.legend(loc="upper right")
    plt.xlabel("Classifiers")
    plt.ylabel("Dimension")
    plt.grid()
    plt.savefig('./figures/'+dataset_name+'_'+str(1-parameters.removing_percentage)+'_'+time.strftime("%Y%m%d-%H%M%S")+'sparsity.png')
    plt.show()
    plt.clf()
