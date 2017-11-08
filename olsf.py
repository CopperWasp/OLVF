import numpy as np
import preprocess2
import random
import copy


# trapezoidal experiment needs shufflable method


class OLSF:
    def __init__(self, mode):
        self.weights = []
        self.X = []
        self.y = []
        self.C = 0.1
        self.Lambda = 30
        self.B = 0.64
        self.rounds = 1
        self.option = 2
        self.mode = mode
        if mode == 'stream':
            self.rounds = 1

    def initialize(self, data, labels):
        self.X = data
        self.y = labels

    def set_classifier(self):
        self.weights = np.zeros(np.count_nonzero(self.X[0]))

    def parameter_set(self, i, loss):
        if self.option == 0:
            return loss / np.dot(self.X[i], self.X[i])
        if self.option == 1:
            return np.minimum(self.C, loss / np.dot(self.X[i], self.X[i]))
        if self.option == 2:
            return loss / ((1 / (2 * self.C)) + np.dot(self.X[i], self.X[i]))

    def sparsity_step(self):
        projected = np.multiply(np.minimum(1, self.Lambda / np.linalg.norm(self.weights, ord=1)), self.weights)
        self.weights = self.truncate(projected)

    def truncate(self, projected):
        if np.linalg.norm(projected, ord=0) > self.B * len(projected):
            remaining = int(np.maximum(1, np.floor(self.B * len(projected))))
            for i in projected.argsort()[:(len(projected) - remaining)]:
                projected[i] = 0
            return projected
        else:
            return projected

    def fit(self, data, labels):

        self.initialize(data, labels)

        for i in range(0, self.rounds):
            train_error = 0
            train_error_vector = []
            # total_error_vector = np.zeros(len(self.y))

            self.set_classifier()

            for i in range(0, len(self.y)):
                row = self.X[i][:np.count_nonzero(self.X[i])]
                if len(row) == 0:
                    continue
                y_hat = np.sign(np.dot(self.weights, row[:len(self.weights)]))
                if y_hat != self.y[i]:
                    train_error += 1
                loss = (np.maximum(0, 1 - self.y[i] * (np.dot(self.weights, row[:len(self.weights)]))))
                tao = self.parameter_set(i, loss)
                w_1 = self.weights + np.multiply(tao * self.y[i], row[:len(self.weights)])
                w_2 = np.multiply(tao * self.y[i], row[len(self.weights):])
                self.weights = np.append(w_1, w_2)
                self.sparsity_step()
                # self.shuffle()

                train_error_vector.append(train_error / (i + 1))
            # total_error_vector = np.add(train_error_vector, total_error_vector)
        # total_error_vector = np.divide(total_error_vector, self.rounds)
        return train_error_vector

    def predict(self, X_test):
        prediction_results = np.zeros(len(X_test))
        for i in range(0, len(X_test)):
            row = X_test[i]
            prediction_results[i] = np.sign(np.dot(self.weights, row[:len(self.weights)]))
        return prediction_results

    def shuffle(self):
        random.seed(50)
        if self.mode == 'stream':
            c = list(zip(self.X, self.y))
            random.shuffle(c)
            self.X, self.y = zip(*c)


def preprocess_data(data, mode='variable'):
    random.seed(50)
    copydata = copy.deepcopy(data)
    random.shuffle(copydata)
    if mode == 'trapezoidal': dataset = preprocess2.remove_features_trapezoidal(copydata)
    if mode == 'variable': dataset = preprocess2.remove_features_random(copydata)
    all_keys = set().union(*(d.keys() for d in dataset))
    X, y = [], []
    for row in dataset:
        for key in all_keys:
            if key not in row.keys(): row[key] = 0
        y.append(row['class_label'])
        del row['class_label']
    if 0 not in row.keys(): start = 1
    if 0 in row.keys(): start = 0
    for row in dataset:
        x_row = []
        for i in range(start, len(row)):
            x_row.append(row[i])
        X.append(x_row)
    return X, y


