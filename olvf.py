# trapezoidal experiment needs shufflable method

import numpy as np
import preprocess2
import random
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import miscMethods
import olsf as sf

phi = 2


class OLVF:
    def __init__(self, mode):
        self.X = []
        self.y = []
        self.variance_vector = []
        self.average_vector = []
        self.count_vector = []
        self.weights = []

        self.C = 1
        self.Lambda = 30
        self.B = 0.64
        self.rounds = 1
        self.option = 2
        self.mode = mode
        if mode == 'stream':
            self.rounds = 20

    def update_metadata(self, i):
        classifier = self.weights
        for index in range(0, len(self.X[i])):
            if self.X[i][index] != 0:
                self.variance_vector[index] = (np.square(classifier[index]-self.average_vector[index])+(self.variance_vector[index]*self.count_vector[index]))/(self.count_vector[index]+1)
                self.average_vector[index] = ((self.average_vector[index]*self.count_vector[index])+classifier[index])/(self.count_vector[index]+1)
                self.count_vector[index] += 1
                
    def set_metadata(self):
        self.variance_vector = [1 for i in range(0, len(self.X[0]))]
        self.average_vector = [0 for i in range(0, len(self.X[0]))]
        self.count_vector = [1 for i in range(0, len(self.X[0]))]
        
    def initialize(self, data, labels):
        self.X = data
        self.y = labels
        
    def shuffle(self):
        random.seed(50)
        if self.mode == 'stream':
            c = list(zip(self.X, self.y))
            random.shuffle(c)
            self.X, self.y = zip(*c)
    
    def set_classifier(self):
        self.weights = np.zeros(len(self.X[0]))
        
    def scale_loss(self, i):
        inconfidence = 0
        nonzerosum = 0
        
        for index in range(0, len(self.X[i])):
            if self.X[i][index] != 0:
                nonzerosum += self.variance_vector[index]

        for index in range(0, len(self.X[i])):
            if self.X[i][index] != 0:
                inconfidence += (self.variance_vector[index]/nonzerosum)/np.count_nonzero(self.X[i])
            # print(inconfidence)
        return inconfidence*phi

    def parameter_set(self, i, loss):
        if self.option == 0:
            return loss/np.dot(self.X[i], self.X[i])
        if self.option == 1:
            return np.minimum(self.C, loss/np.dot(self.X[i], self.X[i]))
        if self.option == 2:
            return loss/((1/(2*self.C))+np.dot(self.X[i], self.X[i]))

    def sparsity_step(self):
        projected = np.multiply(np.minimum(1, self.Lambda/np.linalg.norm(self.weights, ord=1)), self.weights)
        self.weights = self.truncate(projected)
        
    def truncate(self, projected):
        if np.linalg.norm(projected, ord=0) > self.B*len(projected):
            remaining = int(np.maximum(1, np.floor(self.B*len(projected))))
            for i in projected.argsort()[:(len(projected)-remaining)]:
                projected[i] = 0
            return projected
        else:
            return projected
                
    def fit(self, data, labels):
        for i in range(0, self.rounds):
            self.initialize(data, labels)
            self.set_classifier()
            self.set_metadata()
            train_error = 0
            train_error_vector = []
            # total_error_vector = np.zeros(len(self.y))

            for i in range(0, len(self.y)):
                row = self.X[i][:len(self.X[i])]
                y_hat = np.sign(np.dot(self.weights, row[:len(self.weights)]))
                if len(row) == 0:
                    continue
                if y_hat != self.y[i]:
                    train_error += 1

                loss = (np.maximum(0, (1-self.y[i]*(np.dot(self.weights, row[:len(self.weights)])))))*self.scale_loss(i)
                tao = self.parameter_set(i, loss)
                self.weights = self.weights+np.multiply(tao*self.y[i], row[:len(self.weights)])
  
                self.update_metadata(i)
                self.sparsity_step()
                self.shuffle()
                
                train_error_vector.append(train_error/(i+1))
            # total_error_vector = np.add(train_error_vector, total_error_vector)
        # total_error_vector = np.divide(total_error_vector, self.rounds)
        return train_error_vector
                
    def predict(self, x_test):
        prediction_results = np.zeros(len(x_test))
        for i in range(0, len(x_test)):
            row = x_test[i]
            prediction_results[i] = np.sign(np.dot(self.weights, row[:len(self.weights)]))
        return prediction_results


def preprocess_data(data, mode='trapezoidal'):
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

    X, y = [], []

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
        X.append(X_row)
    return X, y
            
        
def olsf_online(data, dataset_name, shuffle_var):
    heldout = [0.90,  0.80,  0.70, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # heldout = [0.90, 0.70, 0.5, 0.3, 0.1]
    xx = 1. - np.array(heldout)
    rounds = 20
    classifiers = [
        ("OLVF", OLVF()),
        # ("SGD", SGDClassifier()),
        # ("ASGD", SGDClassifier(average=True, max_iter=1)),
        # ("ASGD2", SGDClassifier(average=True, max_iter=1)),
        # ("Perceptron", Perceptron()),
        # ("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge', C=1.0)),
        # ("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge', C=1.0)),
        # ("SAG", LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / len(data[0])))
    ]
    
    for name, clf in classifiers:
        print("training %s" % name)
        rng = np.random.RandomState(30)
        yy = []

        for i in heldout:
            print("heldout: "+str(i))
            yy_ = []
            for r in range(rounds):
                # print("round: "+str(r))
                X, y = preprocess_data(data)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, random_state=rng, shuffle=shuffle_var)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                yy_.append(1 - np.mean(y_pred == y_test))
            yy.append(np.mean(yy_))
        plt.plot(xx, yy, label=name)
        # plt.plot(xx, german, label=name)
    plt.legend(loc="upper right")
    plt.xlabel("Proportion train")
    plt.ylabel("Test Error Rate")
    plt.savefig('./figures/'+'olvf_'+str("_")+time.strftime("%H%M%S")+'.png')
    plt.show()
    
    feature_summary=[np.count_nonzero(row) for row in X]
    miscMethods.plot_features(feature_summary, "")
    

def olsf_stream(data, dataset_name):
    plot_list = [
               ("OLvf", OLVF("stream").fit, 1),
               ("OLsf", sf.OLSF("stream").fit, 1)
               ]
    
    x = list(range(len(data)))
    plotlist=[]
    X, y = preprocess_data(data)  # data is being shuffled inside
    # X= X[:3000]
    # y= y[:3000]
    # x=x[:3000]

    for triple in plot_list:
        if triple[2] == 1:
            plotlist.append((triple[1](X, y), triple[0]))
    for i in range(len(plotlist)):
        plt.plot(x[1:][0::int(len(y)/10)], plotlist[i][0][1:][0::int(len(y)/10)], label=plotlist[i][1], marker='o', linestyle='--')
    plt.legend()
    plt.xlabel("Instance")
    plt.ylabel("Test Error Rate %")
    plt.title(dataset_name)
    plt.savefig('./figures/'+'olvf_stream'+str("_")+time.strftime("%H%M%S")+'.png')
    plt.grid()
    plt.show()
    feature_summary=[np.count_nonzero(row) for row in X]
    miscMethods.plot_features(feature_summary, "")


def experiment_uci_stream():
    # olsf_stream(preprocess2.readUrlNormalized(1), "URL")
    # olsf_stream(preprocess2.readWbcNormalized(), "WBC")
    olsf_stream(preprocess2.read_german(), "German")
    # olsf_stream(preprocess2.readIonosphereNormalized(), "Ionosphere")
    # olsf_stream(preprocess2.readMagicNormalized(), "Magic")
    # olsf_stream(preprocess2.readSpambaseNormalized(), "Spambase")
    # olsf_stream(preprocess2.readWdbcNormalized(), "WDBC")
    # olsf_stream(preprocess2.readWpbcNormalized(), "WPBC")
    # olsf_stream(preprocess2.readA8ANormalized(), "A8A")
    #olsf_stream(preprocess2.read_svmguide3(), "svmguide3")


# def experiment_uci_online(shuffle_var):
    # olsf_online(preprocess2.read_german(), "German", shuffle_var)
    # olsf_online(preprocess2.readIonosphereNormalized(), "Ionosphere", shuffle_var)
    # olsf_online(preprocess2.readMagicNormalized(), "Magic", shuffle_var)
    # olsf_online(preprocess2.readSpambaseNormalized(), "Spambase", shuffle_var)
    # olsf_online(preprocess2.readWdbcNormalized(), "WDBC", shuffle_var)
    # olsf_online(preprocess2.readWpbcNormalized(), "WPBC", shuffle_var)
    # olsf_online(preprocess2.readA8ANormalized(), "A8A", shuffle_var)
    # olf_online(preprocess2.readSvmguide3Normalized(), "svmguide3", shuffle_var)

# onlineOverUCIDatasets(False)


experiment_uci_stream()
