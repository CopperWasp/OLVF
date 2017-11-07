import preprocess2
import numpy as np
import matplotlib.pyplot as plt
import parameters
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.base import clone
import copy
import experiment
import time
import random

#german= [0.775000000000000,0.587500000000000,0.506250000000000,0.440625000000000,0.418750000000000,0.386718750000000,0.375390625000000,0.374804687500000,0.367187500000000]

class benchmark:
    def __init__(self, mode):
        self.classifier= SGDClassifier(average=True, max_iter=1,  penalty='l2', learning_rate="constant", eta0=1)
        if mode=='stream': self.rounds=parameters.rounds
        if mode=='online': self.rounds=1
        self.mode=mode
        
    def fit(self, training_set, training_labels):

        self.initialize(training_set, training_labels)
        for i in range(0, self.rounds):
            train_error=0
            train_error_vector=[]
            total_error_vector=np.zeros(len(self.y))
            self.set_classifier()
            self.set_metadata()
            if self.mode=='stream' : self.shuffle(), print(i)
            for i in range(0, len(self.y)):
                result= self.classifier.predict([self.X[i]])                
                if result[0]!=self.y[i] : train_error+=1 
                self.update_weights(i)
                self.update_metadata(i)
                self.update_alpha(i)
                #update error vector
                train_error_vector.append(train_error/(i+1))
            total_error_vector= np.add(train_error_vector, total_error_vector)
        total_error_vector=np.divide(total_error_vector, parameters.rounds)
        return train_error_vector
    
    def predict(self, X_test):
        prediction_results=np.zeros(len(X_test))
        for i in range (0, len(X_test)):
            prediction_results[i]= self.classifier.predict([X_test[i]])
        return prediction_results
    
    def shuffle(self):
        c = list(zip(self.X, self.y))
        random.shuffle(c)
        self.X, self.y = zip(*c)

    def initialize(self, training_set, training_labels):
        self.X= training_set
        self.y= training_labels
        #self.init_classifier=np.zeros(len(self.X[0])).reshape(1,-1)
        self.init_classifier=np.random.uniform(size=len(self.X[0])).reshape(1, -1)
        
    def set_classifier(self): #reset
        self.classifier= clone(self.classifier)
        
        self.classifier.alpha=1
        self.classifier.eta0=1
        
        self.classifier.partial_fit(self.init_classifier, [self.y[0]], np.unique(self.y))
    
    def set_metadata(self):
        self.variance_vector=[1 for i in range(0, len(self.X[0]))]
        self.average_vector=[0 for i in range(0, len(self.X[0]))]
        self.count_vector=[1 for i in range(0, len(self.X[0]))]
    
    def update_weights(self, i):
        self.classifier.partial_fit([self.X[i]], [self.y[i]], np.unique(self.y))
    
    def update_p(self):
        return 0.5
    
    def update_alpha(self, i):
        inconfidence=0
        for index in range(0, len(self.X[i])):
            if self.X[i][index]!=0:
                inconfidence+=self.variance_vector[index]/(np.count_nonzero(self.X[i]))
        
        self.classifier.eta0=(param/inconfidence*(i+1))
        self.classifier.alpha=1/(self.classifier.eta0)

        
    def update_metadata(self, i):
        classifier= self.classifier.coef_[0]
        for index in range(0, len(self.X[i])):
            if self.X[i][index]!=0:
                self.variance_vector[index]=(np.square(classifier[index]-self.average_vector[index])+(self.variance_vector[index]*self.count_vector[index]))/(self.count_vector[index]+1)
                self.average_vector[index]=((self.average_vector[index]*self.count_vector[index])+classifier[index])/(self.count_vector[index]+1)
                self.count_vector[index]+=1


def preprocessData(data, mode='trapezoidal'):
    random.seed(50)
    copydata= copy.deepcopy(data)
    random.shuffle(copydata)
    if mode=='trapezoidal': dataset=preprocess2.removeDataTrapezoidal(copydata)
    if mode=='variable': dataset=preprocess2.removeRandomData(copydata)
    all_keys = set().union(*(d.keys() for d in dataset))
    X,y = [],[]
    for row in dataset:
        for key in all_keys:
            if key not in row.keys() : row[key]=0
        y.append(row['class_label'])
        del row['class_label']
    if 0 not in row.keys(): start=1
    if 0 in row.keys(): start=0
    for row in dataset:
        X_row=[]
        for i in range(start, len(row)):
            X_row.append(row[i])
        X.append(X_row)
    return X,y

def olvf_stream(data, dataset_name):
    b= benchmark("stream")
    plot_list=[
               ("OLVF_random_sparse", experiment.variableFeatureExperiment, 0),
               ("ASGD", b.fit, 1),
               ("Perceptron", b.fit, 0),
               ("PA1", b.fit, 0),
               ("PA2", b.fit, 0)]
    
    x = list(range(len(data)))
    plotlist=[]
    X,y = preprocessData(data) #data is being shuffled inside
    for triple in plot_list:
        if triple[2]==1: plotlist.append((triple[1](X,y, dataset_name), triple[0]))
    for i in range(len(plotlist)):
        plt.plot(x[0::10], plotlist[i][0][0::10], label=plotlist[i][1])  
    plt.legend()
    plt.xlabel("Instance")
    plt.ylabel("Test Error Rate %")
    plt.title(dataset_name)
    plt.savefig('./figures/'+'compare_'+str(experiment.lambda_error)+str("_")+str(experiment.B)+str("_")+time.strftime("%H%M%S")+'.png')
    plt.show()
    
def olvf_online(data, dataset_name, shuffle_var):
    heldout = [0.95 ,0.90,  0.80,  0.70, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    #heldout = [0.90, 0.70, 0.5, 0.3, 0.1]
    xx = 1. - np.array(heldout)
    rounds = 10
    classifiers = [
    ("OLVF", benchmark("online")),
    #("SGD", SGDClassifier()),
    #("ASGD", SGDClassifier(average=True, max_iter=1)),
    #("ASGD2", SGDClassifier(average=True, max_iter=1)),
    #("Perceptron", Perceptron()),
    #("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge', C=1.0)),
    #("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge', C=1.0)),
    #("SAG", LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / len(data[0])))
    ]
    
    for name, clf in classifiers:
        print("training %s" % name)
        rng = np.random.RandomState(42)
        yy = []
        for i in heldout:
            print("heldout: "+str(i))
            yy_ = []
            for r in range(rounds):
                print("round: "+str(r))
                X,y = preprocessData(data) #remove and fill
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, random_state=rng, shuffle=shuffle_var)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                yy_.append(1 - np.mean(y_pred == y_test))
            yy.append(np.mean(yy_))
        plt.plot(xx, yy, label=name)
        #plt.plot(xx, german, label=name)
    plt.legend(loc="upper right")
    plt.xlabel("Proportion train")
    plt.ylabel("Test Error Rate")
    plt.savefig('./figures/'+'online_compare_'+str(experiment.lambda_error)+str("_")+str(experiment.B)+str("_")+time.strftime("%H%M%S")+'.png')
    plt.show()
    
def streamOverUCIDatasets():
    olvf_stream(preprocess2.readGermanNormalized(), "German")
    #olvf_stream(preprocess2.readIonosphereNormalized(), "Ionosphere")
    #olvf_stream(preprocess2.readMagicNormalized(), "Magic")
    #olvf_stream(preprocess2.readSpambaseNormalized(), "Spambase")
    #olvf_stream(preprocess2.readWdbcNormalized(), "WDBC")
    #olvf_stream(preprocess2.readWpbcNormalized(), "WPBC")
    #olvf_stream(preprocess2.readA8ANormalized(), "A8A")
    #olvf_stream(preprocess2.readSvmguide3Normalized(), "svmguide3")
    
def onlineOverUCIDatasets(shuffle_var):
    #olvf_online(preprocess2.readGermanNormalized(), "German", shuffle_var)
    #olvf_online(preprocess2.readIonosphereNormalized(), "Ionosphere", shuffle_var)
    olvf_online(preprocess2.readMagicNormalized(), "Magic", shuffle_var)
    olvf_online(preprocess2.readSpambaseNormalized(), "Spambase", shuffle_var)
    #olvf_online(preprocess2.readWdbcNormalized(), "WDBC", shuffle_var)
    #olvf_online(preprocess2.readWpbcNormalized(), "WPBC", shuffle_var)
    #olvf_online(preprocess2.readA8ANormalized(), "A8A", shuffle_var)
    #olvf_online(preprocess2.readSvmguide3Normalized(), "svmguide3", shuffle_var)
    
param=0.00005
