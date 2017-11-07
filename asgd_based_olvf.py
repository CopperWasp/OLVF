# Author: Rob Zinkov <rob at zinkov dot com>
# License: BSD 3 clause
import preprocess2
import numpy as np
import miscMethods as misc
import matplotlib.pyplot as plt
import parameters
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
import random
import copy
import experiment
import time
from sklearn import preprocessing

class benchmark:

    def train(self, training_set, dataset_name, mode): #uses self.training_dataset        
        self.data=training_set
        if mode=='ASGD':
            self.classifier= SGDClassifier(average=True, max_iter=1, penalty='elasticnet', l1_ratio=0.5)
            self.classifier2= SGDClassifier(average=False, max_iter=1, penalty='elasticnet', l1_ratio=0.5)
        elif mode=='Perceptron':
            self.classifier= Perceptron(max_iter=1)
            self.classifier2= self.classifiers
        elif mode=='PA1':
            self.classifier= PassiveAggressiveClassifier(loss='hinge', C=-1.0, max_iter=1)
            self.classifier2= self.classifier
        elif mode=='PA2':
            self.classifier= PassiveAggressiveClassifier(loss='squared_hinge', C=1.0, max_iter=1)
            self.classifier2= self.classifier
            
        init=np.zeros(len(self.data[0])-1).reshape(1,-1)
        for i in range(0, parameters.rounds):
            train_error_vector=[]
            iterations=0
            train_error=0
            copydata=copy.deepcopy(self.data)
            random.shuffle(copydata)
            self.data_preprocessor(preprocess2.removeDataTrapezoidal(copydata)) #or trapezoidal
            self.classifier=clone(self.classifier)
            self.classifier2=clone(self.classifier2)
            self.classifier.partial_fit(init, [-self.y[0]], np.unique(self.y))
            self.classifier2.partial_fit(init, [-self.y[0]], np.unique(self.y))
            total_error_vector=np.zeros(len(self.y))
            #c = list(zip(self.X, self.y))
            #random.shuffle(c)
            #self.X, self.y= zip(*c)
            self.variance_vector=[np.ones(len(training_set[0])-1)]
            self.average_vector=[np.zeros(len(training_set[0])-1)]
            for i in range(0, len(self.y)):
                #self.classifier.densify()
                row= [self.X[i]]
                label= self.y[i]
                iterations=i+1
                old= self.classifier.coef_
                self.classifier.coef_= self.update_metadata(iterations)
                result= self.classifier.predict(row)
                self.classifier.coef_=old
                if result[0]!=label:
                    train_error+=1
                
                self.classifier.partial_fit(row, [self.y[i]], np.unique(self.y))
                self.classifier2.partial_fit(row, [self.y[i]], np.unique(self.y))

                #self.classifier.sparsify()
                train_error_vector.append(train_error/iterations)
            total_error_vector= np.add(train_error_vector, total_error_vector)
        total_error_vector=np.divide(total_error_vector, parameters.rounds)
        misc.plotError(train_error_vector[0::50], dataset_name)
        return train_error_vector
    
    def update_p(self):
        return 0.5

    def update_metadata(self, i):
        classifier= self.classifier2.coef_
        average= self.average_vector
        self.average_vector= np.divide(np.add(np.multiply(self.average_vector,i), classifier), i+1)
        difference=np.subtract(classifier, average)
        current_variance= np.absolute(difference)
        self.variance_vector= np.divide(np.add(np.multiply(i, self.variance_vector), current_variance), i+1)
        return np.divide(self.classifier.coef_, 30*preprocessing.normalize(self.variance_vector))
        
    def data_preprocessor(self, dataset):
        #find all keys, fill with 0
        all_keys = set().union(*(d.keys() for d in dataset))
        X=[]
        y=[]
        for row in dataset:
            for key in all_keys:
                if key not in row.keys():
                    row[key]=0
            y.append(row['class_label'])
            del row['class_label']
   
        for row in dataset:
            X_row=[]
            for i in range(0, len(row)):
                X_row.append(row[i])
            X.append(X_row)
        self.X= X
        self.y= y

def compare_benchmarks(readed_dataset, dataset_name):
    b= benchmark()
    plot_list=[
               ("OLVF_random_sparse", experiment.variableFeatureExperiment, 0),
               ("ASGD", b.train, 1),
               ("Perceptron", b.train, 0),
               ("PA1", b.train, 0),
               ("PA2", b.train, 0)]
    x = list(range(len(readed_dataset)))
    plotlist=[]
    for triple in plot_list:
        if triple[2]==1:
            plotlist.append((triple[1](readed_dataset, dataset_name, mode=triple[0]), triple[0]))
    for i in range(len(plotlist)):
        plt.plot(x[0::50], plotlist[i][0][0::50], label=plotlist[i][1])  
    plt.legend()
    plt.xlabel("Instance")
    plt.ylabel("Test Error Rate %")
    plt.title(dataset_name)
    plt.savefig('./figures/'+'compare_'+str(experiment.lambda_error)+str("_")+str(experiment.B)+str("_")+time.strftime("%H%M%S")+'.png')
    plt.show()

def compareBenchmarksOverUCIDatasets():
    compare_benchmarks(preprocess2.readGermanNormalized(), "German")
    compare_benchmarks(preprocess2.readIonosphereNormalized(), "Ionosphere")
    compare_benchmarks(preprocess2.readMagicNormalized(), "Magic")
    compare_benchmarks(preprocess2.readSpambaseNormalized(), "Spambase")
    compare_benchmarks(preprocess2.readWdbcNormalized(), "WDBC")
    compare_benchmarks(preprocess2.readWpbcNormalized(), "WPBC")
    #compare_classifiers(preprocess2.readA8ANormalized(), "A8A")
    compare_benchmarks(preprocess2.readSvmguide3Normalized(), "svmguide3")
    
compareBenchmarksOverUCIDatasets()