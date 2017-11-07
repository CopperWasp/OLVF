 # needs to get a dictionary as input
from sklearn.linear_model import SGDClassifier
import miscMethods as misc
import parameters
import numpy as np
import copy
import preprocess2
from sklearn.base import clone

class classifier:
    def __init__(self):
        
        self.classifier= SGDClassifier(average=True, max_iter=1,  penalty='l2', learning_rate="constant", eta0=1)
        

        self.classifier_summary=[] #to plot change in classifier dimension through training
        self.mean_value_dict={} #keep the mean values to fill missing data when needed, mean filling approach


    def fit(self, training_set, training_labels):

        self.initialize(training_set, training_labels)
        for i in range(0, parameters.rounds):
            train_error=0
            train_error_vector=[]
            
            self.set_classifier() #
            #self.set_metadata() #
            
            for i in range(0, len(self.X)):
                
                row=self.X[i].copy()
                label=self.y[i]
                
                if len(row)==0:
                    train_error_vector.append(train_error/(i+1))
                    continue

                #attributes for train
                oldKeys= misc.findDifferentKeys(self.weight_dict, row)
                row_ex= self.extendDictionary(row, oldKeys)
                newKeys= misc.findDifferentKeys(row, self.weight_dict)
                weights_ex= self.extendDictionary(self.weight_dict, newKeys)                               

                train_error+=self.predictSingle(row_ex, weights_ex, label)
                train_error_vector.append(train_error/(i+1))
                
                self.updateWeights(row_ex, weights_ex, label)

                #record classifier lengths
                self.classifier_summary.append(len(self.weight_dict))

          
        return self.classifier_summary, train_error_vector

    def updateWeights(self, row, weights, label):

        row_vec=misc.dictToNumpyArray(row)
        weights_vec=misc.dictToNumpyArray(weights)
        self.classifier.coef_=np.array([weights_vec])
        
        self.classifier.partial_fit([row_vec], [label], np.unique(self.y))


        self.weight_dict= self.repackDict(self.classifier.coef_[0], list(weights.keys()))

    
    def predictSingle(self, row, weights, label):
        row_vec= misc.dictToNumpyArray(row)
        weights_vec= misc.dictToNumpyArray(weights)
        #print(weights_vec)
        self.classifier.coef_= np.array([weights_vec])
        return label!=self.classifier.predict([row_vec])        
        
    def extendDictionary(self, d, keys):
        copyd=copy.deepcopy(d)
        for key in keys:
            copyd[key]=0
        return copyd
            
            
    def set_classifier(self):
        self.classifier= clone(self.classifier)
        
        self.classifier.alpha=1
        self.classifier.eta0=1
        
        self.classifier.partial_fit(self.init_classifier, [self.y[0]], np.unique(self.y))
        self.weight_dict= misc.numpyArrayToDict(self.classifier.coef_[0], list(self.X[0].keys()))
        
        
    def initialize(self, training_set, training_labels):
        self.X= training_set
        self.y= training_labels
        self.init_classifier=np.zeros(len(self.X[0])).reshape(1,-1)
        
    def repackDict(self, values, keys):
        d={}
        #print(len(keys))
        #print(len(values))
        for i in range(0, len(keys)):
            d[keys[i]]=values[i]
        return d


data= preprocess2.readGermanNormalized()
#data2=preprocess2.removeRandomData(data)
X=[]
y=[]
for row in data:
    y.append(int(row['class_label']))
    del row['class_label']
    X.append(row)
    

c= classifier()
summary, error_vector= c.fit(X, y)
misc.plotError(error_vector, "german")

