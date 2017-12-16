import numpy as np
import preprocess
import random
import copy
import parameters as p


class olvf:
    def __init__(self, data):
        self.C = p.olvf_C
        self.Lambda = p.olvf_Lambda
        self.B = p.olvf_B
        self.option = p.olvf_option
        self.data = data
        self.rounds = p.cross_validation_folds

    
    def updateMetadata(self, i):
        classifier= self.weights
        for index in range(0, len(self.X[i])):
            if self.X[i][index]!=0:
                self.variance_vector[index]=(np.square(classifier[index]-self.average_vector[index])+(self.variance_vector[index]*self.count_vector[index]))/(self.count_vector[index]+1)
                self.average_vector[index]=((self.average_vector[index]*self.count_vector[index])+classifier[index])/(self.count_vector[index]+1)
                self.count_vector[index]+=1
            #decaying variance
            #else:
                #self.variance_vector[index] *= ((self.count_vector[index]-1) / (self.count_vector[index]))
                
        
    def set_metadata(self):
        self.variance_vector=[1 for i in range(0, len(self.X[0]))]
        self.average_vector=[0 for i in range(0, len(self.X[0]))]
        self.count_vector=[1 for i in range(0, len(self.X[0]))]
        
        
    def set_classifier(self):
        self.weights= np.zeros(len(self.X[0]))
        
        
    def scale_loss(self, i):
        inconfidence=0
        nonzerosum=0
        
        for index in range(0, len(self.X[i])):
            if self.X[i][index]!=0:
                nonzerosum+=self.variance_vector[index]
        
        for index in range(0, len(self.X[i])):
            if self.X[i][index]!=0:
                inconfidence+=(self.variance_vector[index]/nonzerosum)/np.count_nonzero(self.X[i])
        # print(inconfidence)
        return inconfidence*p.phi
         
       
    def parameter_set(self, i, loss):
        inner_product = np.dot(self.X[i], self.X[i])
        if inner_product == 0:
            inner_product = 1

        if self.option==0: return loss/inner_product
        if self.option==1: return np.minimum(self.C, loss/inner_product)
        if self.option==2: return loss/((1/(2*self.C))+inner_product)


    def sparsity_step(self):
        projected= np.multiply(np.minimum(1, self.Lambda/np.linalg.norm(self.weights, ord=1)), self.weights)
        self.weights= self.truncate(projected)
        
        
    def truncate(self, projected):
        if np.linalg.norm(projected, ord=0)> self.B*len(projected):
            remaining= int(np.maximum(1, np.floor(self.B*len(projected))))
            for i in projected.argsort()[:(len(projected)-remaining)]:
                projected[i]=0
            return projected
        else: return projected
                
        
    def fit(self):
        print("OLVF")
        random.seed(p.random_seed)
        for i in range(0, self.rounds):
            self.getShuffledData()
            print("Round: "+str(i))
            self.set_classifier()
            self.set_metadata()
            train_error=0
            train_error_vector=[]
            total_error_vector= np.zeros(len(self.y))

            for i in range(0, len(self.y)):
                row= self.X[i][:len(self.X[i])]
                y_hat= np.sign(np.dot(self.weights, row[:len(self.weights)]))
                if len(row)==0: continue
                if y_hat!=self.y[i]: train_error+=1
                loss= (np.maximum(0, (1-self.y[i]*(np.dot(self.weights, row[:len(self.weights)])))))*self.scale_loss(i)
                tao= self.parameter_set(i, loss)
                self.weights= self.weights+np.multiply(tao*self.y[i], row[:len(self.weights)])
                self.updateMetadata(i)
                self.sparsity_step()
                train_error_vector.append(train_error/(i+1))

            total_error_vector= np.add(train_error_vector, total_error_vector)
        total_error_vector= np.divide(total_error_vector, self.rounds)
        return train_error_vector
        
        
    def predict(self, X_test):
        prediction_results=np.zeros(len(X_test))
        for i in range (0, len(X_test)):
            row= X_test[i]
            prediction_results[i]= np.sign(np.dot(self.weights, row[:len(self.weights)]))
        return prediction_results

    
    def getShuffledData(self, mode=p.stream_mode): # generate data for cross validation
        data=self.data
        copydata= copy.deepcopy(data)
        random.shuffle(copydata)
        if mode=='trapezoidal': dataset=preprocess.removeDataTrapezoidal(copydata)
        if mode=='variable': dataset=preprocess.removeRandomData(copydata)
        else: dataset= copydata
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
        self.X, self.y = X, y              






    
    
