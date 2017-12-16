import numpy as np
import preprocess
import random
import copy
import parameters as p


class olsf:
    def __init__(self, data):
        self.C=p.olsf_C
        self.Lambda=p.olsf_Lambda
        self.B=p.olsf_B
        self.option=p.olsf_option
        self.data=data
        self.rounds = p.cross_validation_folds

             
    def set_classifier(self):
        self.weights= np.zeros(np.count_nonzero(self.X[0]))
        
        
    def parameter_set(self, i, loss):
        if self.option==0: return loss/np.dot(self.X[i], self.X[i])
        if self.option==1: return np.minimum(self.C, loss/np.dot(self.X[i], self.X[i]))
        if self.option==2: return loss/((1/(2*self.C))+np.dot(self.X[i], self.X[i]))
        
        
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
        print("OLSF")
        random.seed(p.random_seed) #set seed to make consistent experiments
        for i in range(0, self.rounds):
            print("Round: "+str(i))
            self.getShuffledData() # each round get shuffled
            train_error=0
            train_error_vector=[]
            total_error_vector= np.zeros(len(self.y))
            
            self.set_classifier()
            
            for i in range(0, len(self.y)):
                row= self.X[i][:np.count_nonzero(self.X[i])]
                if len(row)==0: continue
                y_hat= np.sign(np.dot(self.weights, row[:len(self.weights)]))
                if y_hat!=self.y[i]: train_error+=1
                loss= (np.maximum(0, 1-self.y[i]*(np.dot(self.weights, row[:len(self.weights)]))))
                tao= self.parameter_set(i, loss)
                w_1= self.weights+np.multiply(tao*self.y[i], row[:len(self.weights)])
                w_2= np.multiply(tao*self.y[i], row[len(self.weights):])
                self.weights= np.append(w_1, w_2)
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


    def getShuffledData(self): # generate data for cross validation
        data=self.data
        copydata= copy.deepcopy(data)
        random.shuffle(copydata)
        dataset=preprocess.removeDataTrapezoidal(copydata)
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
        

