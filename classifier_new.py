 # needs to get a dictionary as input
import algorithm
import miscMethods as misc
import parameters
import numpy as np
import warnings
import sys
import math
import operator
import olsf



class classifier:
    def __init__(self, training_set=[], test_set=[]):
        self.epoch=parameters.epoch
        self.mean_dict={}
        self.covariance_dict={}
        self.prediction_accuracy=0
        self.algorithm_methods=algorithm.algorithm()
        self.training_dataset=training_set #training set
        self.test_dataset=test_set #test set
        self.classifier_summary=[] #to plot change in classifier dimension through training
        self.mean_value_dict={} #keep the mean values to fill missing data when needed, mean filling approach
        self.train_error=0

    def train(self): #use self.training_dataset
        #set initial arbitrary classifier with the dimensions of first example  
        #print("Train called")
        self.setInitialClassifier(self.training_dataset[0].copy())
        self.train_error=0
        #debug
        #print("After removing:")
        #print("Length of the current dataset:"+str(len(self.training_dataset)))
        #for i in range(0,5):
         #   print("len. element "+str(i)+": "+str(len(self.training_dataset[len(self.training_dataset)-1])))
        #debug
        for i in range(0, self.epoch):
            for original_row in self.training_dataset:
                row=original_row.copy() #copy the example to not make changes on original, otherwise no labels for following heldouts
                #check and record training error, for streaming accuracy
                if(len(row))==1: #empty row comes
                    continue
                if self.testSingle(row)==False:
                    self.train_error+=1
                label=row.pop('class_label', None) #get the class label of example and pop it from the dictionary
                #these dicts will be merged, needs initialization to generalize merging
                old_partial_mean_dict={}
                old_partial_covariance_dict={}
                common_mean_dict={}
                common_covariance_dict={}
                new_partial_mean_dict={}
                new_partial_covariance_dict={}
                #Shared attributes
                if bool(misc.findCommonKeys(row, self.mean_dict))==True:
                    commonKeys=misc.findCommonKeys(row, self.mean_dict)
                    row_subset=misc.subsetDictionary(row, commonKeys)
                    mean_subset=misc.subsetDictionary(self.mean_dict, commonKeys)
                    covariance_subset=misc.subsetDictionary(self.covariance_dict, commonKeys)
                    common_mean_dict, common_covariance_dict, indicator=self.algorithm_methods.learnCommon(mean_subset, covariance_subset, row_subset, label)
                    #if classified large margin, then dont learn new attributes, skip to next
                    if indicator==1:
                        continue
                #New attributes
                if bool(misc.subsetDictionary(row, misc.findDifferentKeys(row, self.mean_dict)))==True: #it means there are new attributes
                    new_attribute_dict=misc.subsetDictionary(row, misc.findDifferentKeys(row, self.mean_dict))
                    new_partial_mean_dict, new_partial_covariance_dict=self.algorithm_methods.learnNew(new_attribute_dict, label)
                #Merge mean and covariance dictionaries
                #merge means
                old_partial_mean_dict.update(common_mean_dict)
                old_partial_mean_dict.update(new_partial_mean_dict)
                self.mean_dict=old_partial_mean_dict
                #merge covariances
                old_partial_covariance_dict.update(common_covariance_dict)
                old_partial_covariance_dict.update(new_partial_covariance_dict)
                self.covariance_dict=old_partial_covariance_dict
                #record classifier lengths
                self.classifier_summary.append(len(self.mean_dict))
                #sparsify the current classifier
                #self.sparsity_step() #only works if sparsity parameter is on
                self.impute() #handle overflow and underflow
        #to plot change in classifier dimension through training, and train error for stream accuracy
        return self.classifier_summary, self.train_error/(len(self.training_dataset)*self.epoch)

    def test(self): #returns average test accuracy
        counter=0 #correct counter
        for original_row in self.test_dataset:
            row=original_row.copy()
            label=row.pop('class_label', None) #extract the label
            commonKeys=misc.findCommonKeys(row, self.mean_dict) #do the classifications based on shared keys
            row_subset=misc.subsetDictionary(row, commonKeys)
            mean_subset=misc.subsetDictionary(self.mean_dict, commonKeys)
            row_vector=misc.dictToNumpyArray(row_subset)
            mean_vector=misc.dictToNumpyArray(mean_subset)
            covariance_vector=misc.dictToNumpyArray(misc.subsetDictionary(self.covariance_dict, commonKeys)) #this is for alternative classification style
            if self.algorithm_methods.classify(row_vector, mean_vector, covariance_vector, label)==False: #check if it can classify correctly
                counter+=1
        return counter/len(self.test_dataset) #return number of false classification over all examples

    def testSingle(self, original_row):
            row=original_row.copy()
            label=row.pop('class_label', None) #extract the label
            commonKeys=misc.findCommonKeys(row, self.mean_dict) #do the classifications based on shared keys
            row_subset=misc.subsetDictionary(row, commonKeys)
            mean_subset=misc.subsetDictionary(self.mean_dict, commonKeys)
            row_vector=misc.dictToNumpyArray(row_subset)
            mean_vector=misc.dictToNumpyArray(mean_subset)
            covariance_vector=misc.dictToNumpyArray(misc.subsetDictionary(self.covariance_dict, commonKeys)) #this is for alternative classification style
            if len(covariance_vector)==0:
                return False
            if np.average(covariance_vector) > np.average(list(self.covariance_dict.values()))*2.2:
                return np.random.choice([True, False])
            if np.sum(mean_vector)!=0:
                #confidences= np.divide(np.reciprocal(covariance_vector), np.sum(np.reciprocal(covariance_vector)))
                mean_vector=np.multiply(np.divide(mean_vector, np.sum(np.abs(mean_vector))), np.sum(np.abs(list(self.mean_dict.values()))))
                #mean_vector= [a*b for a,b in zip(mean_vector,confidences)]
            return self.algorithm_methods.classify(row_vector, mean_vector, covariance_vector, label) #check if it can classify correctly
            
    def sparsify(self):
        with warnings.catch_warnings(record=True) as w:
            if parameters.sparsity_switch==0: #dont touch
                return
            lamb= parameters.sparsity_regularizer
            trunc=parameters.sparsity_B
            mean_vector=misc.dictToNumpyArray(self.mean_dict)
            covariance_vector=misc.dictToNumpyArray(self.covariance_dict)
            value_vector=covariance_vector
            if w:
                print("mean vector: "+str(min(mean_vector)))
                print("covariance vector: "+str(min(covariance_vector)))
                warnings.simplefilter("error")
                sys.exit()
            num_elem_remove=int(len(value_vector)*(1-trunc))
            if(np.linalg.norm(value_vector, ord=1) > trunc*len(value_vector)):
                coefficient=np.minimum(1, lamb/np.linalg.norm(value_vector, ord=1))
                copy_mean_dict=self.mean_dict.copy()
                #multiply by coefficient
                for key, value in copy_mean_dict.items():
                    copy_mean_dict[key]=value*coefficient
                #print("remove elements: "+str(num_elem_remove))
                for i in range(0, num_elem_remove):
                    key_to_delete = max(copy_mean_dict, key=lambda k: copy_mean_dict[k])
                    copy_mean_dict.pop(key_to_delete)
                    self.mean_dict.pop(key_to_delete)
                    self.covariance_dict.pop(key_to_delete)

    def setInitialClassifier(self, firstrow):
        del firstrow['class_label']
        for key, value in firstrow.items():
            self.mean_dict[key]=0
            self.covariance_dict[key]=1

    def impute(self): #this is for possible nan values created by overflow/underflow
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            mean=np.nanmean(list(self.mean_dict.values()))
            for key, value in self.mean_dict.items():
                if math.isnan(value):
                    self.mean_dict[key]=mean #replace is by mean

    def meanRecorder(self, row):
        #keep track of feature means
        for key in row.keys():
            if key not in self.mean_value_dict: #if its a new key generate (mean, count) value
                self.mean_value_dict[key]=(row[key], 1)
            else:
                value=self.mean_value_dict[key][0]
                count=self.mean_value_dict[key][1]
                new_value=(value*count+row[key])/(count+1)
                self.mean_value_dict[key]=(new_value, count+1)

    def meanFiller(self, row):
        #check training example, if it has high confidence missing values, fill it up with mean values
        filled_row=row.copy()
        for key in self.mean_dict.keys():
            if (key in self.mean_value_dict) and (key not in filled_row): #if i have a replacement for it and it h
                if self.isAboveConfidenceThreshold(key): #if high enough confidence, fill up with current mean
                    filled_row[key]=self.mean_value_dict[key][0]
        return filled_row

    def isAboveConfidenceThreshold(self, key):
        key_variance=self.covariance_dict[key]
        variance_vector=list(self.covariance_dict.values())
        #print("key:"+str(key)+", average:"+str(sum(variance_vector)/len(variance_vector)))
        #print("key_variance"+str(key_variance))
        #for now, simply check if the variance is above mean variance
        if key_variance> sum(variance_vector)/len(variance_vector):
             #print("not enough variance")
             return False
        else:
            return True
        
    def sparsity_step(self):
        weight_vector=misc.dictToNumpyArray(self.mean_dict)
        #l1 projection
        new_weight_vector= np.multiply(np.minimum(1, olsf.param_lambda/ np.linalg.norm(weight_vector, ord=1)), weight_vector)
        #truncation
        return_dict=self.mean_dict
        if np.linalg.norm(new_weight_vector, ord=0) >= olsf.B*len(new_weight_vector):
            num_elem_remove=int(len(new_weight_vector)- np.maximum(1, np.floor(olsf.B* len(new_weight_vector))))
            #print(len(return_dict))
            return_dict = dict(sorted(return_dict.items(), key=operator.itemgetter(1), reverse=True)[:num_elem_remove])
            #print(len(return_dict))
        return return_dict
