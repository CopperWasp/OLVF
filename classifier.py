 # needs to get a dictionary as input
import algorithm
import miscMethods as misc
import parameters
import numpy as np
import warnings
import math
import experiment
import copy

class classifier:
    def __init__(self, training_set=[], test_set=[], sparse=0):
        self.epoch=parameters.epoch
        self.mean_dict={}
        self.covariance_dict={}
        self.prediction_accuracy=0
        self.train_error=0
        self.algorithm_methods=algorithm.algorithm()
        self.training_dataset=training_set #training set
        self.test_dataset=test_set #test set
        self.classifier_summary=[] #to plot change in classifier dimension through training
        self.mean_value_dict={} #keep the mean values to fill missing data when needed, mean filling approach
        self.sparse=sparse


    def train(self): #uses self.training_dataset
        if len(self.mean_dict)==0:
            self.setInitialClassifier(self.training_dataset[0].copy())

        for i in range(0, self.epoch):
            train_error_vector=[]
            train_error=0
            iterations=0
            for original_row in self.training_dataset:
                iterations+=1
                if(len(original_row))<=1: #empty row comes
                    #train_error+=1
                    train_error_vector.append(train_error/iterations)
                    continue
                row=original_row.copy() #copy the example to not make changes on original
                if self.testSingle(row)==False:
                    train_error+=1
                train_error_vector.append(train_error/iterations)
                #init dicts
                old_partial_mean_dict={}
                old_partial_covariance_dict={}
                common_mean_dict={}
                common_covariance_dict={}
                new_partial_mean_dict={}
                new_partial_covariance_dict={}
                label=row.pop('class_label', None) #get the class label of example and pop it from the dictionary
                #Old attributes
                if bool(misc.subsetDictionary(self.mean_dict, misc.findDifferentKeys(self.mean_dict, row)))==True:
                    old_partial_mean_dict=misc.subsetDictionary(self.mean_dict, misc.findDifferentKeys(self.mean_dict, row))
                    old_partial_covariance_dict=misc.subsetDictionary(self.covariance_dict, misc.findDifferentKeys(self.mean_dict, row))
                #Shared attributes
                if bool(misc.findCommonKeys(row, self.mean_dict))==True:
                    commonKeys=misc.findCommonKeys(row, self.mean_dict)
                    row_subset=misc.subsetDictionary(row, commonKeys)
                    mean_subset=misc.subsetDictionary(self.mean_dict, commonKeys)
                    covariance_subset=misc.subsetDictionary(self.covariance_dict, commonKeys)
                    common_mean_dict, common_covariance_dict, large_margin=self.algorithm_methods.learnCommon(mean_subset, covariance_subset, row_subset, label, self.covariance_dict)
                    #if classified large margin, then dont learn new attributes, skip to next
                    if large_margin==1:
                        continue
                #New attributes
                if bool(misc.subsetDictionary(row, misc.findDifferentKeys(row, self.mean_dict)))==True:
                    new_attribute_dict=misc.subsetDictionary(row, misc.findDifferentKeys(row, self.mean_dict))
                    new_partial_mean_dict, new_partial_covariance_dict=self.algorithm_methods.learnNew(new_attribute_dict, label)
                #Merge mean and covariance dictionaries
                old_partial_mean_dict.update(common_mean_dict)
                old_partial_mean_dict.update(new_partial_mean_dict)
                self.mean_dict=old_partial_mean_dict
                old_partial_covariance_dict.update(common_covariance_dict)
                old_partial_covariance_dict.update(new_partial_covariance_dict)
                self.covariance_dict=old_partial_covariance_dict
                #record classifier lengths
                self.classifier_summary.append(len(self.mean_dict))
                #sparsify the current classifier
                self.impute() #handle overflow and underflow
                if self.sparse==1:
                    self.mean_dict=self.sparsity_step() #only works if sparsity parameter is on
                
        #to plot change in classifier dimension through training, and train error for stream accuracy
        #return self.classifier_summary, self.train_error/(len(self.training_dataset)*self.epoch)
        return self.classifier_summary, train_error_vector

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
            if np.average(covariance_vector) > np.average(list(self.covariance_dict.values()))*2:
                return np.random.choice([True, False])
            #if np.sum(mean_vector)!=0:
            #if 0 not in covariance_vector:
                #covariance_vector_norm=np.divide(covariance_vector, np.sum(covariance_vector))
               # if 0 not in covariance_vector_norm:
                #    mean_vector=np.divide(mean_vector, covariance_vector_norm)
                #confidences= np.divide(np.reciprocal(covariance_vector), np.sum(np.reciprocal(covariance_vector)))
                #mean_vector=np.multiply(np.divide(mean_vector, np.sum(np.abs(mean_vector))), np.sum(np.abs(list(self.mean_dict.values()))))
                #mean_vector= [a*b for a,b in zip(mean_vector,confidences)]
            return self.algorithm_methods.classify(row_vector, mean_vector, covariance_vector, label) #check if it can classify correctly
            
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
     
    def sparsity_step(self):
        weight_vector=misc.dictToNumpyArray(self.mean_dict)
        covariance_vector=misc.dictToNumpyArray(self.covariance_dict)
        #l1 projection
        new_weight_vector= np.multiply(np.minimum(1, experiment.param_lambda/ np.linalg.norm(weight_vector, ord=1)), weight_vector)
        new_weight_vector= np.divide(new_weight_vector, covariance_vector)
        #truncation
        return_dict=self.mean_dict
        if np.linalg.norm(new_weight_vector, ord=0) >= experiment.B*len(new_weight_vector):
            num_elem_remove=int(len(new_weight_vector)- np.maximum(1, np.floor(experiment.B* len(new_weight_vector))))
            copy_dict= copy.deepcopy(return_dict)
            sorted_keys= sorted(copy_dict, key=lambda dict_key: abs(copy_dict[dict_key]))
            for key in sorted_keys[:num_elem_remove]:
                del return_dict[key]
        return return_dict
