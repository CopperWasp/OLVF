 # needs to get a dictionary as input
import algorithm
import miscMethods as misc
import parameters
import numpy as np
import warnings
import math
import copy


class classifier:
    def __init__(self, training_set=[], test_set=[], selector=2, C=10, param_lambda=30, B=0.01):
        self.epoch=parameters.epoch
        self.weight_dict={}
        self.prediction_accuracy=0
        self.algorithm_methods=algorithm.algorithm()
        self.training_dataset=training_set #training set
        self.test_dataset=test_set #test set
        self.classifier_summary=[] #to plot change in classifier dimension through training
        self.train_error=0
        self.selector=selector
        self.C=C
        self.param_lambda=param_lambda
        self.B=B

    def train(self): #use self.training_dataset
        if len(self.weight_dict)==0:
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
                row=original_row.copy()
                #check and record training error, for streaming accuracy
                label=row['class_label'] #get the class label of example and pop it from the dictionary
                loss, product=self.testSingle(row)
                if product<=0:
                    train_error+=1
                train_error_vector.append(train_error/iterations)
                tao=self.setParameter(loss, row)
                row.pop('class_label', None)
                #these dicts will be merged, needs initialization to generalize merging
                common_weight_dict={}
                new_weight_dict={}
                #Shared attributes
                if bool(misc.findCommonKeys(row, self.weight_dict))==True:
                    commonKeys=misc.findCommonKeys(row, self.weight_dict)
                    row_subset=misc.subsetDictionary(row, commonKeys)
                    weight_subset=misc.subsetDictionary(self.weight_dict, commonKeys)
                    common_weight_dict=self.learnCommon(weight_subset, row_subset, label, tao)
                #New attributes
                if bool(misc.subsetDictionary(row, misc.findDifferentKeys(row, self.weight_dict)))==True: #it means there are new attributes
                    new_attribute_dict=misc.subsetDictionary(row, misc.findDifferentKeys(row, self.weight_dict))
                    new_weight_dict=self.learnNew(new_attribute_dict, label, tao)
                #Merge mean and covariance dictionaries
                #merge means
                common_weight_dict.update(new_weight_dict)
                self.weight_dict=common_weight_dict
                #sparsify the current classifier
                self.impute() #handle overflow and underflow
                self.weight_dict=self.sparsity_step() #only works if sparsity parameter is on
                #record classifier lengths
                self.classifier_summary.append(len(self.weight_dict))
                
        #to plot change in classifier dimension through training, and train error for stream accuracy
        return self.classifier_summary, train_error_vector

    def testSingle(self, original_row):
            row=original_row.copy()
            label=row.pop('class_label', None) #extract the label
            commonKeys=misc.findCommonKeys(row, self.weight_dict) #do the classifications based on shared keys
            row_subset=misc.subsetDictionary(row, commonKeys)
            weight_subset=misc.subsetDictionary(self.weight_dict, commonKeys)
            row_vector=misc.dictToNumpyArray(row_subset)
            weight_vector=misc.dictToNumpyArray(weight_subset)
            loss=np.maximum(0, 1-label*(weight_vector.dot(row_vector)))
            product=label*(weight_vector.dot(row_vector))
            return loss, product #check if it can classify correctly

    def setInitialClassifier(self, firstrow):
        del firstrow['class_label']
        for key, value in firstrow.items():
            self.weight_dict[key]=0
            
    def impute(self): #this is for possible nan values created by overflow/underflow
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            mean=np.nanmean(list(self.weight_dict.values()))
            for key, value in self.weight_dict.items():
                if math.isnan(value):
                    self.weight_dict[key]=mean #replace is by mean

    def setParameter(self, loss, row):
        row= misc.dictToNumpyArray(row)
        if self.selector==0: #OLSF
            return loss/((np.linalg.norm(row))*((np.linalg.norm(row))))
        elif self.selector==1: #OLSF1
            return np.minimum(self.C, loss/((np.linalg.norm(row))*((np.linalg.norm(row)))))
        else: #OLSF2
            return loss/((np.linalg.norm(row)*np.linalg.norm(row))+(1/(2*self.C)))

    def learnCommon(self, weight_dict, row_dict, label, tao):
        weight_vector=misc.dictToNumpyArray(weight_dict)
        row_vector=misc.dictToNumpyArray(row_dict)
        new_weight_vector= weight_vector+ np.multiply(tao*label, row_vector)
        new_weight_dict=misc.numpyArrayToDict(new_weight_vector, list(weight_dict.keys()))
        return new_weight_dict

    def learnNew(self, row_dict, label, tao):
        row_vector=misc.dictToNumpyArray(row_dict)
        new_weight_vector= np.multiply(tao*label, row_vector)
        new_weight_dict=misc.numpyArrayToDict(new_weight_vector, list(row_dict.keys()))
        return new_weight_dict

    def sparsity_step(self):
        weight_vector=misc.dictToNumpyArray(self.weight_dict)
        #l1 projection
        new_weight_vector= np.multiply(np.minimum(1, self.param_lambda/ np.linalg.norm(weight_vector, ord=1)), weight_vector)
        #truncation
        return_dict=self.weight_dict
        if np.linalg.norm(new_weight_vector, ord=0) >= self.B*len(new_weight_vector):
            num_elem_remove=int(len(new_weight_vector)- np.maximum(1, np.floor(self.B* len(new_weight_vector))))
            copy_dict= copy.deepcopy(return_dict)
            sorted_keys= sorted(copy_dict, key=lambda dict_key: abs(copy_dict[dict_key]))
            for key in sorted_keys[:num_elem_remove]:
                del return_dict[key]
        return return_dict