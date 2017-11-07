 # needs to get a dictionary as input
import miscMethods as misc
import parameters
import numpy as np
import warnings
import sys
import math
import operator
import olsf

global_phi=0.002

class classifier:
    def __init__(self, training_set=[], test_set=[]):
        self.epoch=parameters.epoch
        self.mean_dict={}
        self.covariance_dict={}
        self.prediction_accuracy=0
        self.training_dataset=training_set #training set
        self.test_dataset=test_set #test set
        self.classifier_summary=[] #to plot change in classifier dimension through training
        self.mean_value_dict={} #keep the mean values to fill missing data when needed, mean filling approach
        self.train_error=0

    def train(self): #use self.training_dataset
        self.setInitialClassifier(self.training_dataset[0].copy())
        self.train_error=0
        for i in range(0, self.epoch):
            for original_row in self.training_dataset:
                row=original_row.copy() #copy the example to not make changes on original, otherwise no labels for following heldouts
                #check and record training error, for streaming accuracy
                if(len(row))==1: #empty row comes
                    continue
                if self.testSingle(row)==False:
                    self.train_error+=1
                label=row.pop('class_label', None) #get the class label of example and pop it from the dictionary
                #update mean value counts, mean fill aproach
                if parameters.meanFiller==1:
                    self.meanRecorder(row)
                    row=self.meanFiller(row) #filled row
                #these dicts will be merged, needs initialization to generalize merging
                old_partial_mean_dict={}
                old_partial_covariance_dict={}
                common_mean_dict={}
                common_covariance_dict={}
                new_partial_mean_dict={}
                new_partial_covariance_dict={}
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
                    common_mean_dict, common_covariance_dict, indicator=self.learnCommon(mean_subset, covariance_subset, row_subset, label)
                    #if classified large margin, then dont learn new attributes, skip to next
                    if indicator==1:
                        continue
                #New attributes
                if bool(misc.subsetDictionary(row, misc.findDifferentKeys(row, self.mean_dict)))==True: #it means there are new attributes
                    new_attribute_dict=misc.subsetDictionary(row, misc.findDifferentKeys(row, self.mean_dict))
                    new_partial_mean_dict, new_partial_covariance_dict=self.learnNew(new_attribute_dict, label)
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
                self.mean_dict=self.sparsity_step() #only works if sparsity parameter is on
                self.impute() #handle overflow and underflow
        #to plot change in classifier dimension through training, and train error for stream accuracy
        return self.classifier_summary, self.train_error/(len(self.training_dataset)*self.epoch)

    def testSingle(self, original_row):
            row=original_row.copy()
            label=row.pop('class_label', None) #extract the label
            commonKeys=misc.findCommonKeys(row, self.mean_dict) #do the classifications based on shared keys
            row_subset=misc.subsetDictionary(row, commonKeys)
            mean_subset=misc.subsetDictionary(self.mean_dict, commonKeys)
            row_vector=misc.dictToNumpyArray(row_subset)
            mean_vector=misc.dictToNumpyArray(mean_subset)
            covariance_vector=misc.dictToNumpyArray(misc.subsetDictionary(self.covariance_dict, commonKeys)) #this is for alternative classification style
            return self.classify(row_vector, mean_vector, covariance_vector, label) #check if it can classify correctly
            
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
    
    def learnCommon(self, mean_dict, covariance_dict, row_dict, label): #return 1 if correctly classified
        mean_vector=misc.dictToNumpyArray(mean_dict)
        row_vector=misc.dictToNumpyArray(row_dict)
        covariance_matrix=misc.dictToUnitNumpyMatrix(covariance_dict)
        covsum=np.sum(covariance_matrix)

        #
        V= np.dot(np.dot(row_vector, covariance_matrix), row_vector)
        phi= global_phi/(V/covsum)
        M=(row_vector.dot(mean_vector))*label
        part1= -(1+2*phi*M)
    
        #print("Part 1:"+str(M))


        
        #print("row vector: "+str(row_vector))
        #print("covariance matrix: "+ str(covariance_matrix))
        #print("v"+str(V))
        part2= np.sqrt(np.square(part1)-8*phi*(M - phi* V))
        #print("part2: "+str(part2))
        part3= 4*phi*V
        gamma= (part1 + part2)/ part3
        #print("gamma: "+str(gamma))
        alpha= np.max(gamma, 0)
        mean_vector=mean_vector + np.dot(np.multiply(alpha*label, covariance_matrix), row_vector)
        #print(mean_vector)
        covariance_matrix=covariance_matrix+ np.multiply(2*alpha*phi, np.diag(row_vector))

        #transform everything back to labeled dictionary
        new_mean_dict=misc.numpyArrayToDict(mean_vector, list(mean_dict.keys()))
        new_covariance_dict=misc.numpyMatrixToDict(covariance_matrix, list(mean_dict.keys()))
        return new_mean_dict, new_covariance_dict, 0

    def learnNew(self, new_attribute_dict, label):
        #mean_new=example_new/(label*example_new*example_new)
        new_attribute_vector=misc.dictToNumpyArray(new_attribute_dict)
        new_partial_mean, new_covariance_dict=self.updateNewPart(new_attribute_vector, label, new_attribute_dict)
        new_mean_dict=misc.numpyArrayToDict(new_partial_mean, list(new_attribute_dict.keys()))
        return new_mean_dict, new_covariance_dict
   
    def computeNewPartialMean(self, new_attribute_vector, label): #compute mean values for new features
        with warnings.catch_warnings(record=True) as w:
            if w:
                print("Label: "+str(label))
                print("New attribute vector: "+str(new_attribute_vector))
                print("Dot product result: "+str(misc.dot([label, new_attribute_vector, new_attribute_vector])))
                warnings.simplefilter("error")  # Cause all warnings to always be triggered.
            return new_attribute_vector/misc.dot([label, new_attribute_vector, new_attribute_vector])

    def computeNewPartialCovariance(self, new_attribute_dict): #put cov values=1 for new attributes
        return dict((key, 1) for key in new_attribute_dict.keys())
    
    def updateNewPart(self, new_attribute_vector, label, new_attribute_dict):
        new_partial_mean=self.computeNewPartialMean(new_attribute_vector, label)
        new_partial_covariance=self.computeNewPartialCovariance(new_attribute_dict)
        return new_partial_mean, new_partial_covariance
    
    def classify(self, row_vector, mean_vector, covariance_vector, label):
        return (np.sign(row_vector.dot(mean_vector))==label)