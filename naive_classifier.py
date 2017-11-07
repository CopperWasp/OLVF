import miscMethods as misc
import numpy as np
import copy

class classifier:
    def __init__(self, training_dataset, test_dataset=[]):
        self.metadata={} #dictionary of features
        self.training_dataset= copy.deepcopy(training_dataset)
        
    def updateMean(self, mean, count, val):
        return (mean*count+val)/(count+1)
        
    def updateVar(self, var, count, val, mean):
        return (var*count + np.square(mean-val))/(count+1)
    
    def updateCount(self, count):
        return count+1
    
    def calculateClassProbability(self, class_label, row_dict):
        result=1
        for key, value in row_dict.items():
            mean= self.metadata[key][class_label][0]
            var= self.metadata[key][class_label][1]
            result*=np.abs(mean-value)/var
        return result
    
    def predict(self, class_label, row_dict):
        if self.calculateClassProbability(1, row_dict)<self.calculateClassProbability(-1, row_dict):
            return class_label*1
        else:
            return class_label*-1

    def train(self):
        error_count=0
        feature_summary=[]
        for row_dict in self.training_dataset:
            #print(row_dict['class_label'])
            label=row_dict.pop('class_label', None)
            feature_summary.append(len(row_dict))
            commonKeys=misc.findCommonKeys(row_dict, self.metadata)
            row_subset=misc.subsetDictionary(row_dict, commonKeys)
            new_attributes=misc.subsetDictionary(row_dict, misc.findDifferentKeys(row_dict, self.metadata))
            #try to classify first
            if self.predict(label, row_subset)==-1:
                error_count+=1
            #update metadata
                for key, value in new_attributes.items():
                    self.metadata[key]={label:[value, 1, 1], -label:[0, 1, 0]} #mean, variance, count
                    
                for key, value in row_subset.items():
                    data=self.metadata[key][label]
                    self.metadata[key][label]=[self.updateMean(data[0],data[2],value),
                                               self.updateVar(data[1], data[2], value, data[0]),
                                               self.updateCount(data[2])]
        return feature_summary, error_count/len(self.training_dataset)
            
            
        
        