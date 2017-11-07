import numpy as np
import miscMethods as misc
import experiment
import warnings

class algorithm:
    def __init__(self):
        self.r_error=experiment.parameter_r_error
        self.r_confidence=experiment.parameter_r_confidence

    def learnCommon(self, mean_dict, covariance_dict, row_dict, label, full_covariance_dict):
        #order of the values in two dictionaries are the same
        #transform dicts to matrix, vector for math operations
        full_cov_vector=misc.dictToNumpyArray(full_covariance_dict)
        mean_vector=misc.dictToNumpyArray(mean_dict)
        row_vector=misc.dictToNumpyArray(row_dict)
        covariance_matrix=misc.dictToUnitNumpyMatrix(covariance_dict)
        if self.classify_train(row_vector, mean_vector, label):
            return mean_dict, covariance_dict, 1 #large margin classified
        else:
            #recalculate mean and covariance
            mean_vector, covariance_matrix=self.update(mean_vector, covariance_matrix, row_vector, label, full_cov_vector)
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

    #classify with sign function constraint, for test
    def classify(self, row_vector, mean_vector, covariance_vector, label):
        return (np.sign(row_vector.dot(mean_vector))==label)

    #classify with max margin constraint, for training
    def classify_train(self, row_vector, mean_vector, label):
        if ((row_vector.dot(mean_vector))*label<1):
            return False
        else:
            return True

    def update(self, mean_vector, covariance_matrix, row_vector, label, full_cov_vector):
        #scaled covariance matrix
        scaled_cov_mat= np.multiply(covariance_matrix, (np.sum(full_cov_vector)/np.maximum(1, float(np.sum(np.diag(covariance_matrix))))))
        beta_for_mean=self.computeBetaForMean(row_vector, covariance_matrix)
        beta_for_covariance=self.computeBetaForCovariance(row_vector, scaled_cov_mat)
        alpha=self.computeAlpha(row_vector, mean_vector, beta_for_mean, label)
        new_mean=self.computeMean(mean_vector, alpha, covariance_matrix, label, row_vector)
        new_covariance=self.computeCovariance(covariance_matrix, beta_for_covariance, row_vector)
        return new_mean, new_covariance

    def updateNewPart(self, new_attribute_vector, label, new_attribute_dict):
        new_partial_mean=self.computeNewPartialMean(new_attribute_vector, label)
        new_partial_covariance=self.computeNewPartialCovariance(new_attribute_dict)
        return new_partial_mean, new_partial_covariance

    def computeBetaForMean(self, row_vector, covariance_matrix): #beta=1/(example_tr*covariance*example+r), constant, passed to alpha
        return 1/(misc.dot([misc.tr(row_vector), covariance_matrix, row_vector])+self.r_error)

    def computeBetaForCovariance(self, row_vector, covariance_matrix): #beta=1/(example_tr*covariance*example+r), constant, passed covariance
        return 1/(misc.dot([misc.tr(row_vector), covariance_matrix, row_vector])+self.r_confidence)

    def computeAlpha(self, row_vector, mean_vector, beta, label): #alpha=max(0, 1-label*example_tr*mean)*beta, constant
        return np.maximum(0, 1-misc.dot([label, misc.tr(row_vector), mean_vector]))*beta

    def computeMean(self, mean_vector, alpha, covariance_matrix, label, row_vector): #new_mean=mean+alpha*covariance*label*example, vector
        return mean_vector+misc.dot([alpha, covariance_matrix, label, row_vector])

    def computeCovariance(self, covariance_matrix, beta, row_vector): #new_covariance=covariance-beta*covariance*example*example_tr*covariance, matrix
        return covariance_matrix-misc.dot([beta, covariance_matrix, row_vector, misc.tr(row_vector), covariance_matrix])

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
