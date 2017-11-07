import numpy as np
import random
import parameters
import matplotlib.pyplot as plt
import time
import experiment
#Misc methods for dictionary-np array- list manipulations
random.seed(parameters.seed)

def findCommonKeys(classifier, row): #find the common keys of two dictionaries
    return (set(classifier.keys()) & set(row.keys()))
      
def findDifferentKeys(dict1, dict2):
    return (set(dict1.keys())-set(dict2.keys()))
         
def subsetDictionary(dictionary, intersection): #extract subset of key-val pairs if in 
    return dict((value, dictionary[value]) for value in intersection)

def dictToNumpyArray(dictionary):
    return np.array(list(dictionary.values()))

def dictToUnitNumpyMatrix(dictionary):
    matrix=np.identity(len(dictionary))
    values=list(dictionary.values())
    for i in range(0, len(dictionary)):
        matrix[i][i]=values[i]
    return matrix

def tr(matrix):
    return np.transpose(matrix)

def dot(listOfValues):
    returnValue=listOfValues[0]
    for i in range(1, len(listOfValues)):
        returnValue=np.dot(returnValue, listOfValues[i])
    return returnValue
        
def numpyArrayToDict(numpyArray, labels):
    return dict((labels[i], numpyArray[i]) for i in range(0, len(labels)))
    
def numpyMatrixToDict(numpyMatrix, labels):
    return dict((labels[i], numpyMatrix[i][i]) for i in range(0, len(labels)))

def generateTrainingTest(dataset, training_test_ratio):
        random.shuffle(dataset)
        training_length=int(len(dataset)*training_test_ratio)
        testDataset=dataset[:training_length]
        trainingDataset=dataset[training_length:]
        return trainingDataset, testDataset

def generateTrainingTestTrapezoidal(dataset, heldout): #dont shuffle for the order
        training_test_ratio=1-heldout
        training_length=int(len(dataset)*training_test_ratio)
        testDataset=dataset[training_length:]
        trainingDataset=dataset[:training_length]
        return trainingDataset, testDataset
    
#Plotting Methods
def plotError(error_vector, dataset_name):
    #xx = 1. - np.array(parameters.heldout) #heldouts
    xx= list(range(len(error_vector)-1))
    yy = (np.array(error_vector[1:])*100)
    plt.plot(xx, yy, label=dataset_name+", Lambda: "+str(experiment.lambda_error)+", Remaining features: "+str(parameters.removing_percentage))
    plt.legend(loc="upper right")
    plt.xlabel("Proportion Train")
    plt.ylabel("Test Error Rate %")
    plt.grid()
    plt.savefig('./figures/'+dataset_name+'_'+str(1-parameters.removing_percentage)+'_'+str(experiment.lambda_error)+'_'+time.strftime("%Y%m%d-%H%M%S")+'.png')
    plt.show()
    plt.clf()
    
def plotFeatures(feature_summary, dataset_name):
    xx = np.array(range(0, len(feature_summary)))
    yy = np.array(feature_summary)
    plt.plot(xx, yy, label=dataset_name, marker='o', linestyle='--')
    plt.legend(loc="upper right")
    plt.xlabel("Training examples")
    plt.ylabel("Number of features")
    plt.grid()
    plt.savefig('./figures/'+dataset_name+time.strftime("%Y%m%d-%H%M%S")+'features.png')
    plt.show()
    plt.clf()
    
def plotClassifierDimension(classifier_summary, dataset_name):
    xx = np.array(range(0, len(classifier_summary)))
    yy = np.array(classifier_summary)
    plt.plot(xx, yy, label=dataset_name+", B: "+str(parameters.sparsity_B)+", Lambda(sparsity): "+str(experiment.sparsity_regularizer))
    plt.legend(loc="upper right")
    plt.xlabel("Classifiers")
    plt.ylabel("Dimension")
    plt.grid()
    plt.savefig('./figures/'+dataset_name+'_'+str(1-parameters.removing_percentage)+'_'+str(experiment.lambda_error)+'_'+time.strftime("%Y%m%d-%H%M%S")+'sparsity.png')
    plt.show()
    plt.clf()
    
#print out variables for a method
def writeVariables():
    variables=("lambda_error: "+str(experiment.lambda_error)+"\n"+
              "removing_percentage: "+str(experiment.removing_percentage)+"\n"+
              "sparsity_switch: "+str(parameters.sparsity_switch)+"\n"+
              "sparsity_regularizer: "+str(parameters.sparsity_regularizer)+"\n"+
              "sparsity_B: "+str(parameters.sparsity_B)+"\n"+
              "rounds: "+str(parameters.rounds)+"\n"+
              "heldouts: "+str(parameters.heldout)+"\n")
    f = open('./figures/parameters.txt', 'w')
    f.write(variables)  # python will convert \n to os.linesep
    f.close()  # you can omit in most cases as the destructor will call it
