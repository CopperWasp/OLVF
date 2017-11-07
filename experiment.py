#define streaming experiments
import parameters
import classifier as c
import preprocess
import miscMethods as misc
import preprocess2
import random
import copy
import olsf
import numpy as np
import matplotlib.pyplot as plt
import time


#Non Changable Parameters
lambda_error=0.5#weight of hinge loss, 1-weight of regularization
lambda_confidence=1-lambda_error
parameter_r_error=1/2*lambda_error
parameter_r_confidence=1/2*lambda_confidence
#sparsity
random.seed(parameters.seed)
param_lambda=30
C=1
B=1

        
def parameterSearch():
    global B
    global lambda_confidence
    global parameter_r_error
    global parameter_r_confidence
    global lambda_error
    lambda_error=0.99
    for i in range (0, 10):
        B=0
        for j in range (0, 10):
            compareClassifiersOverUCIDatasets()
            B+=0.1
            lambda_confidence=1-lambda_error
            parameter_r_error=1/2*lambda_error
            parameter_r_confidence=1/2*lambda_confidence
        lambda_error-=0.1

def compareClassifiersOverUCIDatasets():
    compare_classifiers(preprocess2.readGermanNormalized(), "German")
    #compare_classifiers(preprocess2.readIonosphereNormalized(), "Ionosphere")
    #compare_classifiers(preprocess2.readMagicNormalized(), "Magic")
    #compare_classifiers(preprocess2.readSpambaseNormalized(), "Spambase")
    #compare_classifiers(preprocess2.readWdbcNormalized(), "WDBC")
    #compare_classifiers(preprocess2.readWpbcNormalized(), "WPBC")
    #compare_classifiers(preprocess2.readA8ANormalized(), "A8A")
    #compare_classifiers(preprocess2.readSvmguide3Normalized(), "svmguide3")



def compare_classifiers(readed_dataset, dataset_name):
    plot_list=[("OLVF_random", variableFeatureExperiment, parameters.olvf_random),
               ("OLVF_trapez", trapezoidalExperiment, parameters.olvf_trapez),
               ("OLVF_trapez_shuffled_sparse", trapezoidalExperimentShuffledSparse, parameters.olvf_trapez_shuffled),
               ("OLSF", trapezoidalExperimentOLSF, parameters.olsf),
               ("OLVF_random_sparse", variableFeatureExperiment, parameters.olvf_random_sparse),
               ("OLVF_trapez_sparse", trapezoidalExperimentSparse, parameters.olvf_trapez_sparse)]
    x = list(range(len(readed_dataset)))
    plotlist=[]
    for triple in plot_list:
        if triple[2]==1:
            plotlist.append((triple[1](readed_dataset, dataset_name), triple[0]))
    for i in range(len(plotlist)):
        plt.plot(x[2:], plotlist[i][0][2:], label=plotlist[i][1])  
    plt.legend()
    plt.xlabel("Instance")
    plt.ylabel("Test Error Rate %")
    plt.title(dataset_name)
    plt.savefig('./figures/'+'compare_'+str(lambda_error)+str("_")+str(B)+str("_")+time.strftime("%H%M%S")+'.png')
    plt.show()

def variableFeatureExperiment(input_dataset, dataset_name, mode="OLVF_random_sparse"):
    print("Variable feature experiment with OLVF: "+str(dataset_name))
    error_vector=np.zeros(len(input_dataset))
    feature_summary=[len(row) for row in preprocess2.removeRandomData(copy.deepcopy(input_dataset))]
    current_dataset= preprocess2.removeRandomData(copy.deepcopy(input_dataset))
    for i in range(parameters.rounds):
        print("Round: "+str(i))
        random.seed(parameters.seed)
        random.shuffle(current_dataset)
        current_classifier=c.classifier(current_dataset, [])
        classifier_summary, stream_error= current_classifier.train()
        error_vector=np.add(error_vector, stream_error)
    average_error_vector= np.divide(error_vector, parameters.rounds)
    #misc.plotError(average_error_vector, dataset_name)
    #misc.plotFeatures(feature_summary, dataset_name)
    #misc.plotClassifierDimension(classifier_summary, dataset_name)
    print(current_classifier.mean_dict)
    return average_error_vector

def variableFeatureExperimentSparse(input_dataset, dataset_name):
    print("Variable feature experiment with OLVF: "+str(dataset_name))
    error_vector=np.zeros(len(input_dataset))
    feature_summary=[len(row) for row in preprocess2.removeRandomData(copy.deepcopy(input_dataset))]
    current_dataset= preprocess2.removeRandomData(copy.deepcopy(input_dataset))
    for i in range(parameters.rounds):
        print("Round: "+str(i))
        random.seed(parameters.seed)
        random.shuffle(current_dataset)
        current_classifier=c.classifier(current_dataset, [], 1)
        classifier_summary, stream_error= current_classifier.train()
        error_vector=np.add(error_vector, stream_error)
    average_error_vector= np.divide(error_vector, parameters.rounds)
    #misc.plotError(average_error_vector, dataset_name)
    #misc.plotFeatures(feature_summary, dataset_name)
    #misc.plotClassifierDimension(classifier_summary, dataset_name)
    print(current_classifier.mean_dict)
    return average_error_vector
        
def trapezoidalExperiment(input_dataset, dataset_name):
      print("Trapezoidal experiment with OLVF: "+str(dataset_name)) 
      error_vector=np.zeros(len(input_dataset))
      feature_summary=[len(row) for row in preprocess2.removeDataTrapezoidal(copy.deepcopy(input_dataset))]
      for i in range(parameters.rounds):
          print("Round: "+str(i))
          random.seed(parameters.seed)
          random.shuffle(input_dataset)
          current_dataset= preprocess2.removeDataTrapezoidal(copy.deepcopy(input_dataset))
          current_classifier= c.classifier(current_dataset, [])
          classifier_summary, stream_error= current_classifier.train()
          error_vector=np.add(error_vector, stream_error)
      average_error_vector= np.divide(error_vector, parameters.rounds)
      #misc.plotError(average_error_vector, dataset_name)
      #misc.plotFeatures(feature_summary, dataset_name)
      #misc.plotClassifierDimension(classifier_summary, dataset_name)
      print(current_classifier.mean_dict)
      return average_error_vector

def trapezoidalExperimentSparse(input_dataset, dataset_name):
      print("Trapezoidal experiment with OLVF: "+str(dataset_name)) 
      error_vector=np.zeros(len(input_dataset))
      feature_summary=[len(row) for row in preprocess2.removeDataTrapezoidal(copy.deepcopy(input_dataset))]
      for i in range(parameters.rounds):
          print("Round: "+str(i))
          random.seed(parameters.seed)
          random.shuffle(input_dataset)
          current_dataset= preprocess2.removeDataTrapezoidal(copy.deepcopy(input_dataset))
          current_classifier= c.classifier(current_dataset, [], 1)
          classifier_summary, stream_error= current_classifier.train()
          error_vector=np.add(error_vector, stream_error)
      average_error_vector= np.divide(error_vector, parameters.rounds)
      #misc.plotError(average_error_vector, dataset_name)
      #misc.plotFeatures(feature_summary, dataset_name)
      #misc.plotClassifierDimension(classifier_summary, dataset_name)
      print(current_classifier.mean_dict)
      return average_error_vector

def trapezoidalExperimentShuffled(input_dataset, dataset_name):
      print("Trapezoidal experiment shuffled with OLVF: "+str(dataset_name)) 
      error_vector=np.zeros(len(input_dataset))
      feature_summary=[len(row) for row in preprocess2.removeDataTrapezoidal(copy.deepcopy(input_dataset))]
      for i in range(parameters.rounds):
          print("Round: "+str(i))
          random.seed(parameters.seed)
          random.shuffle(input_dataset)
          current_dataset= preprocess2.removeDataTrapezoidal(copy.deepcopy(input_dataset))
          random.shuffle(current_dataset)
          current_classifier= c.classifier(current_dataset, [])
          classifier_summary, stream_error= current_classifier.train()
          error_vector=np.add(error_vector, stream_error)
      average_error_vector= np.divide(error_vector, parameters.rounds)
      #misc.plotError(average_error_vector, dataset_name)
      #misc.plotFeatures(feature_summary, dataset_name)
      #misc.plotClassifierDimension(classifier_summary, dataset_name)
      print(current_classifier.mean_dict)
      return average_error_vector
  
def trapezoidalExperimentShuffledSparse(input_dataset, dataset_name):
      print("Trapezoidal experiment shuffled with OLVF: "+str(dataset_name)) 
      error_vector=np.zeros(len(input_dataset))
      feature_summary=[len(row) for row in preprocess2.removeDataTrapezoidal(copy.deepcopy(input_dataset))]
      for i in range(parameters.rounds):
          print("Round: "+str(i))
          random.seed(parameters.seed)
          random.shuffle(input_dataset)
          current_dataset= preprocess2.removeDataTrapezoidal(copy.deepcopy(input_dataset))
          random.shuffle(current_dataset)
          current_classifier= c.classifier(current_dataset, [], 1)
          classifier_summary, stream_error= current_classifier.train()
          error_vector=np.add(error_vector, stream_error)
      average_error_vector= np.divide(error_vector, parameters.rounds)
      #misc.plotError(average_error_vector, dataset_name)
      #misc.plotFeatures(feature_summary, dataset_name)
      #misc.plotClassifierDimension(classifier_summary, dataset_name)
      print(current_classifier.mean_dict)
      return average_error_vector

def trapezoidalExperimentOLSF(input_dataset, dataset_name):
      print("Trapezoidal experiment with OLSF: "+str(dataset_name)) 
      error_vector=np.zeros(len(input_dataset))
      feature_summary=[len(row) for row in preprocess2.removeDataTrapezoidal(copy.deepcopy(input_dataset))]
      for i in range(parameters.rounds):
          print("Round: "+str(i))
          random.seed(parameters.seed)
          random.shuffle(input_dataset)
          current_dataset= preprocess2.removeDataTrapezoidal(copy.deepcopy(input_dataset))
          current_classifier= olsf.classifier(current_dataset, [])
          classifier_summary, stream_error= current_classifier.train()
          error_vector=np.add(error_vector, stream_error)
      average_error_vector= np.divide(error_vector, parameters.rounds)
      #misc.plotError(average_error_vector, dataset_name)
      #misc.plotFeatures(feature_summary, dataset_name)
      #misc.plotClassifierDimension(classifier_summary, dataset_name)
      print(current_classifier.weight_dict)
      return average_error_vector

def Enron_experiment():
    print("Enron Experiment")
    for index, folder_tuple in enumerate(parameters.enron_folders): #iterate over all Enron folders
        dataset=(preprocess.data(folder_tuple[0], folder_tuple[1])).dataset
        error_vector=np.zeros(len(dataset))
        feature_summary=[len(i) for i in dataset] #extract the feature dimension vector to plot
        dataset_name="Enron"+str(index+1)
        dataset= olvf_closed_form.preprocessData(dataset)
        my_classifier=olvf_closed_form.olvf("stream")
        classifier_summary, stream_error =my_classifier.train()
        error_vector= np.add(error_vector, stream_error)   
        misc.plotError(error_vector, dataset_name)
        misc.plotFeatures(feature_summary, dataset_name)
        #misc.plotClassifierDimension(classifier_summary, dataset_name)
        #print(my_classifier.mean_dict)
        #return average_error_vector
        
def url_experiment_olsf():
    print("URL Dataset Experiment")
    dataset_name= "URL"
    error_vector=[]
    weights= {}
    for day in range(0, 121):
        print("File: Day"+str(day))
        dataset=preprocess2.readUrlNormalized(day)
        my_classifier=olsf.classifier(dataset, [])
        my_classifier.weight_dict= weights
        classifier_summary, stream_error =my_classifier.train()
        weights=my_classifier.weight_dict
        error_vector= np.append(error_vector,stream_error)

    misc.plotError(error_vector[0::2500], dataset_name)           
    return error_vector

def url_experiment_olvf():
    print("URL Dataset Experiment")
    dataset_name= "URL"
    error_vector=[]
    weights= {}
    for day in range(0, 121):
        print("File: Day"+str(day))
        dataset=preprocess2.readUrlNormalized(day)
        my_classifier=c.classifier(dataset, [], 1)
        my_classifier.weight_dict= weights
        classifier_summary, stream_error =my_classifier.train()
        weights=my_classifier.mean_dict
        error_vector= np.append(error_vector,stream_error)

    misc.plotError(error_vector[0::2500], dataset_name)           
    return error_vector

def TREC_experiment():
    print("TREC Experiment")
    dataset=(preprocess.data(parameters.spam_email[0], parameters.ham_email[0])).dataset
    error_vector=np.zeros(len(dataset))
    feature_summary=[len(i) for i in dataset] #extract the feature dimension vector to plot
    dataset_name="TREC"
    for i in range(0, parameters.rounds):
        print("Round: "+str(i))
        random.shuffle(dataset) #shuffle and split dataset
        my_classifier=c.classifier(dataset, [], 1)
        classifier_summary, stream_error =my_classifier.train()
        error_vector= np.add(error_vector, stream_error)
        average_error_vector= np.divide(error_vector, parameters.rounds)    
    misc.plotError(average_error_vector[1:], dataset_name)
    misc.plotFeatures(feature_summary, dataset_name)
    misc.plotClassifierDimension(classifier_summary, dataset_name)
        #print(my_classifier.mean_dict)
        #return average_error_vector
        

        
    
