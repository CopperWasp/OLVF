import experiment as e
import miscMethods as m
import reutersExperiment as r
import parameters as p
import preprocess2
#Experiment scripts over parameters
def run():

    #e.TREC_experiment()
    #e.url_experiment_olvf()
    #r.Reuters()
    e.compareClassifiersOverUCIDatasets()
   # e.parameterSearch()

    
run()
