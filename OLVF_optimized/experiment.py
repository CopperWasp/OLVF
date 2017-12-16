import numpy as np
import preprocess
import random
import copy
import matplotlib.pyplot as plt
import time
import OLSF as sf
import OLVF as vf
import parameters as p


def streaming_experiment(data, dataset_name):
    plot_list=[
               ("OLvf", vf.olvf(data).fit, 1),
               ("OLsf", sf.olsf(data).fit, 1)
               ]
    
    x = list(range(len(data)))
    plotlist=[]

    for triple in plot_list:
        if triple[2]==1: plotlist.append((triple[1](), triple[0]))
        
    for i in range(len(plotlist)):
        plt.plot(x[1:][0::int(len(x)/10)], plotlist[i][0][1:][0::int(len(x)/10)], label=plotlist[i][1], marker='o', linestyle='--')
    plt.legend()
    plt.xlabel("Instance")
    plt.ylabel("Test Error Rate %")
    plt.title(dataset_name)
    figurename = './figures/'+'olvf_stream'+str("_")+time.strftime("%H%M%S")+'.png'
    plt.savefig(figurename)
    plt.grid()
    plt.show()
    X,y = getSampleData(data, p.stream_mode) # to be plotted
    feature_summary=[np.count_nonzero(row) for row in X]
    plotFeatures(feature_summary, "")
    return figurename


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


def getSampleData(data, mode='trapezoidal'): #get data, set X,y globally
    copydata= copy.deepcopy(data)
    random.shuffle(copydata)
    if mode=='trapezoidal': dataset=preprocess.removeDataTrapezoidal(copydata)
    if mode=='variable': dataset=preprocess.removeRandomData(copydata)
    else: dataset= copydata
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
    return X, y
    

def streamOverUCIDatasets():
    #streaming_experiment(preprocess.readUrlNormalized(1), "URL")
    #streaming_experiment(preprocess.readWbcNormalized(), "WBC")
    #figurename = streaming_experiment(preprocess.readGermanNormalized(), "German")
    #figurename = streaming_experiment(preprocess.readIonosphereNormalized(), "Ionosphere")
    figurename = streaming_experiment(preprocess.readMagicNormalized(), "Magic")
    #streaming_experiment(preprocess.readSpambaseNormalized(), "Spambase")
    #streaming_experiment(preprocess.readWdbcNormalized(), "WDBC")
    #streaming_experiment(preprocess.readWpbcNormalized(), "WPBC")
    #streaming_experiment(preprocess.readA8ANormalized(), "A8A")
    #figurename = streaming_experiment(preprocess.readSvmguide3Normalized(), "svmguide3")
    p.saveParameters(figurename)
 
    

start_time = time.time()
streamOverUCIDatasets()
print("--- %s seconds ---" % (time.time() - start_time))

#for i in range(0,20):
#    print(phi)
#    olsf_stream(preprocess.readMagicNormalized(), "Magic")
#    phi/=5
    
#for i in range(0,10):
#    streamOverUCIDatasets()
#    phi/=5

