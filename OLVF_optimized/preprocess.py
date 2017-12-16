import csv
import random
import parameters
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file
import numpy as np

spambase='./otherDatasets/spambase.csv'
isolet='./otherDatasets/isolet.csv'
german='./otherDatasets/german.csv'
ionosphere='./otherDatasets/ionosphere.csv'
magic04='./otherDatasets/magic04.csv'
wdbc='./otherDatasets/wdbc.csv'
wpbc='./otherDatasets/wpbc.csv'
wbc='./otherDatasets/wbc.csv'
a8a_training='./otherDatasets/a8a_train.txt'
svmguide3='./otherDatasets/svmguide3.txt'
url= './otherDatasets/url_svmlight/'

def readUrlNormalized(file_number):
    filename=url+"Day"+str(file_number)+".svm"
    dataset=[]
    with open(filename) as f:
        for line in f:
            line_dict={}
            x=(line.rstrip()[3:]).split()
            y=int(line[:3])
            for elem in x:
                elem_list=elem.split(":")
                line_dict[int(elem_list[0])]=float(elem_list[1])
            line_dict['class_label']=int(y)
            dataset.append(line_dict)
    return dataset

def readSvmguide3Normalized(): #already normalized
    return_dataset=[]
    X, y = load_svmlight_file(svmguide3)
    numpy_dataset=X.toarray()
    #numpy_dataset=dataset.astype(np.float)
    numpy_dataset=preprocessing.scale(numpy_dataset)
    i=0
    for row in numpy_dataset:
        mydict={v: k for v, k in enumerate(row)}
        mydict['class_label'] = y[i]
        return_dataset.append(mydict)
        i+=1
    return return_dataset

def readA8ANormalized(): #already normalized
    return_dataset=[]
    X, y = load_svmlight_file(a8a_training)
    numpy_dataset=X.toarray()
    #numpy_dataset=dataset.astype(np.float)
    numpy_dataset=preprocessing.scale(numpy_dataset)
    i=0
    for row in numpy_dataset:
        mydict={v: k for v, k in enumerate(row)}
        mydict['class_label'] = y[i]
        return_dataset.append(mydict)
        i+=1
    return return_dataset

def readSpambaseNormalized():
    dataset=[]
    return_dataset=[]
    with open(spambase) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset=np.array(dataset)
    numpy_dataset=dataset.astype(np.float)
    numpy_dataset[:,:-1]=preprocessing.scale(numpy_dataset[:,:-1])
    for row in numpy_dataset:
        mydict={v: k for v, k in enumerate(row)}
        if mydict[57] == 0:
            mydict[57]=-1
        mydict['class_label'] = mydict.pop(57)
        return_dataset.append(mydict)
    return return_dataset
            
def readGermanNormalized():
    dataset=[]
    return_dataset=[]
    with open(german) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row=row[0].split(" ")
            row = list(filter(None, row)) # fastest
            dataset.append(row)
    dataset=np.array(dataset)
    numpy_dataset=dataset.astype(np.float)
    numpy_dataset[:,:-1]=preprocessing.scale(numpy_dataset[:,:-1])
    for row in numpy_dataset:
        mydict={v: k for v, k in enumerate(row)}
        if mydict[24] == 2:
            mydict[24]=-1
        mydict['class_label'] = mydict.pop(24)
        return_dataset.append(mydict)
    return return_dataset

def readIonosphereNormalized():
    dataset=[]
    return_dataset=[]
    with open(ionosphere) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset=np.array(dataset)
    for row in dataset:
        if row[34]== 'b':
            row[34]=-1
        else:
            row[34]=1
    numpy_dataset=dataset.astype(np.float)
    numpy_dataset[:,:-1]=preprocessing.scale(numpy_dataset[:,:-1])
    for row in numpy_dataset:
        mydict={v: k for v, k in enumerate(row)}
        mydict['class_label'] = mydict.pop(34)
        return_dataset.append(mydict)
    return return_dataset

def readMagicNormalized():
    dataset=[]
    return_dataset=[]
    with open(magic04) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset=np.array(dataset)
    for row in dataset:
        if row[10]== 'h':
            row[10]=1
        else:
            row[10]=-1
    numpy_dataset=dataset.astype(np.float)
    numpy_dataset[:,:-1]=preprocessing.scale(numpy_dataset[:,:-1])
    for row in numpy_dataset:
        mydict={v: k for v, k in enumerate(row)}
        mydict['class_label'] = mydict.pop(10)
        return_dataset.append(mydict)
    return return_dataset

def readWdbcNormalized():
    dataset=[]
    return_dataset=[]
    with open(wdbc) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset=np.array(dataset)
    dataset=dataset[:,1:]
    numpy_dataset=dataset.astype(np.float)
    numpy_dataset[:,1:]=preprocessing.scale(numpy_dataset[:,1:])
    for row in numpy_dataset:
        mydict={v: k for v, k in enumerate(row)}
        mydict['class_label'] = mydict.pop(0)
        return_dataset.append(mydict)
    return return_dataset

def readWpbcNormalized():
    dataset=[]
    return_dataset=[]
    with open(wpbc) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset=np.array(dataset)
    dataset=dataset[:,1:]
    numpy_dataset=dataset.astype(np.float)
    numpy_dataset[:,1:]=preprocessing.scale(numpy_dataset[:,1:])
    for row in numpy_dataset:
        mydict={v: k for v, k in enumerate(row)}
        mydict['class_label'] = mydict.pop(0)
        return_dataset.append(mydict)
    return return_dataset

def readWbcNormalized():
    dataset=[]
    return_dataset=[]
    with open(wbc) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset=np.array(dataset)
    dataset=dataset[:,1:]
    numpy_dataset=dataset.astype(np.float)
    numpy_dataset[:,:-1]=preprocessing.scale(numpy_dataset[:,:-1])
    for row in numpy_dataset:
        mydict={v: k for v, k in enumerate(row)}
        label = mydict.pop(len(row)-1)
        if int(label)==2: 
            mydict['class_label']=-1
        elif int(label)==4: 
            mydict['class_label']=1
        return_dataset.append(mydict)
    return return_dataset

def removeRandomData(dataset): #remove based on removing percentage
    for row in dataset:
        for i in list(row):
            if random.random()>0.5:
                if i!='class_label':
                    row.pop(i)
    return dataset

def removeDataTrapezoidal(original_dataset): #trapezoidal
    dataset=original_dataset[:]
    features=len(dataset[0])
    rows=len(dataset)
    for i in range(0, len(dataset)):
        multiplier=int(i/(rows/10))+1
        increment=int(features/10)
        features_left=multiplier*increment
        if(i==len(dataset)-1):
            features_left=features-2
        for key, value in dataset[i].copy().items():
            if key!='class_label' and key>features_left:
                dataset[i].pop(key)
    return dataset

def removeDataTrapezoidalShuffled(original_dataset): #trapezoidal
    dataset=original_dataset[:]
    #random.shuffle(dataset)
    features=len(dataset[0])
    rows=len(dataset)
    for i in range(0, len(dataset)):
        multiplier=int(i/(rows/10))+1
        increment=int(features/10)
        features_left=multiplier*increment
        if(i==len(dataset)-1):
            features_left=features-2
        for key, value in dataset[i].copy().items():
            if key!='class_label' and key>features_left:
                dataset[i].pop(key)
    random.shuffle(dataset)
    return dataset
