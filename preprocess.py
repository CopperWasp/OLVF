#read raw texts, tokenize, remove stopwords and put in a dictionary, every word is a feature
import nltk.corpus
import nltk.tokenize.punkt
import string
import os
import random
import parameters

class data:
    def __init__(self, spam_folder, ham_folder):
        self.dataset=[]
        self.trainingDataset=[]
        self.testDataset=[]
        self.preprocessDocuments(spam_folder, 1)
        self.preprocessDocuments(ham_folder, 0)
        
    def tokenize(self, input_string):
    	stopwords = nltk.corpus.stopwords.words('english')
    	stopwords.extend(string.punctuation)
    	stopwords.append('')
    	tokenizer = nltk.tokenize.TreebankWordTokenizer()
    	tokenized_list = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(input_string) if token.lower().strip(string.punctuation) not in stopwords]
    	return tokenized_list

    def readDocument(self, path):
    	with open(path, 'r', encoding='utf8', errors='ignore') as content_file:
        	content = content_file.read()
        	return content

    def dictionarize(self, tokenized_list, class_label):
        row={}
        for record in tokenized_list:
            if record in row:
                row[record]+=1
            else:
                row[record]=1
        if class_label==1:
        	row["class_label"]=1
        else:
        	row["class_label"]=-1
        self.dataset.append(row)

    def preprocessDocuments(self, folder_name, class_label):
        for file in os.listdir(folder_name):
            self.dictionarize(self.tokenize(self.readDocument(folder_name+file)), class_label)

    def generateTrainingTest(self):
        random.seed(parameters.seed)
        random.shuffle(self.dataset)
        training_length=int(len(self.dataset)*self.training_test_ratio)
        self.testDataset=self.dataset[:training_length]
        self.trainingDataset=self.dataset[training_length:]


