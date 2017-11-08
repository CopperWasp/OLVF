# Enron, read raw texts, tokenize, remove stopwords and put in a dictionary, every word is a feature
import nltk.corpus
import nltk.tokenize.punkt
import string
import os
import random
import parameters


class Data:
    def __init__(self, spam_folder, ham_folder):
        self.dataset = []
        self.trainingDataset = []
        self.testDataset = []
        self.preprocess_documents(spam_folder, 1)
        self.preprocess_documents(ham_folder, 0)
        
    def tokenize(self, input_string):
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.extend(string.punctuation)
        stopwords.append('')
        tokenizer = nltk.tokenize.TreebankWordTokenizer()
        tokenized_list = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(input_string) if token.lower().strip(string.punctuation) not in stopwords]
        return tokenized_list

    def read_document(self, path):
        with open(path, 'r', encoding='utf8', errors='ignore') as content_file:
            content = content_file.read()
            return content

    def convert_to_dict(self, tokenized_list, class_label):
        row = {}
        for record in tokenized_list:
            if record in row:
                row[record] += 1
            else:
                row[record] = 1
        if class_label == 1:
            row["class_label"] = 1
        else:
            row["class_label"] = -1
        self.dataset.append(row)

    def preprocess_documents(self, folder_name, class_label):
        for file in os.listdir(folder_name):
            self.convert_to_dict(self.tokenize(self.read_document(folder_name + file)), class_label)

    def generate_training_test(self):
        random.seed(parameters.seed)
        random.shuffle(self.dataset)
        training_length=int(len(self.dataset)*self.training_test_ratio)
        self.testDataset=self.dataset[:training_length]
        self.trainingDataset=self.dataset[training_length:]


