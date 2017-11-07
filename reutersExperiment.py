from nltk.corpus import reuters 
import string
import nltk
import parameters
import classifier as c
import miscMethods as misc

class Reuters:
    def __init__(self):
        
        self.dataset=[]
        self.trainingDataset=[]
        self.testDataset=[]
        self.dataset_name='Reuters'
        self.extractDataAndRunExperiment() #run experiment by creating the object

    def tokenize(self, input_list):
    	stopwords = nltk.corpus.stopwords.words('english')
    	stopwords.extend(string.punctuation)
    	stopwords.append('')
    	tokenized_list = [token.lower().strip(string.punctuation) for token in input_list if token.lower().strip(string.punctuation) not in stopwords]
    	return tokenized_list

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
        
    def preprocessDocuments(self, spam_or_ham_list, class_label):
        for elem in spam_or_ham_list:
            self.dictionarize(self.tokenize(elem), class_label)

    def corpusMethods(self): #for reference
        documents = reuters.fileids() #List of documents
        print(str(len(documents)) + " documents");
        train_docs = list(filter(lambda doc: doc.startswith("train"), documents));
        print(str(len(train_docs)) + " total train documents");
        test_docs = list(filter(lambda doc: doc.startswith("test"), documents));
        print(str(len(test_docs)) + " total test documents");
        categories = reuters.categories(); #List of categories
        print(str(len(categories)) + " categories");
        category_docs = reuters.fileids("acq"); #Documents in a category
        document_id = category_docs[0] #Words for a document
        document_words = reuters.words(category_docs[0]);
        print(document_words);
        print(reuters.raw(document_id)); #raw document
    
    def extractDataAndRunExperiment(self):
        print("Extracting data from NLTK corpus.")
        category_pairs=parameters.reuters_category_pairs
        for pair in category_pairs: #do the experiment on selected pairs
            #Extract data for corresponding categories
            spam_category=pair[0]
            ham_category=pair[1]
            print("Experiment pair: "+str(spam_category)+", "+str(ham_category))
            self.dataset_name=str(spam_category)+", "+str(ham_category)
            spam_doc_ids=reuters.fileids(spam_category) #acquire documents using all docids for spam
            ham_doc_ids=reuters.fileids(ham_category)  #acquire documents using all docids for ham
            spamlist=[] #dataset for spam words
            hamlist=[] #dataset for ham words
            for docid in spam_doc_ids: 
                wordlist=reuters.words(docid) #get the words for document
                spamlist.append(wordlist) #append the documents words to spamlist as a row
            for docid in ham_doc_ids:
                wordlist=reuters.words(docid) #get the words for document
                hamlist.append(wordlist) #append the documents words to hamlist as a row
            #preprocess extracted data and merge them into self.dataset
            self.preprocessDocuments(spamlist, 1) #will tokenize (remove stopwords, punct.etc.), insert labels and dictionarize
            self.preprocessDocuments(hamlist, 0) #will tokenize (remove stopwords, punct.etc.), insert labels and dictionarize
            #call generate training test and experiment
            self.runExperiment()
            
    def runExperiment(self):
        total_error=0
        error_vector=[] #this will be plotted
        dataset=self.dataset
        feature_summary=[len(row) for row in dataset]
        for training_test_ratio in parameters.heldout:
            print("Heldout:"+str(training_test_ratio))
            for i in range(0, parameters.rounds):
                #randomly permute and split training test set
                training_set, test_set=misc.generateTrainingTest(dataset, training_test_ratio)
                my_classifier=c.classifier(training_set, test_set)
                classifier_summary=my_classifier.train()
                current_error=my_classifier.test()
                total_error+=current_error
                #print("Iteration:"+str(i)+", Iteration error: "+str(current_error))
            total_error/=parameters.rounds
            error_vector.append(total_error)
        misc.plotError(error_vector, self.dataset_name)
        misc.plotFeatures(feature_summary, self.dataset_name)
        misc.plotClassifierDimension(classifier_summary, self.dataset_name)        
