epoch=1 #Number of iterations of training, epochs
seed=50#seed for random class
removing_percentage=0.5#random data remove, actually the percentage of features left, 1-removed
sparsity_regularizer=0.1 #lambda for sparsity strength
sparsity_B=0.7 #sparsity ratio
rounds=20 #raining-test rounds
meanFiller=0 #mean filler approach
heldout=[0.99, 0.90, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]

#comparison switches
olvf_random=0
olvf_random_sparse=0
olvf_trapez=0
olvf_trapez_sparse=0
olvf_trapez_shuffled=0
olsf=0

#List of tuples, spam label -1, ham label 1
enron_folders=[("./data/enron1/spam/", "./data/enron1/ham/"),
               ("./data/enron2/spam/", "./data/enron2/ham/"),
               ("./data/enron3/spam/", "./data/enron3/ham/"),
               ("./data/enron4/spam/", "./data/enron4/ham/"),
               ("./data/enron5/spam/", "./data/enron5/ham/"),
               ("./data/enron6/spam/", "./data/enron6/ham/")]

#category pairs to be experimented to classify, try to change similar categories for challenge
reuters_category_pairs=[("castor-oil", "coconut-oil"),
                        ("castor-oil", "cotton-oil"),
                        ("cotton-oil", "coconut-oil"),
                        ("soy-oil", "soy-meal"),
                        ("soybean", "soy-meal"),
                        ("wheat", "grain"),
                        ("interest", "income")]

