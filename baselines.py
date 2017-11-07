
import trapezoidalAlgorithm as trapez
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression

heldout = [0.95, 0.90, 0.75, 0.50, 0.01]
rounds = 20
digits = datasets.load_breast_cancer()
X, y = preprocessing.scale(digits.data), digits.target

classifiers = [
    ("OLSF", trapez.OLSF()),
    ("ASGD", SGDClassifier(average=True,max_iter=1)),
    ("Perceptron", Perceptron()),
    ("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge',
                                                         C=1.0, max_iter=1)),
    ("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge',
                                                          C=1.0, max_iter=1)),
    ("SAG", LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X.shape[0], max_iter=1))
]




class passiveAgressive:
    
    def __init_(self):
        self.classifier=PassiveAggressiveClassifier(loss='hinge', C=1.0, max_iter=1)
    
    def train(self):
        


xx = 1. - np.array(heldout)

for name, clf in classifiers:
    print("training %s" % name)
    rng = np.random.RandomState(42)
    yy = []
    for i in heldout:
        yy_ = []
        for r in range(rounds):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=i, random_state=rng)
            if name=="OLSF":
                for index, item in enumerate(y_train):
                    if item==0:
                        y_train[index] = -1
                for index, item in enumerate(y_test):
                    if item==0:
                        y_test[index] = -1               
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            yy_.append(1 - np.mean(y_pred == y_test))
        yy.append(np.mean(yy_))
    plt.plot(xx, yy, label=name)

plt.legend(loc="upper right")
plt.xlabel("Proportion train")
plt.ylabel("Test Error Rate")
plt.show()
