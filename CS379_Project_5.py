#Importing required libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


###################################################################################################################################
#Iris datset is from the UCI Machine learning repository: https://archive.ics.uci.edu/ml/datasets/iris                            #
#The Hackerearch decision tree tutorial code was referenced and used for data manipulation                                        #
#The sklearn documentation was also heavily referenced for info on splitter information                                                                           #
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier  #
###################################################################################################################################

def main():

    #Loading the iris data
    data = load_iris() 
    print('Classes to predict: ', data.target_names)


    X = data.data

    y = data.target

    print('Number of examples in the data:', X.shape[0])
    X[:4]



    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 47, test_size = 0.25)

        #####ASSIGNMENT INFO HERE########
    #When creating the decision tree the sklearn DecisionTreeClassifier supports mutiple splitting parameters
    #"gini," "Entropy 
    #It also has support for mutiple splitters best and random best being the default 
    #We can run the algortm across the dataset mutiple times with diffetent splitting and criterion parameters set
    #to show different accuarcys within the model
  

    from sklearn.tree import DecisionTreeClassifier
    ebclf = DecisionTreeClassifier(criterion = 'entropy', splitter = "best") #this would be default parameters within sklearn
    ebclf.fit(X_train, y_train)

    grclf = DecisionTreeClassifier(criterion = 'gini', splitter = "random") #this is another decistion tree with the alternate methods
    grclf.fit(X_train, y_train)


    #Predicting labels on the test set.
    y_pred =  ebclf.predict(X_test)
    y_pred2 = grclf.predict(X_test)


    #Importing the accuracy metric from sklearn.metrics library

    from sklearn.metrics import accuracy_score
    print('Accuracy Score on train data with best splitting and random criterion: ', accuracy_score(y_true=y_train, y_pred = ebclf.predict(X_train)))
    print('Accuracy Score on test data with best splitting and random criterion: ', accuracy_score(y_true=y_test, y_pred=y_pred))

    print('Accuracy Score on train data with random splitting and gini criterion: ', accuracy_score(y_true = y_train, y_pred = grclf.predict(X_train)))
    print('Accuracy Score on test data with random splitting and gini criterion: ', accuracy_score(y_true = y_test, y_pred = y_pred2))

    #as we can see, swapping to a random splitting method actually lowers the accuarcy score 
    #of the algorithm opposed to the learned spluitting algorithm from the test data


    return

main()