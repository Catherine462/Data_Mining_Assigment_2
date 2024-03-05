#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv']

for ds in dataSets:

    X = []
    Y = []
    df = pd.read_csv(ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:,1:] #creating a training matrix without the id (NumPy library)
    for row in data_training:
        features = []
       
        if row[0] == 'Yes':   
            features.append(1)
        else:
            features.append(2)

        if row[1] == 'Single':  
            features.append(1)
            features.extend([0, 0])
        elif row[1] == 'Divorced':
            features.extend([0, 1, 0])
        else:  # Married
            features.extend([0, 0, 1])


    #     # Converting 'Taxable Income' to float and adding to features
        taxable_income = float(row[2][:-1])  # Remove 'k' and convert to float
        features.append(taxable_income)

    #     # Adding features to X
        X.append(features)

    #     # Transforming the original training classes to numbers and adding them to the vector Y
        if row[3] == 'Yes':
            Y.append(1)
        else:
            Y.append(0)

print(X,Y)
accuracies = []
#loop your training and test tasks 10 times here
for i in range (10):

  #fitting the decision tree to the data by using Gini index and no max_depth
  clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
  clf = clf.fit(X, Y)
  
  #plotting the decision tree
  tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
  plt.show()
  data_test = pd.read_csv('cheat_test.csv', sep=',', header=0)
  data_test = np.array(data_test)
  
  correct_predictions = 0 
  for i, data in enumerate(data_test):
       
        features_test = []

        if data[1] == 'Yes':
            features_test.append(1)
        else:
            features_test.append(2)

        if data[2] == 'Single':
            features_test.append(1)
            features_test.extend([0, 0])
        elif data[2] == 'Divorced':
            features_test.extend([0, 1, 0])
        else:  
            features_test.extend([0, 0, 1])

        taxable_income_test = float(data[3][:-1])  
        features_test.append(taxable_income_test)

        class_predicted = clf.predict([features_test])[0]
        

        true_label = 1 if true_label == "Yes" else 0
        if class_predicted == true_label:
            correct_predictions += 1 
  
  accuracy = correct_predictions / len(data_test)
  accuracies.append(accuracy)

# Calculate average accuracy over 10 runs
avg_accuracy = sum(accuracies) / len(accuracies)
print("Average accuracy:", avg_accuracy)


