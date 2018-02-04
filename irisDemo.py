import numpy as np
import pandas as pd
from pandas import Series,DataFrame

# Plot imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

#%matplotlib inline

from sklearn import linear_model
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics

# Import the data
iris = load_iris()

# Grab features (X) and the Target (Y)
X = iris.data
print(type(X))

Y = iris.target
print(type(Y))

# Show the Built-in Data Description
print(iris.DESCR)

# Grab data
iris_data = DataFrame(X,columns=['Sepal Length','Sepal Width','Petal Length','Petal Width'])

# Grab Target
iris_target = DataFrame(Y,columns=['Species'])

def flower(num):
    ''' Takes in numerical class, returns flower name'''
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'Veriscolour'
    else:
        return 'Virginica'

# Apply
iris_target['Species'] = iris_target['Species'].apply(flower)

# Create a combined Iris DataSet
iris = pd.concat([iris_data,iris_target],axis=1)

# Preview all data
iris.head()

# First a pairplot of all the different features
#sns.pairplot(iris,hue='Species',size=2)

#sns.factorplot('Petal Length',data=iris,hue='Species',size=10)
#sns.factorplot('Petal Length',data=iris,size=10, kind='count', hue='Species')

#sns.lmplot(data=iris, x='Sepal Length', y='Sepal Width')
#plt.title('Bad regression')

#sns.lmplot(data=iris, x='Sepal Length', y='Sepal Width', hue='Species')

#sns.lmplot(data=iris, x='Sepal Length', y='Sepal Width', col='Species')

# Create a Logistic Regression Class object
logreg = LogisticRegression()

# Split the data into Trainging and Testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4,random_state=3)

# Train the model with the training set
logreg.fit(X_train, Y_train)



# Prediction from X_test
Y_pred = logreg.predict(X_test)

#Check accuracy
print(metrics.accuracy_score(Y_test,Y_pred))