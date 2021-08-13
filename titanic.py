import csv
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from six import StringIO
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pydotplus
from IPython.display import Image 

columnNames = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Survived']
df = pd.read_csv("titanic.csv", names=columnNames).iloc[1:]
features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
X = df[features]
Y = df[columnNames[6]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 1)

clf = DecisionTreeClassifier(max_depth = 3)
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
# print("Accuracy(Nice): ",accuracy_score(Y_test, Y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file = dot_data, filled = True, rounded = True, special_characters = True, feature_names = features, class_names = ['0', '1'])
# print(dot_data.getvalue())
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('titanic.png')
Image(graph.create_png())