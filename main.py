from Decision_tree import DecisionTree
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# Prepare the Dataset
df = pd.read_csv("WINE.csv")
print(df.head())

# Split into train and test
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
