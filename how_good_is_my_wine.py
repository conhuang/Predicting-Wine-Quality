import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
import seaborn as sns
import matplotlib.pyplot as plt

clf2 = joblib.load('wine_selector.pkl')

dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
data = pd.read_csv(dataset_url, sep=";")

y_test = data["quality"]
x_test = data.drop('quality', axis=1)

print(x_test)

y_pred = clf2.predict(x_test)

print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

plt.hist((y_pred - y_test), 15)

plt.show()
