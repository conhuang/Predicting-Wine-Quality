import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

#obtaining csv dataset from online source (wine metrics)
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=";")

# setting data up for supervised learning
# wine metrics such as pH, etc. and y = quality (desired)
y = data.quality
X = data.drop('quality', axis=1)

# creating the test and training sets from the data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=123, stratify=y)

# saving the mean and standard deviations
# in other words fitting the transformer on the dataset (fitting the scaling object)
scalar = preprocessing.StandardScaler().fit(X_train)

# transforming the test and training data to mean = 0 standard deviation = 1 (standardizing)
X_train_scaled = scalar.transform(X_train)
X_test_scaled = scalar.transform(X_test)

# creating the pipeline and identifying the parameters that must be tuned (hyperparams)
pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

# doing the cross-validation through the pipeline, and preprocessing the training sets in
# the pipeline also training using random forest
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train, y_train)

# check the model is refit with the best set of hyperparams
print(clf.refit)

y_pred = clf.predict(X_test)

print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

joblib.dump(clf, 'wine_selector.pkl')

# Takeaways:
# Try other regression model families (e.g. regularized regression, boosted trees, etc.).
# Collect more data if it's cheap to do so.
# Engineer smarter features after spending more time on exploratory analysis.
# Speak to a domain expert to get more context (...this is a good excuse to go wine tasting!).
