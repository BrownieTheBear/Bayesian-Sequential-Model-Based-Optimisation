#Sequential Model-Based Hyperparameter Optimisation using Bayesian Probabilistic Reasoning

#This Jupyter Notebook contains 3 experiments with the aim to visualise the hyperparameter selection process for a number of key algorithms used for classification problems within the field of applied machine learning.

#Results: Experimental results show that Bayesian optimization algorithm outperforms other global optimization algorithms. The results also show that there are significant computational savings with respect to the algorithm runtime from using probabilistic approach.

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import backend as K
Using TensorFlow backend.
from keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import datasets, svm
import matplotlib.pyplot as plt
#Experiment 1: For this experiment, we will first import and frame the MNIST dataset from the internal sklearn library.

# Load the digit data
digits = datasets.load_digits()
# View the features of the first observation
digits.data[0:1]
array([[ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.,  0.,  0., 13., 15., 10.,
        15.,  5.,  0.,  0.,  3., 15.,  2.,  0., 11.,  8.,  0.,  0.,  4.,
        12.,  0.,  0.,  8.,  8.,  0.,  0.,  5.,  8.,  0.,  0.,  9.,  8.,
         0.,  0.,  4., 11.,  0.,  1., 12.,  7.,  0.,  0.,  2., 14.,  5.,
        10., 12.,  0.,  0.,  0.,  0.,  6., 13., 10.,  0.,  0.,  0.]])
# Create dataset 1
data1_features = digits.data[:1000]
data1_target = digits.target[:1000]
# Before looking for which combination of parameter values produces the most accurate model, we must specify the different candidate values we want to try. In the code below we have a number of candidate parameter values, including four different values for C (1, 10, 100, 1000), two values for gamma (0.001, 0.0001), and two kernels (linear, rbf). The grid search will try all combinations of parameter values and select the set of parameters which provides the most accurate model.
#output
#parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]
Now we are ready to conduct the grid search using scikit-learn’s GridSearchCV which stands for grid search cross validation.

# Create a classifier object with the classifier and parameter candidates
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)

# Train the classifier on data1's feature and target data
clf.fit(data1_features, data1_target)
GridSearchCV(estimator=SVC(), n_jobs=-1,
             param_grid=[{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                         {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001],
                          'kernel': ['rbf']}])
import plotly.express as px
import pandas as pd
# Now we can visualise the GSCV results using the external Plotly API

# Test scores for object to be passed to plotly module
scores_mean = clf.cv_results_['mean_score_time']
fit_mean = clf.cv_results_['mean_fit_time']
test_score_mean = clf.cv_results_['mean_test_score']
color = clf.cv_results_['param_C']
title = 'Grid SearchCV for SVC Regressor'

fig = px.scatter_3d(x=scores_mean,y=fit_mean,
                    z=test_score_mean,color=color,
                    labels={'x' : 'Mean Score',
                            'y' : 'Mean Fit Time',
                            'z' : 'Mean Test Score',
                           'color' : 'Regularization (C) Parameter'},title=title,
                    height=1000,width=1000,
                    )


#manually change the labels
fig.show()
# Experiment 2: This experiment will investigate the tuning process using a Random Forest Classifier algorithm coupled with Random Search as opposed to Grid Search.

#We will be using the built-in temperature and rainfall dataset, a small amount of data pre-processing, then feature scaling prior to the tuning phase.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
rf = RandomForestClassifier(random_state=50)

print(rf.get_params())
{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': 50, 'verbose': 0, 'warm_start': False}
#We will analyse and compare the following hyperparameters as they are the most impactful when tuning the RFC algorithm

#n_estimators = number of trees in the forest

#max_features = max number of features considered for splitting a node

#max_depth = max number of levels in each decision tree

#min_samples_split = min number of data points placed in a node before the node is split

#min_samples_leaf = min number of data points allowed in a leaf node

#bootstrap = method for sampling data points (with or without replacement)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
{'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000], 'max_features': ['auto', 'sqrt'], 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'bootstrap': [True, False]}
#Now we will import and pre-process the tempertaure dataset, below is a cross section of the output

data = pd.read_csv('C:/Users/onemi/Downloads/temps_extended.csv')

# one-hot encoding
features = pd.get_dummies(data)

print(features)
      year  month  day   ws_1  prcp_1  snwd_1  temp_2  temp_1  average  \
0     2011      1    1   4.92    0.00       0      36      37     45.6
1     2011      1    2   5.37    0.00       0      37      40     45.7
2     2011      1    3   6.26    0.00       0      40      39     45.8
3     2011      1    4   5.59    0.00       0      39      42     45.9
4     2011      1    5   3.80    0.03       0      42      38     46.0
...    ...    ...  ...    ...     ...     ...     ...     ...      ...
2186  2016     12   28  15.21    0.05       0      42      44     45.3
2187  2016     12   29   8.72    0.00       0      44      47     45.3
2188  2016     12   30   8.50    0.05       0      47      48     45.4
2189  2016     12   31   6.93    0.02       0      48      45     45.5
2190  2017      1    1   8.05    0.03       0      45      38     45.6

      actual  friend  weekday_Fri  weekday_Mon  weekday_Sat  weekday_Sun  \
0         40      40            0            0            1            0
1         39      50            0            0            0            1
2         42      42            0            1            0            0
3         38      59            0            0            0            0
4         45      39            0            0            0            0
...      ...     ...          ...          ...          ...          ...
2186      47      30            0            0            0            0
2187      48      63            0            0            0            0
2188      45      57            1            0            0            0
2189      38      56            0            0            1            0
2190      37      27            0            0            0            1

      weekday_Thurs  weekday_Tues  weekday_Wed
0                 0             0            0
1                 0             0            0
2                 0             0            0
3                 0             1            0
4                 0             0            1
...             ...           ...          ...
2186              0             0            1
2187              1             0            0
2188              0             0            0
2189              0             0            0
2190              0             0            0

[2191 rows x 18 columns]
# Extract features and labels
labels = features['actual']
features = features.drop('actual', axis = 1)
# Names of six features accounting for 95% of total importance
important_feature_names = ['temp_1', 'average', 'ws_1', 'temp_2', 'friend', 'year']

# Update feature list for visualizations
feature_list = important_feature_names[:]

features = features[important_feature_names]
features.head(5)
temp_1	average	ws_1	temp_2	friend	year
0	37	45.6	4.92	36	40	2011
1	40	45.7	5.37	37	50	2011
2	39	45.8	6.26	40	42	2011
3	42	45.9	5.59	39	59	2011
4	38	46.0	3.80	42	39	2011
# Convert to numpy arrays
import numpy as np

features = np.array(features)
labels = np.array(labels)

# Training and Testing Sets
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                            test_size = 0.25, random_state = 42)
#Random Search with Random Forest

from sklearn.ensemble import RandomForestRegressor
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_features, train_labels)
Fitting 3 folds for each of 100 candidates, totalling 300 fits
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=-1)]: Done  33 tasks      | elapsed:   57.1s
[Parallel(n_jobs=-1)]: Done 154 tasks      | elapsed:  4.2min
[Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed:  8.1min finished
RandomizedSearchCV(cv=3, estimator=RandomForestRegressor(), n_iter=100,
                   n_jobs=-1,
                   param_distributions={'bootstrap': [True, False],
                                        'max_depth': [10, 20, 30, 40, 50, 60,
                                                      70, 80, 90, 100, 110,
                                                      None],
                                        'max_features': ['auto', 'sqrt'],
                                        'min_samples_leaf': [1, 2, 4],
                                        'min_samples_split': [2, 5, 10],
                                        'n_estimators': [200, 400, 600, 800,
                                                         1000, 1200, 1400, 1600,
                                                         1800, 2000]},
                   random_state=42, verbose=2)
print(rf_random.best_params_)
{'n_estimators': 1000, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 100, 'bootstrap': True}
Now that we have determined the optimal sequence of hyperparameters, we will look at how this tuning process evolved during execution and how the results; mean_fit_time, mean_score_time and n_estimators, interact with one another.

import plotly.graph_objects as go
z = rf_random.cv_results_['mean_fit_time']
y = rf_random.cv_results_['mean_score_time']
x = rf_random.cv_results_['mean_test_score']
color=rf_random.cv_results_['param_n_estimators']
color.sort()


rf_fig = px.scatter_3d(z=z,
                       x=x,
                       y=y,width=700,height=800,
                       labels={'x' : 'Mean Test Score',
                               'y' : 'Mean Score Time (s)',
                               'z' : 'Mean Fit Time (s)',
                              'color' : 'Number of Trees'},
                       title='Random Forest Regressor Hyperparameter Tuning',
                       animation_frame=rf_random.cv_results_['param_n_estimators'],
                       color=color)

rf_fig.show()
#Probablistic approaches

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
#Experiment 3: For this final experiment, we will use the standardised Iris dataset which stores 3 classes of 50 instances where each class corresponds to petal colour. Applications of this search process can be implemented in subsequent research in computer vision and object detection.

from sklearn import datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target
print(iris.feature_names)
print(iris.target_names)
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
['setosa' 'versicolor' 'virginica']
#The code below uses the external Hyperopt API which is a distributed computing library that implementes sequential decision based optimisation methods using Bayesian conjugates.

#The function 'def(hyperopt_train_test)(params)' constructs a classifier (clf) which implements the K Nearest Neighbors algorithm. The mean cross validation score is returned. Space4knn is a required parameter for the Hyperopt module that takes a parameter to search with and valid upper/lower bounds.

#The function 'def(f(params)' accepts an arbitrary params argument which then implements an internal call to hyperopt_train_test which instantiates the classifier.

#The variable 'best' calls 'fmin', a built-in Hyperopt function that finds the global minium of a convex optimisation problem. It implemets two internal calls to the prior functions and returns a Trials() object which stores, in sequential order, all parameter combinations that we searched for as an n-dimensional numpy array.

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
def hyperopt_train_test(params):
    clf = KNeighborsClassifier(**params)
    return cross_val_score(clf, x, y).mean()
space4knn = {
    'n_neighbors': hp.choice('n_neighbors', range(1,100))
}
def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}
trials = Trials()
best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
print ('best:')
print (best)
100%|█████████████████████████████████████████████| 100/100 [00:02<00:00, 36.59trial/s, best loss: -0.9800000000000001]
best:
{'n_neighbors': 5}
#Bayesian Tuning Process Visualised

import numpy as np
import plotly.express as px
import pandas as pd

df = trials
# x axis is the number of neighbors (k)
x_axis = [t['misc']['vals']['n_neighbors'] for t in trials.trials]
# y axis is the cross validation accuracy score
y_axis = [-t['result']['loss'] for t in trials.trials]
# z_axis is the time_id - essentially the iteration number
z_axis = [t['misc']['tid'] for t in trials.trials]

# recast the arrays to numpy N-D array
x_array = np.asarray(x_axis)
y_array = np.asarray(y_axis)
z_array = np.asarray(z_axis)
# flatten and create copy to convert from N-D to 1-D
x_1 = x_array.flatten()
y_1 = y_array.flatten()
z_1 = z_array.flatten()

# create fig object to visualise the results
fig2 = px.scatter_3d(x=x_1,y=y_1,z=y_1,
                     width=900,height=900,
                     labels={'x' : 'Number of Neighbors (K)',
                             'y' : 'Cross Validation Accuracy Score',
                             'z' : 'Number of iterations',
                             'color' : 'Accuracy Score'},
                     title= 'Visual Modelling of Bayesian Hyperparameter tuning with the K-NN algorithm',
                     color=x_1)
fig2.show()
