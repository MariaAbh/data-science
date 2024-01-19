import pytest
import math
import numpy as np
from shapley_values import shapley_value
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = datasets.load_iris()
iris_data = iris_dataset.data
iris_target = iris_dataset.target
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.3, random_state=42)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

def model(test_data_point):
    test_data_point = np.array(test_data_point)
    test_data_point = test_data_point.reshape(1,-1)
    prediction = knn.predict(test_data_point)
    return prediction

def test_shapley_value_single():
    res = shapley_value(X_test,model,feature_index=3,sample_index=1)
    print(res)
    assert(False)

def test_shapley_value_all_features():
    res = shapley_value(X_test,model,sample_index=1)
    print(res)
    assert(False)

def test_shapley_value_all_samples():
    res = shapley_value(X_test,model,feature_index=3)
    print(res)
    assert(False)

def test_shapley_value_all():
    res = shapley_value(X_test,model)
    print(res)
    assert(False)
