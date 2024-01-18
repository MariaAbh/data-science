from shapley_values import shapley_value
import pytest
import math

m = [
    [100,5,50,8],
    [200,7,70,8],
    [300,6,80,7],
    ]
def normalize(data_set):
    means = [sum(column)/len(column) for column in zip(*data_set)]
    std_devs = [
        (sum((xi-mean)**2 for xi in column)/len(data_set))**.5
        for column, mean in
        zip(zip(*data_set), means)
    ]
    return [
        [(pi - mean)/std_dev for pi,mean,std_dev in zip(point, means, std_devs)]
        for point in data_set
    ]
m = normalize(m)

def model_declaration(data_point):
    weights = [0.2,0.3,-0.4,0.1]
    operation = [math.exp(wi*li) for wi,li in zip(weights,data_point)]
    sum_op = sum(operation)
    print(sum_op)
    op = [e/sum_op for e in operation]
    print(op,max(op))
    return max(op)

def test_shapley_value_single():
    res = shapley_value(m,model_declaration,feature_index=3,sample_index=1)
    print(res)
    assert(False)

