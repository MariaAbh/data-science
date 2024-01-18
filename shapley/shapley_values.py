import random

def shapley_value_single(data_set,model,feature_index,sample_index,coef=0.5):
    original_sample = data_set[sample_index]
    original_len = len(original_sample)
    n = coef * (len(data_set) * 2**len(original_sample))
    for i in range(n):
        random_point = random.choice(data_set)
        partition = [random.randint(0,1) for _ in range(original_len)]
        partition[feature_index] = 0
        x1 = [(p*r)+(o*(1-p)) for p,r,o in zip(partition,random_point,original_sample)]
        x2 = x1[::]
        x2[feature_index] = random_point[feature_index]
        prediction_x1 = model(x1)
        prediction_x2 = model(x2)
        difference = prediction_x1 - prediction_x2
        result += difference
    return result/n

def shapley_value_all_features(data_set,model,sample_index,coef=0.5):
    values = []
    for feature_index in range(len(data_set[0])):
        values.append(shapley_value_single(data_set,model,feature_index,sample_index,coef))
    return values

def shapley_value_all_samples(data_set,model,feature_index):
    nb_of_samples = len(data_set)
    for sample_index in range(nb_of_samples):
        result += shapley_value_single(data_set,model,feature_index,sample_index,coef)
    return result/nb_of_samples

def shapley_value_all(data_set,model):
    values = []
    nb_of_samples = len(data_set)
    for feature_index in range(len(data_set[0])):
        for sample_index in range(nb_of_samples):
            result += shapley_value_single(data_set,model,feature_index,sample_index,coef)
        values.append(result/nb_of_samples)
    return values

def shapley_value(data_set, model, *, feature_index = None, sample_index = None):
    if feature_index == None and sample_index == None:
        return shapley_value_all(data_set,model)
    elif sample_index == None:
        return shapley_value_all_samples(data_set,model,feature_index)
    elif feature_index == None:
        return shapley_value_all_features(data_set,model,sample_index)
    else:
        return shapley_value_single(data_set,model,feature_index,sample_index)
