def shapley_value_single(data_set,model,feature_index,sample_index):
    prediction = model(sample)



def shapley_value(data_set, model, *, feature_index = None, sample_index = None):
    if feature_index == None and sample_index == None:
        return shapley_value_all(data_set,model)
    elif sample_index == None:
        return shapley_value_all_samples(data_set,model,feature_index)
    elif feature_index == None:
        return shapley_value_all_features(data_set,model,sample_index)
    else:
        return shapley_value_single(data_set,model,feature_index,sample_index)
