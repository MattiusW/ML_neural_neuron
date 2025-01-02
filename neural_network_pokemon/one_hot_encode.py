from sklearn.preprocessing import OneHotEncoder

def one_hot_encode(labels, num_classes):
    labels = labels.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False, categories=[range(num_classes)])
    one_hot = encoder.fit_transform(labels)
    return one_hot
