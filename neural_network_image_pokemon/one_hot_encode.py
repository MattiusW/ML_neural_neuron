from sklearn.preprocessing import OneHotEncoder

def one_hot_encode(labels, num_classes):
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    labels = labels.reshape(-1, 1)
    return encoder.fit_transform(labels)