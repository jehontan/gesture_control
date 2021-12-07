import numpy as np
from gesture_control.knn import KNNClassifier

def generate_data(n=100):
    centers = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]

    data = []
    labels = np.repeat([0,1,2,3], n, axis=0).flatten()
    for center in centers:
        data.append(np.random.normal(center, scale=0.5, size=(n,2)))

    data = np.vstack(data)

    return data, labels

def test_knn():
    X_train, Y_train = generate_data()
    knn = KNNClassifier(X_train, Y_train, k=5)

    point = [0.5, 0.5]
    res, _ = knn.predict(point)
    assert res == 0

    point = [2.5, 0.5]
    res, _ = knn.predict(point)
    assert res == 1

    point = [1.5, 1.5]
    res, _ = knn.predict(point)
    assert res == 2