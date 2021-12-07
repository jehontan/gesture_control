from gesture_control.utils import euclidean_distance
import numpy as np

def test_euclidean_distance():
    a = np.array([[0,0]])
    b = np.array([4,3])

    res = euclidean_distance(a,b)
    assert res == 5
