import numpy as np
from itertools import combinations, tee
from utils import euclidean_distance

def embed_hand_pose(landmarks):
    A, B = tee(combinations(landmarks))
    A = np.array(list(A))
    B = np.array(list(B))

    return euclidean_distance(A, B)