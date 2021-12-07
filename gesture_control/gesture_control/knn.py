import numpy as np
from .utils import euclidean_distance

class KNNClassifier:
    def __init__(self, X_train, Y_train, k=5, distance_fn=euclidean_distance, weight='distance'):
        '''
        Initilaize KNN classifer.

        Parameters
        ==========
        X_train : (m,n) ArrayLike (double)
            m number of n-length feature vectors.
        y_train : (m,) ArrayLike (int)
            Corresponding classes.
        k : int
            Number of nearest neighbours to consider.
        distance_fn : function
            Function to use for calculating distance between feature vectors.
        '''
        self.X_train = X_train
        self.Y_train = Y_train
        self.k = k
        self.distance_fn = distance_fn
        self.weight = weight

    def predict(self, x):
        '''
        Predict the class of given feature vector.

        Parameters
        ==========
        x : (n,) ArrayLike
            Feature vector classify.

        Returns
        =======
        y : int
            Predicted class.
        score : float
            Score of prediction.
        '''
        distances = self.distance_fn(self.X_train, x)
        indices = np.argsort(distances)
        k_nearest_labels = self.Y_train[indices[:self.k]]
        
        if self.weight == 'uniform':
            return np.argmax(np.bincount(k_nearest_labels))
        elif self.weight == 'distance':
            scores = dict()
            best_score = 0
            best_label= None

            for i in indices[:self.k]:
                label = self.Y_train[i]
                if not label in scores:
                    scores[label] = 0
                scores[label] += 1.0/distances[i]
                if scores[label] > best_score:
                    best_score = scores[label]
                    best_label = label

            return best_label, best_score
        else:
            raise ValueError('Unknown weight type.')
        