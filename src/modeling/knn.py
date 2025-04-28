# models/knn.py
from sklearn.neighbors import KNeighborsClassifier

def get_model(n_neighbors=5):
  return KNeighborsClassifier(n_neighbors=n_neighbors)

def get_hyperparameters():
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],  # Number of neighbors
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    return param_grid