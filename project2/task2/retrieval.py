from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import numpy as np
from pathlib import Path
import os
import pickle


class NNS:
    def __init__(self, k=5):
        """
        Initialize the NNS with a specified value of k.

        Parameters:
        - k: Number of neighbors (default is 5).
        """
        self.k = k
        self.nn_model = NearestNeighbors(n_neighbors=k, algorithm="auto")

    def fit(self, X_train):
        """
        Fit the NNS to the repository data.

        Parameters:
        - X_train: Repository data.
        """
        self.nn_model.fit(X_train)

    def predict(self, X_test):
        """
        Find the IDs of the k repository data points that are closest to the test sample points.

        Parameters:
        - X_test: Test data points.

        Returns:
        - y_pred: IDs of the k repository data points that are closest to the test sample points.
        """
        _, indices = self.nn_model.kneighbors(X_test)
        return indices


class Retrieval:
    def __init__(self, repository_data):
        """
        Initialize the Retrieval model with the repository data.
        Args:
            repository_data: The image repository which you need to search in.
        """
        retrieval_repository_data = normalize(repository_data[:, 1:], axis=1)

        self.model = NNS(k=5)
        self.model.fit(X_train=retrieval_repository_data)

    def inference(self, X: np.array) -> np.array:
        """
        Find 5 images that are most similar to the given image in the repository.
        Args:
            X: Feature vectors of the data for which similar images need to be retrieved. X.shape=[a, 256]

        Returns:
            A numpy array with shape=[a, 5], where 'a' is the number of data points.
        """
        X_normalized = normalize(X, axis=1)
        return self.model.predict(X_normalized)


if __name__ == "__main__":
    repository_data_path = Path(
        os.path.dirname(os.path.abspath(__file__)),
        "image_retrieval_repository_data.pkl",
    )
    with open(repository_data_path, "rb") as f:
        repository_data = pickle.load(f)

    retrieval_system = Retrieval(repository_data=repository_data)

    # Example test data
    test_data = np.random.rand(10, 256)

    retrieved_indices = retrieval_system.inference(test_data)
    print("Retrieved indices:", retrieved_indices)
