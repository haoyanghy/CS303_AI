import numpy as np
import pickle
import joblib
from pathlib import Path
import os


class Classifier:
    def __init__(self):
        """
        Load the model and normalization parameters as member variables when instantiating the classifier.
        """
        root_path = os.path.dirname(os.path.abspath(__file__))
        self.model = pickle.load(
            open(Path(root_path, "classification_model.pkl"), "rb")
        )
        self.mean = pickle.load(open(Path(root_path, "classification_mean.pkl"), "rb"))
        self.std_dev = pickle.load(
            open(Path(root_path, "classification_std.pkl"), "rb")
        )

    def inference(self, X: np.array) -> np.array:
        """
        Inference on the test data
        Args:
            X: All the feature vectors with dim=256 of the data which needs to be classified, X.shape=[a, 256],
               where 'a' is the number of the test data.

        Returns:
            All classification results as an int vector with dim=a, where 'a' is the number of the test data.
            The ith element of the results vector is the classification result of the ith test data, which is
            the index of the category.
        """
        if X.shape[1] == 257:
            X = X[:, 1:]
        X_normalized = (X - self.mean) / self.std_dev
        predictions = self.model.predict(X_normalized)
        return predictions


if __name__ == "__main__":
    test_data_path = Path(
        os.path.dirname(os.path.abspath(__file__)), "classification_train_data.pkl"
    )
    test_data = pickle.load(open(test_data_path, "rb"))

    classifier = Classifier()
    predictions = classifier.inference(test_data)
    predictions_path = Path(
        os.path.dirname(os.path.abspath(__file__)), "classification_predictions.pkl"
    )
    with open(predictions_path, "wb") as f:
        pickle.dump(predictions, f)

    print(f"Predictions saved to {predictions_path}")
