import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class DataHandler:
    """
    Handles loading, preprocessing, and splitting of the dataset.
    """
    def __init__(self, filepath):
        """
        Initializes the DataHandler with the path to the dataset.
        
        Args:
            filepath (str): The path to the CSV file.
        """
        self.filepath = filepath
        self.X = None
        self.y = None

    def load_data(self, feature_cols, label_col):
        """
        Loads data from the CSV file and separates features and labels.

        Args:
            feature_cols (slice): A slice object for selecting feature columns.
            label_col (slice): A slice object for selecting the label column.
        """
        try:
            df = pd.read_csv(self.filepath, header=None)
            X_df = df.iloc[:, feature_cols].astype(float)
            y_df = df.iloc[:, label_col].astype(float)
            
            # Normalize features
            normalizer = MinMaxScaler()
            self.X = normalizer.fit_transform(X_df)
            self.y = y_df.to_numpy().flatten() # Flatten to make it a 1D array
        except FileNotFoundError:
            print(f"Error: The file '{self.filepath}' was not found.")
            exit()
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            exit()

    def split_data(self, test_size=0.3, random_state=None):
        """
        Splits the data into training and testing sets.

        Args:
            test_size (float): The proportion of the dataset to allocate to the test split.
            random_state (int, optional): Seed for the random number generator for reproducibility.

        Returns:
            tuple: A tuple containing (X_train, X_test, y_train, y_test).
        """
        if self.X is None or self.y is None:
            raise ValueError("Data has not been loaded yet. Call load_data() first.")

        dataset_size = self.X.shape[0]
        test_set_size = int(dataset_size * test_size)

        rng = np.random.default_rng(random_state)
        indices = rng.permutation(dataset_size)
        
        test_indices = indices[:test_set_size]
        train_indices = indices[test_set_size:]

        X_train, X_test = self.X[train_indices], self.X[test_indices]
        y_train, y_test = self.y[train_indices], self.y[test_indices]
        
        return X_train, X_test, y_train, y_test


class KNNClassifier:
    """
    A K-Nearest Neighbors classifier implemented from scratch.
    """
    def __init__(self, k=3):
        """
        Initializes the KNNClassifier.
        
        Args:
            k (int): The number of nearest neighbors to consider.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        "Trains" the model by storing the training data.
        
        Args:
            X_train (np.ndarray): The training feature data.
            y_train (np.ndarray): The training label data.
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Predicts the labels for the test data.
        
        Args:
            X_test (np.ndarray): The test feature data.
            
        Returns:
            np.ndarray: The predicted labels for the test data.
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("The model has not been fitted yet. Call fit() first.")
        
        distances = self._calculate_distances(X_test)
        predictions = self._get_predictions_from_distances(distances)
        return predictions

    def _calculate_distances(self, X_test):
        """
        Calculates the Euclidean distance between each test point and all training points.
        
        Args:
            X_test (np.ndarray): The test feature data.
            
        Returns:
            np.ndarray: A matrix of distances.
        """
        # Using broadcasting for an efficient calculation
        diff = X_test[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]
        sq_diff = diff ** 2
        sum_sq_diff = np.sum(sq_diff, axis=2)
        return np.sqrt(sum_sq_diff)

    def _get_predictions_from_distances(self, distances):
        """
        Assigns labels based on the k-nearest neighbors.
        
        Args:
            distances (np.ndarray): A matrix of distances.
            
        Returns:
            np.ndarray: An array of predicted labels.
        """
        # Get indices of the k nearest neighbors for each test sample
        neighbor_indices = np.argsort(distances, axis=1)[:, :self.k]
        
        predictions = []
        for indices in neighbor_indices:
            # Get the labels of the nearest neighbors
            neighbor_labels = self.y_train[indices]
            # Find the most common label
            values, counts = np.unique(neighbor_labels, return_counts=True)
            most_common_label = values[np.argmax(counts)]
            predictions.append(most_common_label)
            
        return np.array(predictions)


class Visualizer:
    """
    A utility class for creating visualizations.
    """
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, k):
        """
        Calculates accuracy and plots a confusion matrix.
        
        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.
            k (int): The k-value used for the prediction, for titling the plot.
        """
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy (k={k}): {accuracy:.3f}")

        cm = confusion_matrix(y_true, y_pred)
        labels = np.unique(y_true)

        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix (k={k}), Accuracy={accuracy:.2f}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()

