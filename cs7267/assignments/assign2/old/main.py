from knn_classifier import DataHandler, KNNClassifier, Visualizer

def main():
    """
    Main function to run the KNN classification demonstration.
    """
    # --- Configuration ---
    FILEPATH = 'sample_data.csv'
    # The original code used the first 30 columns for features 
    # and the last column for the label.
    FEATURE_COLS = slice(0, 30)
    LABEL_COL = slice(30, 31)
    TEST_SPLIT_SIZE = 0.3
    RANDOM_STATE = 42
    K_NEIGHBORS = 5

    # 1. Initialize DataHandler and process the data
    print(f"Loading data from '{FILEPATH}'...")
    data_handler = DataHandler(FILEPATH)
    data_handler.load_data(feature_cols=FEATURE_COLS, label_col=LABEL_COL)
    
    # 2. Split data into training and testing sets
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = data_handler.split_data(
        test_size=TEST_SPLIT_SIZE, 
        random_state=RANDOM_STATE
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # 3. Initialize the KNN classifier, fit the model, and make predictions
    print(f"\nTraining KNN model with k={K_NEIGHBORS}...")
    knn = KNNClassifier(k=K_NEIGHBORS)
    knn.fit(X_train, y_train)
    
    print("Making predictions on the test set...")
    predictions = knn.predict(X_test)

    # 4. Visualize the results
    print("Displaying results...")
    Visualizer.plot_confusion_matrix(y_test, predictions, k=K_NEIGHBORS)

if __name__ == "__main__":
    main()

