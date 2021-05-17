import numpy as np
from sklearn.model_selection import GridSearchCV

class CustomGridSearchCV(GridSearchCV):
    """Enable multioutputclassifier grid search using classifiers that don't
       support single class labels (all 0 or all 1), by extracting single class labels
       before fitting the pipeline, and restoring them after predicting classifications.
    """

    def fit(self, X, y=None, **fit_params):
        """ Extract and store single class label columns, then fit remaining labels as normal.
        """

        # Extract and store single class label columns from y array
        if y is not None:
            # Find single class labels' column numbers and their classifications
            colsums = y.sum(axis=0)
            col_numbers = np.where((colsums==0) + (colsums==y.shape[0]))[0]
            classifications = y[0][col_numbers]

            # Store the column numbers and classifications
            self.single_class_labels = list(zip(col_numbers, classifications))

            # Delete these columns from the y array
            y = np.delete(y, col_numbers, axis=1)

        super().fit(X, y, **fit_params)

    def predict(self, X):
        """ Predict as normal, then restore single class label columnss to predictions.
        """
     
        y_pred = super().predict(X)

        # Restore single class labels to y_pred
        for col_number, classification in self.single_class_labels:
            col = np.full(y_pred.shape[0], classification)
            y_pred = np.insert(y_pred, col_number, col, axis=1)

        return y_pred
