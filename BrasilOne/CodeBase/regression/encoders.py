import pandas as pd

class CustomTargetEncoder:
    def __init__(self, smoothing=1.0, columns_to_encode=None):
        """
        Initialize the encoder.
        :param smoothing: Smoothing parameter to balance between category mean and global mean.
        """
        self.smoothing = smoothing
        self.encodings = {}  # Store encodings for each categorical column
        self.global_mean = None  # Store the global mean of the target
        self.columns_to_encode = columns_to_encode

    def fit(self, X, y):
        """
        Fit the encoder on the training data.
        :param X: DataFrame containing categorical columns.
        :param y: Target variable.
        """
        self.global_mean = y.mean()

        if self.columns_to_encode is None:
            cols = X.columns
        else:
            cols = self.columns_to_encode

        for col in cols:
            # Calculate the mean target for each category
            category_means = y.groupby(X[col]).mean()
            # Calculate the count of each category
            category_counts = y.groupby(X[col]).count()
            # Apply smoothing
            smoothed_encoding = (category_means * category_counts + self.global_mean * self.smoothing) / (
                        category_counts + self.smoothing)
            # Store the encodings
            self.encodings[col] = smoothed_encoding

    def transform(self, X):
        """
        Transform the categorical columns using the learned encodings.
        :param X: DataFrame containing categorical columns.
        :return: Transformed DataFrame.
        """
        X_transformed = X.copy()

        if self.columns_to_encode is None:
            cols = X.columns
        else:
            cols = self.columns_to_encode
            
        for col in cols:
            # Replace categories with their encodings
            X_transformed[col] = X[col].map(self.encodings[col]).fillna(self.global_mean)
        return X_transformed

    def fit_transform(self, X, y):
        """
        Fit the encoder and transform the data in one step.
        :param X: DataFrame containing categorical columns.
        :param y: Target variable.
        :return: Transformed DataFrame.
        """
        self.fit(X, y)
        return self.transform(X)
    

class CustomOneHotEncoder:
    def __init__(self):
        self.columns = None

    def fit(self, data):
        self.columns = pd.get_dummies(data).columns
        return self

    def transform(self, data):
        encoded_data = pd.get_dummies(data)
        # Ensure all columns from fit are present, even if some categories are missing in transform
        for col in self.columns:
            if col not in encoded_data.columns:
                encoded_data[col] = 0
        return encoded_data[self.columns]
    


class CustomOrdinalEncoder:
    def __init__(self, feature_order_dict):
        """
        Initialize the encoder with a dictionary of feature orders.
        
        Args:
            feature_order_dict (dict): A dictionary where keys are feature names and values are lists
                                       specifying the ascending order of categories for that feature.
                                       Example: {'feature_1': ['low', 'med', 'high']}
        """
        self.feature_order_dict = feature_order_dict
        self.encoding_maps = {}  # To store the mapping of categories to integers for each feature

    def fit(self, data):
        """
        Create encoding maps for each feature based on the provided order.
        
        Args:
            data (pd.DataFrame): The input data to fit the encoder.
        """
        for feature, order in self.feature_order_dict.items():
            # Create a mapping from category to integer based on the order
            self.encoding_maps[feature] = {category: idx for idx, category in enumerate(order)}
        return self

    def transform(self, data):
        """
        Transform the input data using the encoding maps.
        
        Args:
            data (pd.DataFrame): The input data to transform.
        
        Returns:
            pd.DataFrame: The transformed data with ordinal encoding.
        """
        encoded_data = data.copy()
        for feature, encoding_map in self.encoding_maps.items():
            if feature in encoded_data.columns:
                encoded_data[feature] = encoded_data[feature].map(encoding_map)
            else:
                raise ValueError(f"Feature '{feature}' not found in the input data.")
        return encoded_data

    def fit_transform(self, data):
        """
        Fit the encoder and transform the input data in one step.
        
        Args:
            data (pd.DataFrame): The input data to fit and transform.
        
        Returns:
            pd.DataFrame: The transformed data with ordinal encoding.
        """
        self.fit(data)
        return self.transform(data)