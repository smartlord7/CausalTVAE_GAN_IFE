import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler


def reduce_dimensions(X, n_components=10):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced


def sample_data(X, sample_size=1000):
    if X.shape[0] > sample_size:
        X_sampled = X[np.random.choice(X.shape[0], sample_size, replace=False)]
    else:
        X_sampled = X
    return X_sampled




# Function to preprocess the pandas DataFrame
def preprocess_data(df, target_column=None, scale_features=True, discard_columns=None):
    # Discard specified columns
    if discard_columns:
        df = df.drop(columns=discard_columns)

    # Separate features and target if a target column is specified
    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df
        y = None

    # Integer Encoding of categorical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_features) > 0:
        label_encoders = {}
        for feature in categorical_features:
            encoder = LabelEncoder()
            X[feature] = encoder.fit_transform(X[feature])
            label_encoders[feature] = encoder

    # Feature Scaling
    if scale_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y