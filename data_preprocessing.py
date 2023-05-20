import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def load_data(path = "dataset/preprocessedDataset_Years.csv", nrows = None ):
    """Return the loaded data\n
    :return X"""

    data = pd.read_csv(path, nrows = nrows )
    X = data.iloc[:, : -1] # Features (all columns except the last one)
    y = data.iloc[:, -1] # Target (last column)
    return X, y



def clean_data():
    """Use preprocessing techniques for cleaning data"""
    pass

def feature_scaler(X):
    """Feature Scaling
    :return X"""
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled



def feature_selection(X):
    """Extracting best features for the dataset\n
    :return X"""
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X)
    return X_pca

def split_train_test(X,y,test_size_percentage = 30):
    """Split the data into train and test data\n
    :returns X_train, X_test, y_train, y_test"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=(test_size_percentage/100))
    return X_train, X_test, y_train, y_test

