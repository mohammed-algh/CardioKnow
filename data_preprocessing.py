import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import ExhaustiveFeatureSelector
import time


def load_data(path="dataset/preprocessedDataset_Years.csv",drop_features:list[str] = None, nrows=None, remove_dup=True):
    """Return the loaded data\n
    :return X"""

    data = pd.read_csv(path, nrows=nrows)

    if remove_dup:
        data.drop_duplicates(inplace=True)

    X = data.iloc[:, : -1]  # Features (all columns except the last one)
    y = data.iloc[:, -1]  # Target (last column)
    if drop_features is not None:
        X.drop(drop_features, axis=1, inplace=True)
        print(f"{drop_features} these features has been dropped")
    return X, y, data


def clean_data():
    """Use preprocessing techniques for cleaning data"""
    pass


def feature_scaler(X):
    """Feature Scaling
    :return X"""
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


def feature_selection(X, n_features=None):
    """Extracting best features for the dataset\n
    :return X"""
    pca = PCA(n_components=n_features)
    X_pca = pca.fit_transform(X)
    return X_pca


def use_feature_selection_model(X_train, y_train, n_features, tol: float, direction="forward"):
    """Return best feature combination using Random Forest classifier\n
    n_features: number of the features to be selected\n [int, auto]
    direction: method of the selecting [ "forward", "backward"]\n
    tol: If the score is not incremented by at least tol between two consecutive feature additions or removals, stop adding or removing.\n
        :return list of selected features contain either True(selected) or False(ignored)"""

    classifier = SVC()

    sfs = SequentialFeatureSelector(classifier, n_features_to_select=n_features, direction=direction, tol=tol,n_jobs=-1)

    print("Training Started...")
    start_time = time.time()
    sfs.fit(X_train, y_train)
    print(f"Training Finished in {time.time() - start_time} seconds")
    return sfs.get_support()

def Exhaustive_selection(X, y):
    classifier = SVC()

    efs = ExhaustiveFeatureSelector(classifier,min_features=2,max_features=8, n_jobs=-1)
    print("Exhaustive Selection Started...")
    start_time = time.time()
    efs.fit(X,y)
    print(f"Feature selecting finished in {time.time() - start_time} seconds")
    print(efs.best_feature_names_)
    print(efs.best_score_)



def split_train_test(X, y, test_size_percentage=30):
    """Split the data into train and test data\n
    :returns X_train, X_test, y_train, y_test"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=(test_size_percentage / 100))
    return X_train, X_test, y_train, y_test
