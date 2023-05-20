from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import time

def train_model(X_train, y_train, classifier_num:int):
    """Train the model with the training data, and return the train model \n
    classifier:
        1 = Random Forest classifier\n
        2 = Gradient Boosting classifier\n
        3 = Support Vector Machine classifier\n
        4 = Logistic Regression classifier\n
        :return trained_model"""
    classifier = None

    if classifier_num == 1:
        classifier = RandomForestClassifier()
    elif classifier_num == 2:
        classifier = GradientBoostingClassifier()
    elif classifier_num == 3:
        classifier = SVC()
    elif classifier_num == 4:
        classifier = LogisticRegression()
    else:
        raise Exception("Wrong classifier was chosen")

    print("Training Started...")
    start_time = time.time()
    classifier.fit(X_train, y_train)
    print(f"Training Finished in {time.time() - start_time} seconds")
    return classifier

def get_best_parameters(X_train, y_train, classifier_num:int, cv:int, use_full_cores=True):
    """Train the GridSearchCV model with the training data, and return the train GridSearchCV model \n
        classifier:
            1 = Random Forest classifier\n
            2 = Gradient Boosting classifier\n
            3 = Support Vector Machine classifier\n
            4 = Logistic Regression classifier\n
            :returns best_parameters, best_score"""

    classifier = None
    param_grid = None
    njobs = None

    if classifier_num == 1:
        classifier = RandomForestClassifier()

        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt']
        }

    elif classifier_num == 2:
        classifier = GradientBoostingClassifier()

        param_grid = {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.1, 0.01, 0.001],
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt']
        }

    elif classifier_num == 3:
        classifier = SVC()

        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }

    elif classifier_num == 4:
        classifier = LogisticRegression()

        param_grid = {
            'penalty': ['l1', 'l2'],
            'C': [0.1, 1, 10, 100],
            'solver': ['liblinear']
        }
    else:
        raise Exception("Wrong classifier was chosen")

    if use_full_cores:
        njobs = -1


    gsc = GridSearchCV(classifier,param_grid, cv=cv, n_jobs=njobs,verbose=2)


    print("Training Started...")
    start_time = time.time()
    gsc.fit(X_train,y_train)
    print(f"Training Finished in {time.time() - start_time} seconds")

    best_params = gsc.best_params_
    best_score = gsc.best_score_

    return best_params, best_score, gsc




