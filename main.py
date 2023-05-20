from data_preprocessing import *
from model_training import *
from model_evaluation import *

X, y = load_data()
X = feature_scaler(X)
X_train, X_test, y_train, y_test = split_train_test(X,y)

best_parameters, best_score, trained_model = get_best_parameters(X,y,3,5)
evaluation = evaluate_model(trained_model,X_test,y_test,all_methods=True)
print(evaluation)
print(best_parameters)
print(best_score)


