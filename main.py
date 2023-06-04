from data_preprocessing import *
from model_training import *
from model_evaluation import *


X, y, data = load_data()
X = feature_scaler(X)
# X_train, X_test, y_train, y_test = split_train_test(X,y)
# best_param, best_score, train_model = get_best_parameters(X_train,y_train,1,5)
# evalu= evaluate_model(trained_model=train_model,X_test=X_test,y_test=y_test ,all_methods=True)
# print(evalu)
# print(best_param)
# print(best_score)



