from data_preprocessing import *
from model_training import *
from model_evaluation import *

X, y, data = load_data()
X = feature_scaler(X)
X_train, X_test, y_train, y_test = split_train_test(X, y)

trained_model = train_model(X_train,y_train,2)
eval = evaluate_model(trained_model,X_test,y_test,all_methods=True)

print(eval)
save_model(trained_model,"GradientBoosting")
