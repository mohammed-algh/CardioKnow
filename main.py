from data_preprocessing import *
from model_training import *
from model_evaluation import *

X, y, data = load_data()
X = feature_scaler(X)
features = use_feature_selection_model(X,y,"auto",0.002)
print(features)
