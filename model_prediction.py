import pickle
def model_predict(age,height,weight,gender,systolic_bp,diastolic_bp,cholesterol,glucose,smoking,alcohol,active):
    """Use the Trained model to predict data"""
    trained_model = pickle.load(open('models/GradientBoosting.pkl', 'rb'))
    prediction = trained_model.predict(age,height,weight,gender,systolic_bp,diastolic_bp,cholesterol,glucose,smoking,alcohol,active)
    # use chatgpt api to suggest an action
    pass