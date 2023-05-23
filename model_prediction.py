import openai
import json
import pickle
import pandas as pd

openai.api_key = "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"


def model_predict(age,gender,height,weight,systolic_bp,diastolic_bp,cholesterol,glucose,smoking,alcohol,active):
    """Use the Trained model to predict data"""
    BMI = weight/pow((height/100),2)
    trained_model = pickle.load(open('models/GradientBoosting.pkl', 'rb'))
    df = pd.DataFrame(columns=["age","gender","height","weight","BMI","ap_hi","ap_lo","cholesterol","gluc","smoke","alco","active"])
    df.loc[0] = [age,gender,height,weight,BMI,systolic_bp,diastolic_bp,cholesterol,glucose,smoking,alcohol,active]
    df['gender'] = df['gender'].map({'Female': 1, 'Male': 2})
    df['cholesterol'] = df['cholesterol'].map({'Normal': 1, 'Above Normal': 2, 'Well Above Normal': 3})
    df['gluc'] = df['gluc'].map({'Normal': 1, 'Above Normal': 2, 'Well Above Normal': 3})
    df['smoke'] = df['smoke'].map({'Yes': 1, 'No': 0})
    df['alco'] = df['alco'].map({'Yes': 1, 'No': 0})
    df['active'] = df['active'].map({'Yes': 1, 'No': 0})

    prediction = trained_model.predict(df)
    if prediction == 1:
        prediction = "Presence Cardiovascular disease"
    elif prediction == 0:
        prediction = "Absence Cardiovascular disease"
    return prediction
    # use chatgpt api to suggest an action

def generate_text(prompt):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = response.choices[0].text.strip()
    return message
