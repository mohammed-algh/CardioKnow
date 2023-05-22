import streamlit as st

hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def main():
    # Display the input fields
    age = st.number_input("Age", min_value=0, max_value=150, value=30)
    height = st.number_input("Height (cm)", min_value=0, max_value=300, value=160)
    weight = st.number_input("Weight (kg)", min_value=0, max_value=500, value=70)
    gender = st.selectbox("Gender", ["Male", "Female"])
    systolic_bp = st.number_input("Systolic Blood Pressure", min_value=0, value=120)
    diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=0, value=80)
    cholesterol = st.selectbox("Cholesterol", ["Normal", "Above Normal", "Well Above Normal"])
    glucose = st.selectbox("Glucose", ["Normal", "Above Normal", "Well Above Normal"])
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol Intake", ["No", "Yes"])
    active = st.selectbox("Active", ["No", "Yes"])

    # Perform calculations or display results based on the inputs

if __name__ == "__main__":
    main()
