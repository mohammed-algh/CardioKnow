import streamlit as st
import base64
from pathlib import Path
from model_prediction import *

# Remove Streamlit logo
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
      img_to_bytes(img_path)
    )
    return img_html


def main():
    st.markdown(
        """
        <div style='text-align:center'>
            <img src='data:image/png;base64,{}' alt='Example image' style='display:block;margin:auto;width:50%;'>
        </div>
        """.format(img_to_bytes("images/cardioknow.png")),
        unsafe_allow_html=True
    )
    # Display the input fields in two columns
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=150, value=30)
        height = st.number_input("Height (cm)", min_value=0, max_value=300, value=160)
        diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=0, value=80)
        cholesterol = st.selectbox("Cholesterol", ["Normal", "Above Normal", "Well Above Normal"], format_func=lambda x: x if x != "No typing" else "No")


    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"], format_func=lambda x: x if x != "No typing" else "No")
        weight = st.number_input("Weight (kg)", min_value=0, max_value=500, value=70)
        systolic_bp = st.number_input("Systolic Blood Pressure", min_value=0, value=120)
        glucose = st.selectbox("Glucose", ["Normal", "Above Normal", "Well Above Normal"], format_func=lambda x: x if x != "No typing" else "No")

    smoking = st.selectbox("Smoking", ["No", "Yes"], format_func=lambda x: x if x != "No typing" else "No")
    alcohol = st.selectbox("Alcohol Intake", ["No", "Yes"], format_func=lambda x: x if x != "No typing" else "No")
    active = st.selectbox("Active", ["No", "Yes"], format_func=lambda x: x if x != "No typing" else "No")

    # Perform calculations or display results based on the inputs

    if st.button("Process Data"):
        text = model_predict(age,gender,height,weight,systolic_bp,diastolic_bp,cholesterol,glucose,smoking,alcohol,active)
        st.write(f"# {text}")

if __name__ == "__main__":
    main()

