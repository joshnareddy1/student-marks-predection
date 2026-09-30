import streamlit as st
import pandas as pd
import joblib
model = joblib.load("model_trained.pk4")

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓")
st.title("🎓 Student Performance Prediction")
st.write("Predict the Performance Index of a student.")
hours_studied = st.number_input("Hours Studied", min_value=0.0, value=5.0)
previous_scores = st.number_input("Previous Scores", min_value=0.0, max_value=100.0, value=75.0)
extracurricular = st.selectbox(
    "Extracurricular Activities",
    ["Yes", "No"]
)

sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)

sample_papers = st.number_input(
    "Sample Question Papers Practiced",
    min_value=0,
    value=5
)

if st.button("Predict Performance"):

    input_df = pd.DataFrame({
        "Hours Studied": [hours_studied],
        "Previous Scores": [previous_scores],
        "Extracurricular Activities": [extracurricular],
        "Sleep Hours": [sleep_hours],
        "Sample Question Papers Practiced": [sample_papers]
    })
    prediction = model.predict(input_df)
    st.success(f"Predicted Performance Index: {prediction[0]:.2f}")
    