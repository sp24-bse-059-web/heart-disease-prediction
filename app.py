
import streamlit as st
import pickle
import pandas as pd

st.title("Heart Disease Prediction System")
st.write("Enter the four details below to get a model prediction.")

with open("heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1])
cp = st.selectbox("Chest pain type (0–3)", [0, 1, 2, 3])
thalach = st.number_input(
    "Maximum heart rate achieved",
    min_value=50, max_value=250, value=150
)

if st.button("Predict"):
    data = pd.DataFrame(
        [[age, sex, cp, thalach]],
        columns=["age", "sex", "cp", "thalach"]
    )
    prediction = model.predict(data)[0]

    if prediction == 1:
        st.warning("The model predicts an indication of heart disease.")
    else:
        st.success("The model predicts no indication of heart disease.")

    st.caption("Educational project only; this is not a medical diagnosis.")
