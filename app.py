import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Health Risk Prediction", page_icon="🏥", layout="wide")


# Load the model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('logistic_regression_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please make sure the model and scaler files are in the correct location.")
        return None, None


model, scaler = load_model_and_scaler()
if model is None or scaler is None:
    st.stop()

# Define the correct order of features
feature_order = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
                 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']

# Custom CSS for a beautiful design
st.markdown("""
<style>
    body {
        color: #333;
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #4CAF50;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Health Risk Prediction")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("This app uses a Logistic Regression model to predict health risks based on various factors.")

# Input fields
st.subheader("Enter Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    high_bp = st.selectbox("High Blood Pressure", ["No", "Yes"])
    high_chol = st.selectbox("High Cholesterol", ["No", "Yes"])
    chol_check = st.selectbox("Cholesterol Check", ["No", "Yes"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    stroke = st.selectbox("Stroke History", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease or Attack", ["No", "Yes"])

with col2:
    phys_activity = st.selectbox("Physical Activity", ["No", "Yes"])
    fruits = st.selectbox("Consumes Fruits Regularly", ["No", "Yes"])
    veggies = st.selectbox("Consumes Vegetables Regularly", ["No", "Yes"])
    hvy_alcohol_consump = st.selectbox("Heavy Alcohol Consumption", ["No", "Yes"])
    any_healthcare = st.selectbox("Any Healthcare", ["No", "Yes"])
    no_doc_bc_cost = st.selectbox("No Doctor because of Cost", ["No", "Yes"])
    gen_hlth = st.selectbox("General Health", ["Excellent", "Very Good", "Good", "Fair", "Poor"])

with col3:
    ment_hlth = st.number_input("Mental Health (0-30 days)", min_value=0, max_value=30, value=0)
    phys_hlth = st.number_input("Physical Health (0-30 days)", min_value=0, max_value=30, value=0)
    diff_walk = st.selectbox("Difficulty Walking", ["No", "Yes"])
    sex = st.selectbox("Sex", ["Female", "Male"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    education = st.selectbox("Education Level",
                             ["Never attended school or only kindergarten", "Grades 1 through 8 (Elementary)",
                              "Grades 9 through 11 (Some high school)", "Grade 12 or GED (High school graduate)",
                              "College 1 year to 3 years (Some college or technical school)",
                              "College 4 years or more (College graduate)"])
    income = st.selectbox("Income",
                          ["Less than $10,000", "$10,000 to less than $15,000", "$15,000 to less than $20,000",
                           "$20,000 to less than $25,000", "$25,000 to less than $35,000",
                           "$35,000 to less than $50,000", "$50,000 to less than $75,000", "$75,000 or more"])

# Prediction button
if st.button("Predict Health Risk"):
    # Prepare input data
    input_data = {
        'HighBP': 1 if high_bp == "Yes" else 0,
        'HighChol': 1 if high_chol == "Yes" else 0,
        'CholCheck': 1 if chol_check == "Yes" else 0,
        'BMI': bmi,
        'Smoker': 1 if smoker == "Yes" else 0,
        'Stroke': 1 if stroke == "Yes" else 0,
        'HeartDiseaseorAttack': 1 if heart_disease == "Yes" else 0,
        'PhysActivity': 1 if phys_activity == "Yes" else 0,
        'Fruits': 1 if fruits == "Yes" else 0,
        'Veggies': 1 if veggies == "Yes" else 0,
        'HvyAlcoholConsump': 1 if hvy_alcohol_consump == "Yes" else 0,
        'AnyHealthcare': 1 if any_healthcare == "Yes" else 0,
        'NoDocbcCost': 1 if no_doc_bc_cost == "Yes" else 0,
        'GenHlth': ["Excellent", "Very Good", "Good", "Fair", "Poor"].index(gen_hlth) + 1,
        'MentHlth': ment_hlth,
        'PhysHlth': phys_hlth,
        'DiffWalk': 1 if diff_walk == "Yes" else 0,
        'Sex': 1 if sex == "Male" else 0,
        'Age': age,
        'Education': ["Never attended school or only kindergarten", "Grades 1 through 8 (Elementary)",
                      "Grades 9 through 11 (Some high school)", "Grade 12 or GED (High school graduate)",
                      "College 1 year to 3 years (Some college or technical school)",
                      "College 4 years or more (College graduate)"].index(education) + 1,
        'Income': ["Less than $10,000", "$10,000 to less than $15,000", "$15,000 to less than $20,000",
                   "$20,000 to less than $25,000", "$25,000 to less than $35,000", "$35,000 to less than $50,000",
                   "$50,000 to less than $75,000", "$75,000 or more"].index(income) + 1
    }

    # Create DataFrame and ensure correct feature order
    input_df = pd.DataFrame([input_data])[feature_order]

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    # Display result
    st.subheader("Prediction Result")
    if prediction == 0:
        st.success("The patient is not at risk of disease.")
    elif prediction == 1:
        st.warning("The patient is at mild risk of diabetes.")
    else:
        st.error("The patient is at severe risk of diabetes.")

    # Display probabilities
    st.subheader("Prediction Probabilities")
    fig, ax = plt.subplots()
    sns.barplot(x=['No Risk', 'Mild Risk', 'Severe Risk'], y=probabilities, ax=ax)
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probabilities')
    st.pyplot(fig)

# Feature Importance
st.subheader("Feature Importance")

# Get the coefficients from the model
coefficients = model.coef_[0]

# Create feature importance DataFrame
feature_importance = pd.DataFrame({
    'feature': feature_order,
    'importance': np.abs(coefficients)
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
ax.set_title('Feature Importance (Coefficient Magnitude)')
st.pyplot(fig)

