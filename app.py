#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import pandas as pd

# Load the trained Gradient Boosting model
model = joblib.load('gradient_boosting_model.pkl')

# Define the Streamlit app
st.title("Employee Attrition Prediction")

# Input form for user data
st.header("Enter Employee Details:")

# Input fields for all 27 features
job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
remote_work = st.selectbox("Remote Work", [0, 1])  # 0: No, 1: Yes
gender = st.selectbox("Gender", [0, 1])  # 0: Female, 1: Male
number_of_promotions = st.number_input("Number of Promotions", min_value=0, value=0)
number_of_dependents = st.number_input("Number of Dependents", min_value=0, value=0)
years_at_company = st.number_input("Years at Company", min_value=0, value=5)
education_level = st.selectbox("Education Level", [1, 2, 3, 4, 5])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
company_tenure = st.number_input("Company Tenure (in years)", min_value=0, value=5)
innovation_opportunities = st.number_input("Innovation Opportunities (scale 1-5)", min_value=1, max_value=5, value=3)
monthly_income = st.number_input("Monthly Income", min_value=0, value=5000)
leadership_opportunities = st.number_input("Leadership Opportunities (scale 1-5)", min_value=1, max_value=5, value=3)
job_role = st.selectbox("Job Role", list(range(1, 11)))  # Assuming 1-10 job roles
overtime = st.selectbox("Overtime (Yes: 1, No: 0)", [0, 1])
work_life_balance = st.number_input("Work-Life Balance (scale 1-5)", min_value=1, max_value=5, value=3)
distance_from_home = st.number_input("Distance from Home (in km)", min_value=0, value=10)
performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4, 5])
job_satisfaction = st.number_input("Job Satisfaction (scale 1-5)", min_value=1, max_value=5, value=3)
company_size = st.number_input("Company Size (number of employees)", min_value=0, value=500)
employee_recognition = st.number_input("Employee Recognition (scale 1-5)", min_value=1, max_value=5, value=3)
marital_status = st.selectbox("Marital Status (Single: 0, Married: 1, Divorced: 2)", [0, 1, 2])
company_reputation = st.number_input("Company Reputation (scale 1-5)", min_value=1, max_value=5, value=3)
employee_id = st.number_input("Employee ID (for internal use)", min_value=1, value=1)

# Feature engineering
tenure_to_age_ratio = company_tenure / (age + 1)  # Avoid division by zero
income_to_joblevel_ratio = monthly_income / (job_level + 1)  # Avoid division by zero
promotions_to_tenure_ratio = number_of_promotions / (years_at_company + 1)  # Avoid division by zero
dependents_per_year = number_of_dependents / (years_at_company + 1)  # Avoid division by zero

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    'Job Level': [job_level],
    'Remote Work': [remote_work],
    'Gender': [gender],
    'Number of Promotions': [number_of_promotions],
    'Number of Dependents': [number_of_dependents],
    'Promotions_to_Tenure_Ratio': [promotions_to_tenure_ratio],
    'Years at Company': [years_at_company],
    'Dependents_per_Year': [dependents_per_year],
    'Education Level': [education_level],
    'Age': [age],
    'Company Tenure': [company_tenure],
    'Innovation Opportunities': [innovation_opportunities],
    'Monthly Income': [monthly_income],
    'Leadership Opportunities': [leadership_opportunities],
    'Job Role': [job_role],
    'Tenure_to_Age_Ratio': [tenure_to_age_ratio],
    'Employee Recognition': [employee_recognition],
    'Employee ID': [employee_id],
    'Company Size': [company_size],
    'Performance Rating': [performance_rating],
    'Company Reputation': [company_reputation],
    'Job Satisfaction': [job_satisfaction],
    'Overtime': [overtime],
    'Work-Life Balance': [work_life_balance],
    'Distance from Home': [distance_from_home],
    'Income_to_JobLevel_Ratio': [income_to_joblevel_ratio],
    'Marital Status': [marital_status]
})

# Prediction
if st.button("Predict Attrition"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]
    if prediction == 1:
        st.error(f"The employee is likely to attrite with a probability of {prob[1]:.2f}")
    else:
        st.success(f"The employee is likely to stay with a probability of {prob[0]:.2f}")




# In[ ]:




