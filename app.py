#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('gradient_boosting_model.pkl')

# Define the Streamlit app
st.title("Employee Attrition Prediction")

# Input form for user data
st.header("Enter Employee Details:")

# Input fields for all 27 features
age = st.number_input("Age", min_value=18, max_value=100, value=30)
monthly_income = st.number_input("Monthly Income", min_value=0, value=5000)
job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
remote_work = st.selectbox("Remote Work", [0, 1])  # 0: No, 1: Yes
gender = st.selectbox("Gender", [0, 1])  # 0: Female, 1: Male
years_at_company = st.number_input("Years at Company", min_value=0, value=5)
number_of_promotions = st.number_input("Number of Promotions", min_value=0, value=0)
number_of_dependents = st.number_input("Number of Dependents", min_value=0, value=0)
education_level = st.selectbox("Education Level", [1, 2, 3, 4, 5])
company_tenure = st.number_input("Company Tenure", min_value=0, value=5)
innovation_opportunities = st.selectbox("Innovation Opportunities", [0, 1])  # 0: No, 1: Yes
leadership_opportunities = st.selectbox("Leadership Opportunities", [0, 1])  # 0: No, 1: Yes
job_role = st.selectbox("Job Role", [1, 2, 3, 4, 5, 6, 7, 8, 9])  # Example mapping for roles
employee_recognition = st.selectbox("Employee Recognition", [0, 1])  # 0: No, 1: Yes
company_size = st.selectbox("Company Size", [1, 2, 3])  # Example: Small, Medium, Large
performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4, 5])
company_reputation = st.selectbox("Company Reputation", [1, 2, 3, 4, 5])
job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4, 5])
overtime = st.selectbox("Overtime", [0, 1])  # 0: No, 1: Yes
work_life_balance = st.selectbox("Work-Life Balance", [1, 2, 3, 4, 5])
distance_from_home = st.number_input("Distance from Home", min_value=0, value=10)
marital_status = st.selectbox("Marital Status", [0, 1, 2])  # Example mapping: Single, Married, Divorced

# Feature engineering for additional columns
tenure_to_age_ratio = company_tenure / (age + 1)  # Avoid division by zero
income_to_joblevel_ratio = monthly_income / (job_level + 1)
promotions_to_tenure_ratio = number_of_promotions / (years_at_company + 1)
dependents_per_year = number_of_dependents / (years_at_company + 1)

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    'Age': [age],
    'Monthly Income': [monthly_income],
    'Job Level': [job_level],
    'Remote Work': [remote_work],
    'Gender': [gender],
    'Years at Company': [years_at_company],
    'Number of Promotions': [number_of_promotions],
    'Number of Dependents': [number_of_dependents],
    'Education Level': [education_level],
    'Company Tenure': [company_tenure],
    'Innovation Opportunities': [innovation_opportunities],
    'Leadership Opportunities': [leadership_opportunities],
    'Job Role': [job_role],
    'Employee Recognition': [employee_recognition],
    'Company Size': [company_size],
    'Performance Rating': [performance_rating],
    'Company Reputation': [company_reputation],
    'Job Satisfaction': [job_satisfaction],
    'Overtime': [overtime],
    'Work-Life Balance': [work_life_balance],
    'Distance from Home': [distance_from_home],
    'Marital Status': [marital_status],
    'Tenure_to_Age_Ratio': [tenure_to_age_ratio],
    'Income_to_JobLevel_Ratio': [income_to_joblevel_ratio],
    'Promotions_to_Tenure_Ratio': [promotions_to_tenure_ratio],
    'Dependents_per_Year': [dependents_per_year]
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




