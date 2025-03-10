import pandas as pd
import streamlit as st
from fbprophet import Prophet
import numpy as np

# Function to load data from uploaded CSV
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

# Function to calculate additional metrics
def calculate_metrics(data, total_sales_col, credit_sales_col, total_expenses_col, credit_expenses_col):
    data['expense_ratio'] = data[total_expenses_col] / data[total_sales_col]  # Ratio of total expenses to total sales
    data['collection_days'] = (data[credit_sales_col] / data[total_sales_col]) * 30  # Days in collection based on credit sales
    data['credit_sales_ratio'] = data[credit_sales_col] / data[total_sales_col]  # Ratio of credit sales to total sales
    return data

# Function to forecast sales using Prophet
def forecast_sales(data, total_sales_col):
    df = data[[total_sales_col]].reset_index()
    df.columns = ['ds', 'y']  # Rename columns for Prophet
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=12, freq='M')  # Forecast for 12 months
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Function to calculate cash flow
def calculate_cash_flow(forecast, cash_sales, cash_expenses, cash_paid_to_suppliers):
    total_cash_collected = forecast['yhat'] + cash_sales
    forecast['cash_flow'] = total_cash_collected - cash_expenses - cash_paid_to_suppliers
    return forecast[['ds', 'cash_flow']]

# Streamlit app layout
st.title('AI Model Training for Cash Flow Predictions')

# Sample CSV download link
st.markdown("Download a sample CSV file to edit:")
st.markdown("[Sample CSV](https://example.com/sample_cash_flow_data.csv)")  # Replace with actual link

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Load data from the uploaded file
data = load_data(uploaded_file)

if data is not None:
    st.write("Data Preview:")
    st.write(data)
    st.write("Available Columns:")
    st.write(data.columns)  # Display the columns in the uploaded DataFrame

    # Mapping user input for column names
    total_sales_col = st.selectbox("Select Total Sales Column", options=data.columns)
    credit_sales_col = st.selectbox("Select Credit Sales Column", options=data.columns)
    total_expenses_col = st.selectbox("Select Total Expenses Column", options=data.columns)
    credit_expenses_col = st.selectbox("Select Credit Expenses Column", options=data.columns)

    # Calculate additional metrics
    data = calculate_metrics(data, total_sales_col, credit_sales_col, total_expenses_col, credit_expenses_col)
    st.write("Data with Calculated Metrics:")
    st.write(data)

    # Forecast sales using Prophet
    if st.button('Forecast Sales'):
        forecast = forecast_sales(data, total_sales_col)
        st.write("Sales Forecast:")
        st.write(forecast)

        # Input fields for assumptions
        cash_sales = st.number_input('Enter Cash Sales', value=0)
        cash_expenses = st.number_input('Enter Cash Expenses', value=0)
        cash_paid_to_suppliers = st.number_input('Enter Cash Paid to Suppliers', value=0)

        # Calculate cash flow
        cash_flow = calculate_cash_flow(forecast, cash_sales, cash_expenses, cash_paid_to_suppliers)
        st.write("Cash Flow Forecast:")
        st.write(cash_flow)

        # Monthly percentage increase assumptions for 2025
        total_sales_increases = []
        total_expenses_increases = []
        for month in range(1, 13):
            sales_increase = st.number_input(f'Assumed % Increase for Total Sales Month {month} (as a decimal, e.g., 0.05 for 5%)', value=0.05)
            total_sales_increases.append(sales_increase)
            expenses_increase = st.number_input(f'Assumed % Increase for Total Expenses Month {month} (as a decimal, e.g., 0.05 for 5%)', value=0.05)
            total_expenses_increases.append(expenses_increase)

        # Calculate projected sales and expenses for 2025 based on assumptions
        for i in range(12):
            forecast.loc[forecast['ds'].dt.month == (i + 1), 'yhat'] *= (1 + total_sales_increases[i])
            cash_expenses *= (1 + total_expenses_increases[i])  # Update cash expenses based on user input

        # Recalculate cash flow with updated forecast
        cash_flow = calculate_cash_flow(forecast, cash_sales, cash_expenses, cash_paid_to_suppliers)
        st.write("Updated Cash Flow Forecast for 2025:")
        st.write(cash_flow)

# Instructions for GitHub and Streamlit deployment
st.write("To deploy this app on Streamlit, follow these steps:")
st.write("1. Push your code to a GitHub repository.")
st.write("2. Use Streamlit sharing or Streamlit Cloud to deploy your app.")
