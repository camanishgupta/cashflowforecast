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
def calculate_metrics(data, total_sales_col, credit_sales_col):
    data['expense_ratio'] = data[total_sales_col] * 0.3  # Example: 30% of total sales
    data['collection_rate'] = data[credit_sales_col] / data[total_sales_col]  # Ratio of credit sales to total sales
    data['payout_days'] = 30  # Example: fixed payout days, can be adjusted based on your logic
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
    forecast['cash_flow'] = forecast['yhat'] + cash_sales - cash_expenses - cash_paid_to_suppliers
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

    # Calculate additional metrics
    data = calculate_metrics(data, total_sales_col, credit_sales_col)
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
        monthly_increases = []
        for month in range(1, 13):
            increase = st.number_input(f'Assumed % Increase for Month {month} (as a decimal, e.g., 0.05 for 5%)', value=0.05)
            monthly_increases.append(increase)

        # Calculate projected sales for 2025 based on assumptions
        for i in range(12):
            forecast.loc[forecast['ds'].dt.month == (i + 1), 'yhat'] *= (1 + monthly_increases[i])

        # Recalculate cash flow with updated forecast
        cash_flow = calculate_cash_flow(forecast, cash_sales, cash_expenses, cash_paid_to_suppliers)
        st.write("Updated Cash Flow Forecast for 2025:")
        st.write(cash_flow)

# Instructions for GitHub and Streamlit deployment
st.write("To deploy this app on Streamlit, follow these steps:")
st.write("1. Push your code to a GitHub repository.")
st.write("2. Use Streamlit sharing or Streamlit Cloud to deploy your app.")
