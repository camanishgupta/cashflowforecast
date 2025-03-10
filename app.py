import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Function to load data from uploaded CSV
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

# Function to calculate additional metrics
def calculate_metrics(data):
    # Calculate expense_ratio, collection_rate, and payout_days
    data['expense_ratio'] = data['total_sales'] * 0.3  # Example: 30% of total sales
    data['collection_rate'] = data['credit_sales'] / data['total_sales']  # Example: ratio of credit sales to total sales
    data['payout_days'] = 30  # Example: fixed payout days, can be adjusted based on your logic
    return data

# Function to train the model
def train_model(data):
    required_columns = ['total_sales', 'credit_sales', 'expense_ratio', 'collection_rate', 'payout_days']

    # Prepare the data
    X = data[required_columns]
    y = data['total_sales']  # Assuming we want to predict total sales

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    st.write(f'Mean Squared Error: {mse}')

    # Save the model
    joblib.dump(model, 'sales_prediction_model.pkl')

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

    # Calculate additional metrics
    data = calculate_metrics(data)
    st.write("Data with Calculated Metrics:")
    st.write(data)

    # Input fields for projections
    expense_ratio_input = st.number_input('Enter Expense Ratio (as a decimal, e.g., 0.3 for 30%)', value=0.3)
    collection_rate_input = st.number_input('Enter Collection Rate (as a decimal, e.g., 0.75 for 75%)', value=0.75)
    payout_days_input = st.number_input('Enter Payout Days', value=30)

    if st.button('Train Model'):
        # Update the calculated metrics with user inputs
        data['expense_ratio'] = expense_ratio_input
        data['collection_rate'] = collection_rate_input
        data['payout_days'] = payout_days_input

        train_model(data)
        st.success('Model trained and saved successfully!')

    # Load and use the model for predictions
    if st.button('Make Prediction'):
        model = joblib.load('sales_prediction_model.pkl')
        input_data = {
            'total_sales': st.number_input('Total Sales'),
            'credit_sales': st.number_input('Credit Sales'),
            'expense_ratio': expense_ratio_input,
            'collection_rate': collection_rate_input,
            'payout_days': payout_days_input
        }
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        st.write(f'Predicted Total Sales: {prediction[0]}')

# Instructions for GitHub and Streamlit deployment
st.write("To deploy this app on Streamlit, follow these steps:")
st.write("1. Push your code to a GitHub repository.")
st.write("2. Use Streamlit sharing or Streamlit Cloud to deploy your app.")
