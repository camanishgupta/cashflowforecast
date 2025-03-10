import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load the data
@st.cache
def load_data():
    data = pd.read_excel('SyntheticCashFlowData_Seasonal.csv')  # Load your data here
    return data

# Train the model
def train_model(data):
    # Prepare the data
    X = data[['total_sales', 'credit_sales', 'expense_ratio', 'collection_rate', 'payout_days']]
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

data = load_data()
st.write(data)

if st.button('Train Model'):
    train_model(data)
    st.success('Model trained and saved successfully!')

# Load and use the model for predictions
if st.button('Make Prediction'):
    model = joblib.load('sales_prediction_model.pkl')
    input_data = {
        'total_sales': st.number_input('Total Sales'),
        'credit_sales': st.number_input('Credit Sales'),
        'expense_ratio': st.number_input('Expense Ratio'),
        'collection_rate': st.number_input('Collection Rate'),
        'payout_days': st.number_input('Payout Days')
    }
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    st.write(f'Predicted Total Sales: {prediction[0]}')

# Instructions for GitHub and Streamlit deployment
st.write("To deploy this app on Streamlit, follow these steps:")
st.write("1. Push your code to a GitHub repository.")
st.write("2. Use Streamlit sharing or Streamlit Cloud to deploy your app.")
