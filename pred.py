import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import pickle
import tensorflow as tf
from datetime import datetime, timedelta
import os

# Loading scaler and models

xgb = pickle.load(open('xgboost_model_final.pickle', 'rb'))



# Prediction functions
def predict_with_xgboost(model, data):
    predictions = model.predict(data)
    return predictions


# Load data
energy_weather_ohe_model = pd.read_csv('energy_weather_ohe_model.csv')

# Convert 'time' column to datetime format for filtering
energy_weather_ohe_model['time'] = pd.to_datetime(energy_weather_ohe_model['time'])

real_values = energy_weather_ohe_model['price actual']

st.title('Energy Price Prediction App :factory::chart_with_upwards_trend:')
st.write('---')

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("What do you want to do?", ["Home", "Model Evaluation", "Prediction"])

if app_mode == "Home":
    st.write("""
        This app predicts the energy price for the next 24 hours by inputting certain values.
        It is designed to work similarly to the Spanish market system, where the price for the next day is decided at 12 o'clock every day.
             
        In this app you will be able to:
        - Evaluate the performance of our prediction models in the **Model Evaluation** section.
        - Predict future prices with our model in the **Prediction** section .
    """)

    st.image("home_pic.png", use_column_width=True)

elif app_mode == "Model Evaluation":
    st.header("Model Evaluation")
    st.write("This section shows the predictions and performance of our model for the selected date range using the provided (real) data.")

    st.subheader("Select Start Date and Time")

    col1, col2 = st.columns(2)

    with col1:
        # Date and Time Range Picker for Start DateTime
        start_date = st.date_input("Start date", value=pd.to_datetime("2015-01-07").date(), min_value=pd.to_datetime("2015-01-07").date(), max_value=pd.to_datetime("2018-12-31").date())
    
    with col2:
        # Time input for start time (hours only)
        start_time = st.time_input("Start time (hours only)", value=pd.to_datetime("00:00:00").time(), step=timedelta(hours=1))

    start_datetime = datetime.combine(start_date, start_time)
    if st.button("Predict"):
        # Calculate end datetime
        end_datetime = start_datetime + timedelta(hours=23)

        filtered_df = energy_weather_ohe_model[(energy_weather_ohe_model['time'] >= start_datetime) & (energy_weather_ohe_model['time'] <= end_datetime)]
        
        if not filtered_df.empty:
            # Prepare data for XGBoost predictions
            X_for_prediction_xgb = filtered_df.drop(columns=['time', 'price actual'])

            # Make predictions with the XGBoost model
            xgboost_predictions = predict_with_xgboost(xgb, X_for_prediction_xgb)

            # Aligning Time and Real values to match the filtered DataFrame
            aligned_time = filtered_df['time'].values
            aligned_real = filtered_df['price actual'].values
            aligned_xgboost_predictions = xgboost_predictions[:len(filtered_df)]

            # Creating the DataFrame with aligned parts
            df_predictions = pd.DataFrame({
                'Time': aligned_time,
                'Real': aligned_real,
                'XGBoost': aligned_xgboost_predictions,
            })

            # MAPE Calculation
            mape_xgboost = np.mean(np.abs((df_predictions['Real'] - df_predictions['XGBoost']) / df_predictions['Real'])) * 100

            # Display the table and MAPE
            st.write(df_predictions)
            st.write(f"MAPE for XGBoost: {mape_xgboost:.2f}%")

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(df_predictions['Time'], df_predictions['Real'], label='Real')
            plt.plot(df_predictions['Time'], df_predictions['XGBoost'], label='XGBoost Predictions')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.title('Model Predictions vs Real Data')
            st.pyplot(plt)
        else:
            st.write("No data available for the selected date range.")


elif app_mode == "Prediction":
    st.header("Prediction")
    st.write("Choose a method to predict energy prices:")

    method = st.radio("Prediction Method", ("Next Hour", "Next 24 Hours from CSV"))

    if method == "Next Hour":
        st.subheader("Predict Energy Price for Next Hour")
        st.write("Input the required values to predict the energy price for the next hour.")

        # Input form for prediction data for next hour
        with st.form("prediction_form_hour"):
            total_load_forecast = st.number_input("Demand forecast for tomorrow (MW)")
            total_load_actual_168 = st.number_input("Demand a week ago (MW)")
            total_non_renewable_168 = st.number_input("Non-renewable used a week ago (MW)")
            price_actual_48 = st.number_input("Price 2 days ago (€/MWh)")
            price_actual_72 = st.number_input("Price 3 days ago (€/MWh)")
            price_actual_168 = st.number_input("Last week's price (€/MWh)")
            
            submitted_hour = st.form_submit_button("Predict Next Hour")

            if submitted_hour:
                # Prepare data for XGBoost predictions
                data_hour = np.array([[total_load_forecast, total_load_actual_168, total_non_renewable_168, price_actual_48, price_actual_72, price_actual_168]])

                # Predict prices for the next hour using XGBoost
                predicted_price_hour = predict_with_xgboost(xgb, data_hour)

                # Display the predicted price for next hour
                st.write(f"Predicted Price for Next Hour: {predicted_price_hour[0]} €/MWh")

    elif method == "Next 24 Hours from CSV":
        st.subheader("Predict Energy Prices for Next 24 Hours from CSV")
        st.write("Upload a CSV file containing 24 hours of data with the following columns (and the following order):") 
        st.markdown('''
                - **time** - format YYYY-MM-DD HH:MM:SS
                - **total load forecast** - "Demand forecast for tomorrow (MW)"
                - **total load actual_168** -"Demand a week ago (MW)"
                - **total_non_renewable_168** - "Non-renewable used a week ago (MW)"
                - **price actual_48** - "Price 2 days ago (€/MWh)" 
                - **price actual_72** - "Price 3 days ago (€/MWh)"
                - **price actual_168** - "Last week's price (€/MWh)"
        ''')

        
        # File uploader for CSV
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file:
            # Read uploaded CSV file
            df = pd.read_csv(uploaded_file)

            # Ensure the CSV file has the expected columns
            expected_columns = ['time', 'total load forecast', 'total load actual_168', 'total_non_renewable_168', 'price actual_48', 'price actual_72', 'price actual_168']
            if set(df.columns) == set(expected_columns):
                # Make predictions for the next 24 hours using XGBoost
                predicted_prices_24 = predict_with_xgboost(xgb, df.iloc[:, 1:])  # Exclude 'time' column

                # Create DataFrame with predictions
                predictions_df_24 = pd.DataFrame({'Time': df['time'], 'Predicted Price': predicted_prices_24})

                # Display the predictions DataFrame
                st.write(predictions_df_24)

                # Plotting the predictions
                plt.figure(figsize=(10, 6))
                plt.plot(predictions_df_24['Time'], predictions_df_24['Predicted Price'], label='Predicted Price')
                plt.xlabel('Time')
                plt.ylabel('Predicted Price (€/MWh)')
                plt.title('Predicted Energy Price for the Next 24 Hours')
                plt.xticks(rotation=45)
                plt.legend()
                st.pyplot(plt)
            else:
                st.write("Error: Uploaded CSV file does not contain the expected columns.")