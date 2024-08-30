import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def forecast(df):
    # Ensure 'ds' column is in datetime format
    df['ds'] = pd.to_datetime(df['ds'])

    # Create a Prophet model
    m = Prophet(changepoint_prior_scale=0.01)

    # Fit the model to the data
    m.fit(df)

    # Make future predictions
    future = m.make_future_dataframe(periods=500, freq='15T')
    forecast = m.predict(future)

    # Return the forecast DataFrame
    return forecast

def plot_time_series(df, forecast_data):
    # Plot the time series data
    fig, ax = plt.subplots()
    
    # Ensure that 'ds' is in datetime format
    df['ds'] = pd.to_datetime(df['ds'])
    forecast_data['ds'] = pd.to_datetime(forecast_data['ds'])

    ax.plot(df['ds'], df['y'], label='Actual')
    ax.plot(forecast_data['ds'], forecast_data['yhat'], label='Forecast', linestyle='dashed')
    ax.fill_between(forecast_data['ds'], forecast_data['yhat_lower'], forecast_data['yhat_upper'], color='gray', alpha=0.2, label='Uncertainty Interval')

    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()

    # Rotate the x-axis tick labels to be vertical
    ax.tick_params(axis='x', rotation=90)

    st.pyplot(fig)

def main():
    # Set Streamlit app title and description
    st.title('Time Series Forecasting App')
    st.write('Upload your CSV file to forecast data.')

    # File uploader for CSV
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the uploaded CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Display the uploaded data
        st.write('Uploaded Data:')
        st.dataframe(df)

        # Perform forecasting
        forecast_data = forecast(df)

        # Display the forecast output
        st.write('Forecast Data:')
        st.dataframe(forecast_data)

        # Plot the time series graph
        st.write('Time Series Graph:')
        plot_time_series(df, forecast_data)

if __name__ == '__main__':
    main()
