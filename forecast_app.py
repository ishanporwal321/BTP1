import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px

def forecast_with_prophet(df):
    # Ensure 'ds' column is in datetime format
    df['ds'] = pd.to_datetime(df['ds'])

    # Create a Prophet model
    m = Prophet(changepoint_prior_scale=0.01)

    # Fit the model to the data
    m.fit(df)

    # Make future predictions
    future = m.make_future_dataframe(periods=500, freq='15T')
    forecast = m.predict(future)

    # Calculate MSE on the historical data
    forecast_original = forecast[forecast['ds'].isin(df['ds'])]
    mse = mean_squared_error(df['y'], forecast_original['yhat'])
    st.write(f'Mean Squared Error (Prophet Model): {mse}')

    # Return the forecast DataFrame and model
    return forecast, m

def forecast_with_regression(df):
    # Ensure 'ds' column is in datetime format
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Convert 'ds' to numeric format for regression
    df['timestamp'] = df['ds'].map(pd.Timestamp.timestamp)
    
    # Split the data into training and testing sets
    X = df[['timestamp']]
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on the training set
    df['yhat'] = model.predict(X)
    
    # Create future timestamps for the next 500 data points
    last_timestamp = df['timestamp'].max()
    future_timestamps = pd.date_range(start=df['ds'].max(), periods=501, freq='15T')[1:]
    future_df = pd.DataFrame(future_timestamps, columns=['ds'])
    future_df['timestamp'] = future_df['ds'].map(pd.Timestamp.timestamp)
    
    # Predict future values using the regression model
    future_df['yhat'] = model.predict(future_df[['timestamp']])
    
    # Concatenate the original and future data
    full_forecast_df = pd.concat([df[['ds', 'y', 'yhat']], future_df[['ds', 'yhat']]], ignore_index=True)
    
    # Calculate and display the mean squared error for the training data
    mse = mean_squared_error(y, df['yhat'])
    st.write(f'Mean Squared Error (Regression Model): {mse}')
    
    return full_forecast_df, model

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

def plot_components(m, forecast_data):
    # Plot the forecast components
    fig2 = m.plot_components(forecast_data)
    st.pyplot(fig2)

def plot_interactive(m, forecast_data):
    # Create interactive plots using Plotly
    fig = plot_plotly(m, forecast_data)
    st.plotly_chart(fig)

    fig_components = plot_components_plotly(m, forecast_data)
    st.plotly_chart(fig_components)

def plot_interactive_regression(df):
    # Create an interactive Plotly plot for the regression model
    fig = px.scatter(df, x='ds', y='y', labels={'ds': 'Date', 'y': 'Actual'}, title='Regression Forecast vs Actual')
    fig.add_traces(go.Scatter(x=df['ds'], y=df['yhat'], mode='lines', name='Regression Forecast'))
    st.plotly_chart(fig)

def main():
    # Set Streamlit app title and description
    st.title('Time Series Forecasting App')
    st.write('Upload your CSV file to forecast data using Prophet or Regression.')

    # File uploader for CSV
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the uploaded CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Display the uploaded data
        st.write('Uploaded Data:')
        st.dataframe(df)

        # Model selection
        model_choice = st.selectbox('Choose a model for forecasting:', ['Prophet', 'Regression'])

        if model_choice == 'Prophet':
            # Perform forecasting with Prophet
            forecast_data, m = forecast_with_prophet(df)

            # Display the forecast output
            st.write('Prophet Forecast Data:')
            st.dataframe(forecast_data)

            # Plot the time series graph
            st.write('Time Series Graph (Prophet):')
            plot_time_series(df, forecast_data)

            # Plot the forecast components
            st.write('Prophet Forecast Components:')
            plot_components(m, forecast_data)

            # Plot interactive plots
            st.write('Interactive Plotly Forecast Graph (Prophet):')
            plot_interactive(m, forecast_data)

        elif model_choice == 'Regression':
            # Perform forecasting with Regression
            forecast_data, model = forecast_with_regression(df)

            # Display the forecast output
            st.write('Regression Forecast Data:')
            st.dataframe(forecast_data)

            # Plot the time series graph
            st.write('Interactive Plotly Time Series Graph (Regression):')
            plot_interactive_regression(forecast_data)

if __name__ == '__main__':
    main()
