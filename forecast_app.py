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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor

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


def forecast_with_lstm(df):
    # Prepare data
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.set_index('ds')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['y'].values.reshape(-1, 1))

    # Set sequence length and create training data
    seq_len = 60  # Sequence length, can try 30, 60, 120 as well
    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)

    # Reshape data for LSTM model
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build improved LSTM model
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    # Compile the model with a lower learning rate
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Early stopping to avoid overfitting
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    # Fit the model with more epochs and early stopping
    model.fit(X, y, epochs=30, batch_size=32, callbacks=[early_stop])

    # Forecast future values
    last_seq = scaled_data[-seq_len:]
    future_predictions = []
    for _ in range(500):
        pred = model.predict(last_seq.reshape(1, seq_len, 1))
        future_predictions.append(pred[0, 0])
        last_seq = np.append(last_seq[1:], pred, axis=0)

    # Transform forecast back to original scale
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Prepare forecast DataFrame
    future_dates = pd.date_range(df.index[-1], periods=501, freq='15T')[1:]
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': future_predictions.flatten()})

    # Calculate and display the MSE
    train_predictions = model.predict(X)
    train_predictions = scaler.inverse_transform(train_predictions)
    mse = mean_squared_error(df['y'].values[-len(train_predictions):], train_predictions)
    st.write(f'Mean Squared Error (LSTM Model): {mse}')

    # Reset index of df for plotting
    df.reset_index(inplace=True)
    
    return forecast_df, model


def forecast_with_holt_winters(df):
    # Ensure 'ds' column is in datetime format
    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index('ds', inplace=True)

    # Fit the Holt-Winters Exponential Smoothing model
    model = ExponentialSmoothing(df['y'], trend='add', seasonal='add', seasonal_periods=96)  # assuming 96 periods for daily seasonality in 15-min intervals
    model_fit = model.fit(optimized=True)

    # Forecast the next 500 points
    forecast = model_fit.forecast(steps=500)
    
    # Prepare forecast DataFrame
    future_dates = pd.date_range(df.index[-1], periods=501, freq='15T')[1:]
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast.values})

    # Calculate Mean Squared Error on training data
    mse = mean_squared_error(df['y'], model_fit.fittedvalues)
    st.write(f'Mean Squared Error (Holt-Winters Model): {mse}')
    
    # Reset index of df for plotting
    df.reset_index(inplace=True)

    return forecast_df, model_fit


def create_lagged_features(df, lags=5):
    """Create lagged features for time series data."""
    df = df.copy()
    for lag in range(1, lags + 1):
        df[f'y_lag_{lag}'] = df['y'].shift(lag)
    df.dropna(inplace=True)
    return df

def forecast_with_random_forest(df):
    # Ensure 'ds' column is in datetime format and set as index
    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index('ds', inplace=True)

    # Create lagged features for Random Forest
    df_with_lags = create_lagged_features(df, lags=5)
    
    # Define features (lagged values) and target variable
    X = df_with_lags.drop(columns=['y'])
    y = df_with_lags['y']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on the training data for calculating MSE
    df_with_lags['yhat'] = model.predict(X)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y, df_with_lags['yhat'])
    st.write(f'Mean Squared Error (Random Forest Model): {mse}')

    # Forecast future values
    last_known_values = df.iloc[-5:]['y'].values.reshape(1, -1)
    future_predictions = []
    for _ in range(500):
        prediction = model.predict(last_known_values)
        future_predictions.append(prediction[0])
        last_known_values = np.roll(last_known_values, -1)
        last_known_values[0, -1] = prediction

    # Prepare forecast DataFrame
    future_dates = pd.date_range(df.index[-1], periods=501, freq='15T')[1:]
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': future_predictions})

    # Reset index of df for plotting
    df.reset_index(inplace=True)
    
    return forecast_df, model

def plot_interactive_random_forest(df, forecast_df):
    # Create an interactive Plotly plot for the Random Forest forecast
    fig = px.scatter(df, x='ds', y='y', labels={'ds': 'Date', 'y': 'Actual'}, title='Random Forest Forecast vs Actual')
    fig.add_traces(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Random Forest Forecast'))
    st.plotly_chart(fig)

def plot_interactive_holt_winters(df, forecast_df):
    # Create an interactive Plotly plot for the Holt-Winters forecast
    fig = px.scatter(df, x='ds', y='y', labels={'ds': 'Date', 'y': 'Actual'}, title='Holt-Winters Forecast vs Actual')
    fig.add_traces(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Holt-Winters Forecast'))
    st.plotly_chart(fig)







def plot_interactive_lstm(df, forecast_df):
    fig = px.scatter(df, x='ds', y='y', labels={'ds': 'Date', 'y': 'Actual'}, title='LSTM Forecast vs Actual')
    fig.add_traces(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='LSTM Forecast'))
    st.plotly_chart(fig)



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
    st.title('Time Series Forecasting App')
    st.write('Upload your CSV file to forecast data using Prophet, Regression, LSTM, Holt-Winters, or Random Forest.')

    # File uploader for CSV
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the uploaded CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Display the uploaded data
        st.write('Uploaded Data:')
        st.dataframe(df)

        # Model selection
        model_choice = st.selectbox('Choose a model for forecasting:', ['Prophet', 'Regression', 'LSTM', 'Holt-Winters', 'Random Forest'])

        if model_choice == 'Prophet':
            forecast_data, m = forecast_with_prophet(df)
            st.write('Prophet Forecast Data:')
            st.dataframe(forecast_data)
            st.write('Time Series Graph (Prophet):')
            plot_time_series(df, forecast_data)
            st.write('Prophet Forecast Components:')
            plot_components(m, forecast_data)
            st.write('Interactive Plotly Forecast Graph (Prophet):')
            plot_interactive(m, forecast_data)

        elif model_choice == 'Regression':
            forecast_data, model = forecast_with_regression(df)
            st.write('Regression Forecast Data:')
            st.dataframe(forecast_data)
            st.write('Interactive Plotly Time Series Graph (Regression):')
            plot_interactive_regression(forecast_data)

        elif model_choice == 'LSTM':
            forecast_data, lstm_model = forecast_with_lstm(df)
            st.write('LSTM Forecast Data:')
            st.dataframe(forecast_data)
            st.write('Interactive Plotly Time Series Graph (LSTM):')
            plot_interactive_lstm(df, forecast_data)

        elif model_choice == 'Holt-Winters':
            forecast_data, hw_model = forecast_with_holt_winters(df)
            st.write('Holt-Winters Forecast Data:')
            st.dataframe(forecast_data)
            st.write('Interactive Plotly Time Series Graph (Holt-Winters):')
            plot_interactive_holt_winters(df, forecast_data)

        elif model_choice == 'Random Forest':
            forecast_data, rf_model = forecast_with_random_forest(df)
            st.write('Random Forest Forecast Data:')
            st.dataframe(forecast_data)
            st.write('Interactive Plotly Time Series Graph (Random Forest):')
            plot_interactive_random_forest(df, forecast_data)

if __name__ == '__main__':
    main()








