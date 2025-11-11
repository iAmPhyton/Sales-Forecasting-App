#store Sales Forecasting Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from prophet import Prophet
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

model_choice = st.selectbox("Choose a forecasting model:", ["Prophet", "ARIMA", "SARIMA", "XGBoost", "Hybrid"]) 

#utility Functions
@st.cache_resource
def load_models():
    prophet_model = joblib.load("prophet_model.pkl")
    arima_model = joblib.load("arima_model.pkl")
    return prophet_model, arima_model

@st.cache_data
def load_data():
    trains = pd.read_csv("train.csv")
    trains['date'] = pd.to_datetime(trains['date'])
    daily_sales = trains[(trains['store_nbr'] == 1) & (trains['family'] == 'GROCERY I')]
    daily_sales = daily_sales.groupby('date')['sales'].sum().reset_index()
    daily_sales.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)
    return daily_sales

#loading assets
prophet_model, arima_model = load_models()
daily_sales = load_data()

#streamlit UI
st.set_page_config(page_title="Sales Forecasting App", layout="wide")

st.title("Store Sales Forecasting Dashboard")
st.markdown("""
Welcome to the **Sales Forecasting App**!  
Predict future sales using **Prophet** and **ARIMA** models,  
compare their accuracy, and download the forecast results.
""")

forecast_days = st.slider("Select forecast period (days):", 30, 180, 90)
tab = st.radio("Choose view:", ["Prophet Forecast", "ARIMA Forecast", "Comparison Dashboard"])

#prophet Forecast
if tab == "Prophet Forecast":
    st.subheader(f"Prophet Forecast ({forecast_days} days)")
    future = prophet_model.make_future_dataframe(periods=forecast_days)
    prophet_forecast = prophet_model.predict(future)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily_sales['ds'], daily_sales['y'], label='Historical')
    ax.plot(prophet_forecast['ds'], prophet_forecast['yhat'], label='Forecast', color='orange')
    ax.set_title("Prophet Forecast")
    ax.legend()
    st.pyplot(fig)

    st.download_button(
        label="Download Prophet Forecast (CSV)",
        data=prophet_forecast.to_csv(index=False).encode('utf-8'),
        file_name='prophet_forecast.csv',
        mime='text/csv'
    )
#ARIMA Forecast
elif tab == "ARIMA Forecast":
    st.subheader(f"ARIMA Forecast ({forecast_days} days)")
    y = daily_sales['y']
    arima_forecast = arima_model.forecast(steps=forecast_days)
    future_dates = pd.date_range(start=daily_sales['ds'].iloc[-1], periods=forecast_days + 1, inclusive='right')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily_sales['ds'], daily_sales['y'], label='Historical')
    ax.plot(future_dates, arima_forecast, label='Forecast', color='green')
    ax.set_title("ARIMA Forecast")
    ax.legend()
    st.pyplot(fig)

    #combining into dataframe
    arima_trains = pd.DataFrame({'ds': future_dates, 'forecast': arima_forecast})
    st.download_button(
        label="Download ARIMA Forecast (CSV)",
        data=arima_trains.to_csv(index=False).encode('utf-8'),
        file_name='arima_forecast.csv',
        mime='text/csv'
    )
#comparison Dashboard
else:
    st.subheader(f"Prophet vs ARIMA Comparison ({forecast_days} days)")

    #generating forecasts
    future = prophet_model.make_future_dataframe(periods=forecast_days)
    prophet_forecast = prophet_model.predict(future)
    y = daily_sales['y']
    arima_forecast = arima_model.forecast(steps=forecast_days)
    future_dates = pd.date_range(start=daily_sales['ds'].iloc[-1], periods=forecast_days + 1, inclusive='right')

    #aligning data
    comparison_trains = pd.DataFrame({
        'Date': future_dates,
        'Prophet Forecast': prophet_forecast['yhat'].iloc[-forecast_days:].values,
        'ARIMA Forecast': arima_forecast
    })

    #plotting comparisons
    fig_compare = go.Figure()
    fig_compare.add_trace(go.Scatter(x=comparison_trains['Date'], y=comparison_trains['Prophet Forecast'],
                                     mode='lines', name='Prophet', line=dict(color='royalblue')))
    fig_compare.add_trace(go.Scatter(x=comparison_trains['Date'], y=comparison_trains['ARIMA Forecast'],
                                     mode='lines', name='ARIMA', line=dict(color='firebrick')))
    fig_compare.update_layout(
        title='Prophet vs ARIMA Forecast Comparison',
        xaxis_title='Date', yaxis_title='Sales',
        template='plotly_white'
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    #computing metrics using overlap of recent real data
    test_size = min(forecast_days, 30)
    y_true = daily_sales['y'].iloc[-test_size:]
    prophet_test_pred = prophet_forecast['yhat'].iloc[-test_size:]
    arima_test_pred = arima_model.forecast(steps=test_size)

    #metrics
    prophet_rmse = np.sqrt(mean_squared_error(y_true, prophet_test_pred))
    arima_rmse = np.sqrt(mean_squared_error(y_true, arima_test_pred))
    prophet_mape = mean_absolute_percentage_error(y_true, prophet_test_pred) * 100
    arima_mape = mean_absolute_percentage_error(y_true, arima_test_pred) * 100

    #displaying metrics
    st.markdown("Model Performance Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prophet RMSE", f"{prophet_rmse:,.2f}")
        st.metric("Prophet MAPE", f"{prophet_mape:.2f}%")
    with col2:
        st.metric("ARIMA RMSE", f"{arima_rmse:,.2f}")
        st.metric("ARIMA MAPE", f"{arima_mape:.2f}%")

    #downloading comparison
    st.download_button(
        label="Download Comparison (CSV)",
        data=comparison_trains.to_csv(index=False).encode('utf-8'),
        file_name='model_comparison.csv',
        mime='text/csv'
    ) 
    
model_files = {
    "Prophet": "prophet_model.pkl",
    "ARIMA": "arima_model.pkl",
    "SARIMA": "sarima_model.pkl",
    "XGBoost": "xgb_model.pkl",
    "Hybrid": "prophet_hybrid.pkl"
}

model = joblib.load(model_files[model_choice]) 