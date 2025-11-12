#streamlit App — Prophet, ARIMA & Auto ARIMA + Metrics Dashboard
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from prophet import Prophet
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

st.set_page_config(page_title="Sales Forecasting App", layout="wide")

#preparing the Dataset
@st.cache_data
def load_data():
    trains = pd.read_csv("train.csv")
    trains['date'] = pd.to_datetime(trains['date'])
    daily_sales = trains[(trains['store_nbr'] == 1) & (trains['family'] == 'GROCERY I')]
    daily_sales = daily_sales.groupby('date')['sales'].sum().reset_index()
    daily_sales.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)
    daily_sales = daily_sales.set_index('ds').asfreq('D')
    daily_sales['y'] = daily_sales['y'].ffill().bfill().fillna(0)
    daily_sales = daily_sales.replace([np.inf, -np.inf], np.nan).dropna()
    return daily_sales.reset_index()

daily_sales = load_data()

#loading Existing Models
def try_load_model(filename):
    try:
        return joblib.load(filename)
    except:
        return None

prophet_model = try_load_model("prophet_model.pkl")
arima_model = try_load_model("arima_model.pkl")
auto_arima_model = try_load_model("arima_best.pkl")

#Streamlit UI
st.title("Store Sales Forecasting App")
st.write("Forecast daily sales using Prophet, ARIMA, or Auto ARIMA models, and compare them interactively.")

tabs = st.tabs(["Forecasting", "Model Comparison"])

#TAB 1: Individual Forecasts
with tabs[0]:
    model_choice = st.selectbox("Choose a forecasting model:", ["Prophet", "ARIMA", "Auto ARIMA"])
    forecast_days = st.slider("Select forecast period (days):", 30, 180, 90)

    if st.button("Generate Forecast"):
        st.write(f"### Forecasting next {forecast_days} days using {model_choice}")
        y = daily_sales['y']

        if model_choice == "Prophet" and prophet_model is not None:
            future = prophet_model.make_future_dataframe(periods=forecast_days)
            forecast = prophet_model.predict(future)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(daily_sales['ds'], daily_sales['y'], label='Historical')
            ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='orange')
            ax.set_title("Prophet Forecast")
            ax.legend()
            st.pyplot(fig)

        elif model_choice == "ARIMA" and arima_model is not None:
            arima_forecast = arima_model.forecast(steps=forecast_days)
            future_dates = pd.date_range(start=daily_sales['ds'].iloc[-1],
                                         periods=forecast_days + 1, inclusive='right')

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(daily_sales['ds'], daily_sales['y'], label='Historical')
            ax.plot(future_dates, arima_forecast, label='Forecast', color='green')
            ax.set_title("ARIMA Forecast")
            ax.legend()
            st.pyplot(fig)

        elif model_choice == "Auto ARIMA":
            with st.spinner("Training Auto ARIMA model..."):
                train_size = int(len(y) * 0.8)
                train, test = y.iloc[:train_size], y.iloc[train_size:]
                train = train.replace([np.inf, -np.inf], np.nan).dropna()
                test = test.replace([np.inf, -np.inf], np.nan).dropna()

                auto_arima_model = pm.auto_arima(
                    train,
                    seasonal=True,
                    m=7,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    trace=False
                )

                forecast = auto_arima_model.predict(n_periods=len(test))
                forecast = forecast[:len(test)]

                r2 = r2_score(test, forecast)
                mse = mean_squared_error(test, forecast)

                st.success("Auto ARIMA model trained successfully!")
                st.write(f"**R² Score:** {r2:.4f}")
                st.write(f"**MSE:** {mse:.2f}")

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(test.index, test.values, label='Actual', color='black')
                ax.plot(test.index, forecast, label='Forecast', color='orange')
                ax.set_title("Auto ARIMA Forecast vs Actual")
                ax.legend()
                st.pyplot(fig)

                joblib.dump(auto_arima_model, "arima_best.pkl")
                st.info("Model saved as `arima_best.pkl`")

        else:
            st.warning("Model not found. Please train or load it first.")

#TAB 2: Model Comparison
with tabs[1]:
    st.subheader("Prophet vs ARIMA vs Auto ARIMA Forecast Comparison")

    try:
        forecast_days = 90
        y = daily_sales['y']
        train_size = int(len(y) * 0.8)
        train, test = y.iloc[:train_size], y.iloc[train_size:]

        metrics_data = []

        #prophet forecast
        future_p = prophet_model.make_future_dataframe(periods=len(test))
        prophet_forecast = prophet_model.predict(future_p)
        prophet_pred = prophet_forecast['yhat'].iloc[-len(test):]
        prophet_mse = mean_squared_error(test, prophet_pred)
        prophet_rmse = np.sqrt(prophet_mse)
        prophet_r2 = r2_score(test, prophet_pred)
        metrics_data.append(['Prophet', prophet_mse, prophet_rmse, prophet_r2])

        #ARIMA forecast
        arima_forecast = arima_model.forecast(steps=len(test))
        arima_mse = mean_squared_error(test, arima_forecast)
        arima_rmse = np.sqrt(arima_mse)
        arima_r2 = r2_score(test, arima_forecast)
        metrics_data.append(['ARIMA', arima_mse, arima_rmse, arima_r2])

        #auto ARIMA forecast
        if auto_arima_model is not None:
            auto_forecast = auto_arima_model.predict(n_periods=len(test))
            auto_mse = mean_squared_error(test, auto_forecast)
            auto_rmse = np.sqrt(auto_mse)
            auto_r2 = r2_score(test, auto_forecast)
            metrics_data.append(['Auto ARIMA', auto_mse, auto_rmse, auto_r2])

        #creating DataFrame
        metrics_trains = pd.DataFrame(metrics_data, columns=['Model', 'MSE', 'RMSE', 'R²'])

        #show metrics table
        st.write("### Model Performance Metrics")
        st.dataframe(metrics_trains.style.format({
            'MSE': '{:.2f}',
            'RMSE': '{:.2f}',
            'R²': '{:.4f}'
        }))

        #plotly bar chart for metrics
        fig = go.Figure()
        for metric in ['MSE', 'RMSE', 'R²']:
            fig.add_trace(go.Bar(x=metrics_trains['Model'],
                                 y=metrics_trains[metric],
                                 name=metric))
        fig.update_layout(barmode='group',
                          title='Model Performance Comparison',
                          xaxis_title='Model',
                          yaxis_title='Score',
                          template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Unable to generate comparison plot: {e}")

#footer
st.markdown("---")
st.markdown("Built with love by iamphyton | Prophet + ARIMA + Auto ARIMA Forecasting Dashboard") 