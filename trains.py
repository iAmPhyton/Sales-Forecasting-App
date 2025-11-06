#core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#time-series forecasting
from prophet import Prophet
import statsmodels.api as sm 
import warnings
warnings.filterwarnings('ignore', category=FutureWarning) 

trains = pd.read_csv("train.csv")

#quick overview
print("Shape:", trains.shape)
trains.head() 

#basic dataset inspection
trains.info()
#checking for missing values
print("\nMissing values per column:\n", trains.isna().sum())
#checking date range
print("\nDate range:", trains['date'].min(), "to", trains['date'].max())
#summary stats
trains.describe() 

#focusing on one store and one product family for clarity
sample_store = trains[(trains['store_nbr'] == 1) & (trains['family'] == 'GROCERY I')]

#converting date to datetime
sample_store['date'] = pd.to_datetime(sample_store['date'])
sample_store = sample_store.sort_values('date')

#plot
plt.figure(figsize=(12,5))
plt.plot(sample_store['date'], sample_store['sales'], label='Daily Sales')
plt.title("Store 1 - GROCERY I Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()

#data cleaning and feature engineering
#checking missing values again
print("Missing values per column:\n", trains.isna().sum())

#dropping rows with missing sales or dates
trains.dropna(subset=['sales', 'date'], inplace=True)

#filling missing promotion data with 0 (if any)
trains['onpromotion'] = trains['onpromotion'].fillna(0)
print("After cleaning:", trains.shape) 

trains['date'] = pd.to_datetime(trains['date'])
trains = trains.sort_values('date') 

#creating additional time features
trains['year'] = trains['date'].dt.year
trains['month'] = trains['date'].dt.month
trains['day'] = trains['date'].dt.day
trains['dayofweek'] = trains['date'].dt.dayofweek
trains['is_weekend'] = trains['dayofweek'].isin([5, 6]).astype(int) 

#aggregating sales (daily level per store)
#focusing on one store-family pair
store_family = trains[(trains['store_nbr'] == 1) & (trains['family'] == 'GROCERY I')]

#aggregating daily sales
daily_sales = store_family.groupby('date')['sales'].sum().reset_index()
#renaming for Prophet
daily_sales.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)
daily_sales.head()

#visual check of clean data
plt.figure(figsize=(12,5))
plt.plot(daily_sales['ds'], daily_sales['y'], color='green')
plt.title("Cleaned Daily Sales Data for Prophet/ARIMA", fontweight='bold')
plt.xlabel("Date", fontweight='bold')
plt.ylabel("Sales",fontweight='bold')
plt.show() 

#model building - prophet, then ARIMA, then full comparison
#prophet forecasting model
#splitting the data
train = daily_sales.iloc[:-90]   #all except last 90 days
test = daily_sales.iloc[-90:]    #last 90 days
print("Train size:", train.shape)
print("Test size:", test.shape)

#training prophet
from prophet import Prophet

prophet_trains = trains[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})

#initialising model
prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='additive'
)
#Fit model
prophet_model.fit(train) 

#creating future dataframe
future = prophet_model.make_future_dataframe(periods=90)
#generating forecast
forecast = prophet_model.predict(future)
#viewing forecast tail
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail() 

#visualising the prophet forcast
prophet_model.plot(forecast)
plt.title("Prophet Forecast of Store Sales", fontweight='bold')
plt.xlabel("Date", fontweight='bold')
plt.ylabel("Predicted Sales", fontweight='bold')
plt.show() 

#comparing predictions vs actuals
#merging actual and predicted for test period
prophet_pred = forecast.set_index('ds').loc[test['ds']]
prophet_results = test.copy()
prophet_results['Predicted'] = prophet_pred['yhat'].values

#plotting actual vs predicted
plt.figure(figsize=(12,5))
plt.plot(prophet_results['ds'], prophet_results['y'], label='Actual', color='blue')
plt.plot(prophet_results['ds'], prophet_results['Predicted'], label='Prophet Forecast', color='orange')
plt.title("Prophet Model: Actual vs Predicted Sales") 
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show() 

#evaluating prophet perfomance
from sklearn.metrics import mean_squared_error, r2_score

mse_prophet = mean_squared_error(prophet_results['y'], prophet_results['Predicted'])
r2_prophet = r2_score(prophet_results['y'], prophet_results['Predicted'])
print(f"Prophet Model → MSE: {mse_prophet:.2f} | R²: {r2_prophet:.3f}") 

#training ARIMA (ARIMA requires a stationary series, but I’ll let the model auto-detect the best parameters)
from statsmodels.tsa.arima.model import ARIMA

#using only training data
y_train = train['y']
#fitting ARIMA model
arima_model = ARIMA(y_train, order=(5,1,2))  #(p,d,q)
arima_result = arima_model.fit()
#forecasting next 90 days
forecast_arima = arima_result.forecast(steps=90) 

#comparing with actuals
arima_results = test.copy()
arima_results['Predicted'] = forecast_arima.values

plt.figure(figsize=(12,5))
plt.plot(arima_results['ds'], arima_results['y'], label='Actual', color='blue')
plt.plot(arima_results['ds'], arima_results['Predicted'], label='ARIMA Forecast', color='green')
plt.title("ARIMA Model: Actual vs Predicted Sales",fontweight='bold')
plt.xlabel("Date",fontweight='bold')
plt.ylabel("Sales",fontweight='bold')
plt.legend()
plt.show() 

#evaluating ARIMA performance
mse_arima = mean_squared_error(arima_results['y'], arima_results['Predicted'])
r2_arima = r2_score(arima_results['y'], arima_results['Predicted'])
print(f"ARIMA Model → MSE: {mse_arima:.2f} | R²: {r2_arima:.3f}") 

#comparing prophet with ARIMA
comparison_trains = pd.DataFrame({
    'Model': ['Prophet', 'ARIMA'],
    'MSE': [mse_prophet, mse_arima],
    'R²': [r2_prophet, r2_arima]
})
print(comparison_trains) 

#combined visuals
plt.figure(figsize=(12,6))
plt.plot(test['ds'], test['y'], label='Actual', color='black', linewidth=2)
plt.plot(prophet_results['ds'], prophet_results['Predicted'], label='Prophet', color='orange', linestyle='--')
plt.plot(arima_results['ds'], arima_results['Predicted'], label='ARIMA', color='green', linestyle='--')
plt.title("Comparison: Prophet vs ARIMA Forecasts")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show() 

import joblib

#saving Prophet and ARIMA models
joblib.dump(prophet_model, "prophet_model.pkl")
joblib.dump(arima_result, "arima_model.pkl")
print("Models saved successfully!") 

import os
print(os.listdir()) 