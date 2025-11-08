# Sales-Forecasting-App
An interactive Streamlit web app for forecasting store sales using two time-series models — Facebook Prophet and ARIMA — built with Python.
This project combines data cleaning, forecasting, model evaluation, and deployment into a single end-to-end machine learning workflow.

Features:
- Dual Forecasting Models: Prophet and ARIMA
- Interactive Streamlit Dashboard
- Dynamic Forecast Period (30–180 days)
- Plotly Visual Comparison: Prophet vs ARIMA
- Performance Metrics: RMSE, MAPE
- Downloadable Forecast Results (CSV)
- Optimized with Streamlit Caching

Tech Stack:
- Python (3.12+)
- Streamlit (App framework)
- Facebook Prophet (Forecasting)
- Statsmodels (ARIMA model)
- Pandas / Numpy (Data handling)
- Matplotlib / Plotly (Visualization)
- Joblib (Model persistence)

Install Dependencies:
- pip install -r requirements.txt

project structure:
- train.csv: dataset
- trains.py: model training script (Prophet + ARIMA)
- streamlit_trains.py: streamlit web app
- prophet_model.pkl: saved Prophet model
- arima_model.pkl: saved ARIMA model
- requirements.txt: dependencies list
- README.md: documentation

Run the Streamlit App:
- streamlit run streamlit_trains.py (this automatically opens (usually http://localhost:8501/) your already created streamlit app)

How It Works:
- Data Preparation:
- Loads daily sales data (train.csv)
- Aggregates by store and product family
- Renames columns for Prophet (ds, y)
- Model Training
- Trains Prophet and ARIMA models
- Saves models as .pkl files using joblib

App Features (streamlit_trains.py):
- Users select forecast horizon (30–180 days)
- Toggle between Prophet, ARIMA, or Comparison
- Displays plots and metrics (RMSE, MAPE)
- Allows CSV downloads of forecasts

requirements.txt:
- streamlit
- pandas
- numpy
- matplotlib
- plotly
- prophet
- statsmodels
- scikit-learn
- joblib

Future Enhancements:
- Feature Importance Visualization (e.g., impact of promotions or holidays)
- Multiple Store Selection
- Add XGBoost Regressor for Hybrid Forecasting
- Deploy to Streamlit Cloud or Hugging Face Spaces

Author:
- Eugene Phyton
- Machine Learning Engineer & Data Scientist
- praise609@gmail.com
