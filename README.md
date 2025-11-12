# Sales Forecasting App: Prophet, ARIMA & Auto ARIMA
This project delivers an interactive sales forecasting web app built with Streamlit, combining Prophet, ARIMA, and Auto ARIMA models to predict store sales trends. It showcases end-to-end machine learning deployment, from data preprocessing and model training to real-time, user-controlled forecasting with interactive visualizations. The app helps businesses anticipate demand, optimize inventory, and plan future operations with confidence.

Project Overview:
- This app predicts future store sales using historical data and multiple time-series models.
- It includes features for interactive forecasting, model comparison, and metric evaluation (MSE, RMSE, R²).

Project Structure:
- train.csv                     # Dataset
- prophet_model.pkl             # Saved Prophet model
- arima_model.pkl               # Saved ARIMA model
- arima_best.pkl                # Saved Auto ARIMA model (tuned)
- trains.py                     # Streamlit application file
- requirements.txt              # Project dependencies
- README.md                     # Project documentation
- streamlit_train1.py              # Sales Forecasting Dashboard
- streamlit_trains_comparison.py   # Streamlit app: Prophet, ARIMA & Auto ARIMA + Comparison Dashboard
- streamlit_trains_metrics.py     # Streamlit App — Prophet, ARIMA & Auto ARIMA + Metrics Dashboard
- streamlit_trains.py           # Streamlit app with Prophet, ARIMA & Auto ARIMA

Features:
- Multi-model Forecasting: Choose between Prophet, ARIMA, or Auto ARIMA
- Interactive UI: Select forecast horizon (30–180 days) dynamically
- Automatic Model Tuning: Uses pmdarima’s auto_arima for hyperparameter optimization
- Forecast Visualization: View trends, future predictions, and seasonal components
- Model Comparison Tab: Evaluate models using MSE, RMSE, and R² metrics
- Interactive Bar Charts: Compare models visually using Plotly
- Modular Codebase: Easy to extend with new models or features
- Production Ready: Cache-friendly and fully deployable on Streamlit Cloud or Hugging Face Spaces

Models Used:
- Prophet	prophet	Handles seasonality, trends, and holidays for robust forecasting.
- ARIMA	statsmodels	Classical time-series model for autoregression and moving averages.
- Auto ARIMA	pmdarima	Automatically finds the best ARIMA parameters (p, d, q, P, D, Q).

Evaluation Metrics:
- MSE (Mean Squared Error):	Measures average squared difference between predicted and actual values.
- RMSE (Root Mean Squared Error):	Shows error magnitude in the same units as the data.
- R² (Coefficient of Determination):	Indicates how well the model explains variance in the data.

Future Improvements:
- Integrate LSTM / Transformer models for deep learning-based forecasting
- Include external regressors (e.g., holidays, promotions, weather data)
- Deploy with Docker or Streamlit Cloud for public access
- Add feature importance visualization for interpretability
- Enable user-uploaded datasets for general forecasting

Author:
- Chukwuemeka Eugene Obiyo
- Data Scientist
- praise609@gmail.com
