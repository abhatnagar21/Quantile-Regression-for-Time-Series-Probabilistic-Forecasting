# Import necessary libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import logging
import warnings
from neuralprophet import NeuralProphet, set_log_level
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Suppress warnings and set logging level for cleaner output
logging.getLogger('prophet').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv('bike_sharing_daily.csv')

# Convert the date column to datetime format for time series analysis
data["ds"] = pd.to_datetime(data["dteday"])

# Visualize the historical bike usage data
plt.figure(figsize=(10, 6))
plt.plot(data['ds'], data["cnt"], color='blue')
plt.xlabel("Date")
plt.ylabel("Bike Count")
plt.title("Daily Bike Rentals Over Time")
plt.grid(True)
plt.show()

# Prepare the data for NeuralProphet model
df = data[['ds', 'cnt']].copy()
df.columns = ['ds', 'y']  # Rename columns as required by NeuralProphet

# Define quantiles for prediction intervals
quantile_list = [0.05, 0.1, 0.5, 0.9, 0.95]

# Initialize the NeuralProphet model with desired configurations
m = NeuralProphet(
    quantiles=quantile_list,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False  # Set to False as daily seasonality might not be relevant
)

# Add country holidays (US) to capture holiday effects on bike rentals
m = m.add_country_holidays("US")

# Set plotting backend to Matplotlib
m.set_plotting_backend("matplotlib")

# Split the data into training and testing sets (80% train, 20% test)
df_train, df_test = m.split_df(df, valid_p=0.2)

# Fit the model on the training data and validate on the test data
metrics = m.fit(df_train, validation_df=df_test, progress="bar")
print("Training and Validation Metrics:\n", metrics.tail())

# Create a future dataframe to make predictions for the next 365 days
future = m.make_future_dataframe(df, periods=365)

# Generate the forecast based on the trained model
forecast = m.predict(future)
print("Forecast Data (Last 5 Days):\n", forecast.tail())

# Extract actual values from the test set
y_actual = df_test['y'].values

# Prepare the data for Random Forest model
# Note: Random Forest requires engineered features; for simplicity, using 'cnt' as a proxy
X = data[['temp', 'hum', 'windspeed']]  # Example features (you can add more)
y = data['cnt']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_rf_pred = rf_model.predict(X_test)

# Calculate metrics for Random Forest
rf_mae = mean_absolute_error(y_test, y_rf_pred)
rf_mse = mean_squared_error(y_test, y_rf_pred)
rf_rmse = np.sqrt(rf_mse)

print(f"Random Forest MAE: {rf_mae:.2f}, MSE: {rf_mse:.2f}, RMSE: {rf_rmse:.2f}")

# Get predictions from NeuralProphet
y_np_pred = forecast['yhat1'].values[-len(y_actual):]  # Get predictions corresponding to the test set

# Calculate accuracy metrics for NeuralProphet
mae_np = mean_absolute_error(y_actual, y_np_pred)
mse_np = mean_squared_error(y_actual, y_np_pred)
rmse_np = np.sqrt(mse_np)

print(f"\nNeuralProphet Accuracy Metrics:\nMAE: {mae_np:.2f}\nMSE: {mse_np:.2f}\nRMSE: {rmse_np:.2f}")

# Plot the forecast including historical data and prediction intervals
fig_forecast = m.plot(forecast, plotting_backend="matplotlib")
plt.title("Bike Rental Forecast with Prediction Intervals")
plt.show()

# Additional diagnostic plots (optional, useful for understanding the model's performance)
fig_components = m.plot_components(forecast, plotting_backend="matplotlib")
fig_parameters = m.plot_parameters(plotting_backend="matplotlib")
