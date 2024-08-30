from fbprophet import Prophet
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Load data
# Assuming you have a CSV file with columns 'ds' (date) and 'y' (daily visits)
df = pd.read_csv('website_traffic.csv')

# Convert 'ds' column to datetime
df['ds'] = pd.to_datetime(df['ds'])

# Ensure there are no missing values
df = df.dropna()

# Initialize Prophet
model = Prophet(daily_seasonality=True)

# Fit the model to your data
model.fit(df)

# Create a DataFrame for future dates (forecast for the next 180 days)
future = model.make_future_dataframe(periods=180)

# Predict future traffic
forecast = model.predict(future)

# Plot the forecast
model.plot(forecast)
model.plot_components(forecast)

# Evaluate the model (Optional)
# Split the data into training and testing sets
train = df.iloc[:-180]  # Use all but the last 180 days for training
test = df.iloc[-180:]   # Use the last 180 days for testing

# Fit the model on the training set
model.fit(train)

# Create a future DataFrame for the test period
future_test = model.make_future_dataframe(periods=180)

# Predict the test period
forecast_test = model.predict(future_test)

# Calculate Mean Absolute Error (MAE)
y_true = test['y'].values
y_pred = forecast_test['yhat'][-180:].values
mae = mean_absolute_error(y_true, y_pred)

print(f'Mean Absolute Error: {mae}')
