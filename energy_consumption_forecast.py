import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


data = pd.read_csv('energy_consumption.csv')
print(data)

data['date'] = pd.to_datetime(data['date'])

data.set_index('date', inplace=True)

data['month'] = data.index.month
data['day'] = data.index.day
data['hour'] = data.index.hour


# Splitting the data into train and test sets
train, test = train_test_split(data, test_size=0.2, shuffle=False)


# Normalize the data
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)


# Split features and target
X_train, y_train = train_scaled[:, :-1], train_scaled[:, -1]
X_test, y_test = test_scaled[:, :-1], test_scaled[:, -1]


# Build the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


# Make predictions
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')



# Plot the actual vs predicted values
plt.figure(figsize=(14, 7))
plt.plot(test.index, y_test, label='Actual', color='blue')
plt.plot(test.index, predictions, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Energy Consumption')
plt.title('Energy Consumption Forecasting')
plt.legend()
plt.show()
