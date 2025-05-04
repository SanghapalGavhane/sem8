import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
df = pd.read_csv("BostonHousing.csv")

# Drop null values if any
df.dropna(inplace=True)

# Heatmap of correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Prepare features and target
X = df.drop('medv', axis=1)
Y = df['medv']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

# Define model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model and save training history
history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2)

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Model evaluation - Train set
y_train_predict = model.predict(X_train)
rmse = np.sqrt(mean_squared_error(Y_train, y_train_predict))
r2 = r2_score(Y_train, y_train_predict)
print("Train set: RMSE =", rmse, " R2 =", r2)

# Model evaluation - Test set
y_test_predict = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(Y_test, y_test_predict))
r2 = r2_score(Y_test, y_test_predict)
print("Test set: RMSE =", rmse, " R2 =", r2)

# Plot predicted vs actual for test set
plt.figure(figsize=(8, 6))
plt.scatter(Y_test, y_test_predict, alpha=0.7)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], '--r')
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.title('Actual vs Predicted MEDV (Test Set)')
plt.grid(True)
plt.show()

# Plot residuals
residuals = Y_test - y_test_predict.flatten()
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution (Test Set)')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
