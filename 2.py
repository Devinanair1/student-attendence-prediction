import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Read in data
data = pd.read_csv(r"C:\Users\bhasy\Downloads\students - Sheet1.csv")

# Split data into features and target
X = data.drop(['attendance', 'performance'], axis=1)
y = data['attendance']

# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate results
score = model.score(X_test, y_test)
print('R2 score:', score)
