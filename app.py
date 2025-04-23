from sklearn.linear_model import LinearRegression
import numpy as np
import joblib

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 6, 9, 12, 15])

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'model.pkl')

print("Model trained and saved as model.pkl")
