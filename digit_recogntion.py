#using logistic regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

# Load MNIST dataset
mnist = fetch_openml('mnist_784', parser='pandas')
X_digits, y_digits = mnist.data / 255.0, mnist.target.astype(int)

# Split the dataset
X_train_digits, X_test_digits, y_train_digits, y_test_digits = train_test_split(X_digits, y_digits, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model_digits = LogisticRegression(max_iter=100)
model_digits.fit(X_train_digits, y_train_digits)

# Make predictions
y_pred_digits = model_digits.predict(X_test_digits)

# Evaluate the model
accuracy_digits = accuracy_score(y_test_digits, y_pred_digits)
print(f'Digit Recognition Accuracy: {accuracy_digits}')

# Display actual and predicted digits side by side
num_display = 5  # Number of examples to display
plt.figure(figsize=(12, 3))
for i in range(num_display):
    actual_digit = X_test_digits.iloc[i].values.reshape(28, 28)
    predicted_digit = model_digits.predict([X_test_digits.iloc[i]])[0]
    plt.subplot(1, num_display, i + 1)
    plt.imshow(actual_digit, cmap='gray')
    plt.title(f'Actual: {y_test_digits.iloc[i]}, Predicted: {predicted_digit}')
    plt.axis('off')

plt.show()
