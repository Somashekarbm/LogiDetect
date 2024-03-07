#MNIST Digit Recognition Project

Introduction
This project focuses on developing models to recognize handwritten digits from the MNIST dataset using logistic regression and a multi-layer perceptron (MLP) classifier. The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0 to 9), making it a popular benchmark dataset for machine learning tasks.

File Descriptions
1. digit_recognition.py
Description: This Python script implements digit recognition using logistic regression.
Key Components:
Loads the MNIST dataset using fetch_openml from sklearn.datasets.
Splits the dataset into training and testing sets.
Trains a logistic regression model on the training data.
Evaluates the model's accuracy on the testing data.
Displays actual and predicted digits side by side.
2. mnsit(ANN).ipynb
Description: This Jupyter Notebook implements digit recognition using an MLP classifier.
Key Components:
Loads the MNIST dataset using fetch_openml from sklearn.datasets.
Splits the dataset into training and testing sets.
Builds and trains an MLP classifier with one hidden layer.
Evaluates the model's accuracy on the testing data.
Visualizes random images along with their predicted labels.

How to Run
Ensure that Python and required libraries (e.g., numpy, pandas, matplotlib, scikit-learn, opencv) are installed.
Run (1) to perform digit recognition using logistic regression.
Run (2) to perform digit recognition using an MLP classifier.


Results-

Logistic Regression:
Achieved accuracy: -
Displays actual and predicted digits to visually assess the model's performance.

MLP Classifier:
Achieved accuracy: 93.3%.
Visualizes random images along with their predicted labels to assess the model's performance.
Conclusion
Both logistic regression and MLP classifier models were trained and evaluated for digit recognition on the MNIST dataset. The accuracy and visualizations provide insights into the effectiveness of each approach in recognizing handwritten digits.
