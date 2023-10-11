import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset (replace 'data.csv' with your actual dataset)
df = pd.read_csv('data.csv')

# Assuming you have a 'features' column and a 'price' column, you can adjust as needed
X = df.drop(columns=['Price'])  # Input features
y = df['Price']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Create a Flask web application
app = Flask(__name__)

# Define a route to render the HTML form for user input
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = {
            'Avg. Area Income': float(request.form['avg_income']),
            'Avg. Area House Age': float(request.form['house_age']),
            'Avg. Area Number of Rooms': float(request.form['num_rooms']),
            'Avg. Area Number of Bedrooms': float(request.form['num_bedrooms']),
            'Area Population': float(request.form['population'])
        }

        # Convert user input dictionary to a DataFrame
        input_df = pd.DataFrame([user_input])

        # Make predictions using the trained model
        predicted_price = model.predict(input_df)
        predict = predicted_price
        return render_template('index.html', predict=predict[0])

if __name__ == '__main__':
    app.run(debug=True)
