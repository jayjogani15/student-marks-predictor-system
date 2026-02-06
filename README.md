# student-marks-predictor-system
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
from flask import Flask, request, render_template_string, jsonify
import os

# Train and save the model (inline) with error handling
def train_model():
    try:
        np.random.seed(42)
        data = {
            'previous_grades': np.random.uniform(50, 100, 1000),
            'attendance': np.random.uniform(60, 100, 1000),
            'study_hours': np.random.uniform(1, 10, 1000),
            'performance': []
        }
        for i in range(1000):
            perf = (0.5 * data['previous_grades'][i] + 0.3 * data['attendance'][i] + 0.2 * data['study_hours'][i] + np.random.normal(0, 5))
            data['performance'].append(min(100, max(0, perf)))
        df = pd.DataFrame(data)
        X = df[['previous_grades', 'attendance', 'study_hours']]
        y = df['performance']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("Model trained and saved successfully.")
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None

# Load or train model with error handling
model = None
if os.path.exists('model.pkl'):
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded from file.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = train_model()
else:
    model = train_model()

if model is None:
    raise RuntimeError("Failed to load or train the model.")

# Flask app (backend)
app = Flask(__name__)

# HTML template as string (frontend) - unchanged, but added error display
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            width: 400px;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        form {
            display: flex;
            flex-direction: column;
        }
        label {
            margin-top: 10px;
            font-weight: bold;
        }
        input {
            padding: 8px;
            margin-top: 5px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        button {
            margin-top: 20px;
            padding: 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            text-align: center;
            font-size: 18px;
            color: #333;
        }
        .error {
            color: red;
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Student Performance Prediction</h1>
        <form action="/predict" method="post">
            <label for="previous_grades">Previous Grades (0-100):</label>
            <input type="number" id="previous_grades" name="previous_grades" min="0" max="100" required>
            
            <label for="attendance">Attendance (%):</label>
            <input type="number" id="attendance" name="attendance" min="0" max="100" required>
            
            <label for="study_hours">Study Hours per Week:</label>
            <input type="number" id="study_hours" name="study_hours" min="0" max="20" required>
            
            <button type="submit">Predict Performance</button>
        </form>
        
        {% if prediction %}
            <div class="result">
                <h2>Predicted Final Grade: {{ prediction }}/100</h2>
            </div>
        {% endif %}
        
        {% if error %}
            <div class="error">
                <p>{{ error }}</p>
            </div>
        {% endif %}
    </div>
    <script>
        document.querySelector('form').addEventListener('submit', function(e) {
            const inputs = document.querySelectorAll('input[type="number"]');
            for (let input of inputs) {
                if (input.value < input.min || input.value > input.max) {
                    alert(`Value for ${input.name} must be between ${input.min} and ${input.max}`);
                    e.preventDefault();
                    return;
                }
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_template)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        previous_grades = float(request.form['previous_grades'])
        attendance = float(request.form['attendance'])
        study_hours = float(request.form['study_hours'])
        
        # Validate ranges server-side
        if not (0 <= previous_grades <= 100) or not (0 <= attendance <= 100) or not (0 <= study_hours <= 20):
            raise ValueError("Inputs out of range.")
        
        features = np.array([[previous_grades, attendance, study_hours]])
        prediction = model.predict(features)[0]
        prediction = max(0, min(100, prediction))  # Clamp to 0-100
        
        return render_template_string(html_template, prediction=round(prediction, 2))
    except (ValueError, KeyError) as e:
        error_msg = "Invalid input. Please enter valid numbers within the specified ranges."
        return render_template_string(html_template, error=error_msg)
    except Exception as e:
        error_msg = "An error occurred during prediction. Please try again."
        print(f"Prediction error: {e}")  # Log for debugging
        return render_template_string(html_template, error=error_msg)

if __name__ == '__main__':
    try:
        app.run(debug=True, port=5001)
    except OSError as e:
        print(f"Port 5001 is in use. Error: {e}. Try a different port.")
        # Optionally, app.run(debug=True, port=0) for random port
