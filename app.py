from flask import Flask, request, render_template
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve form data
            age = float(request.form['age'])
            salary = float(request.form['salary'])

            # Scale the input
            input_data = scaler.transform([[age, salary]])

            # Make prediction
            prediction = model.predict(input_data)
            result = "Will Purchase" if prediction[0] == 1 else "Will Not Purchase"

        except Exception as e:
            result = f"Error: {e}"

        return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == '__main__':
    app.run(debug=True)
