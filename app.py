from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('xgboost_heart_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form
        features = [float(request.form[key]) for key in request.form]
        data = np.array([features])
        prediction = model.predict(data)[0]

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        return render_template('result.html', prediction=result)

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
