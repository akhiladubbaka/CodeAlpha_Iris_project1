from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('iris_model.pkl')
target_names = ['Setosa', 'Versicolor', 'Virginica']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[x]) for x in ['sl', 'sw', 'pl', 'pw']]
        prediction = model.predict([features])
        species = target_names[prediction[0]]
        return render_template('index.html', prediction_text=f"Predicted Iris Species: {species}")
    except Exception as e:
        print("error",e)
        return render_template('index.html', prediction_text="enter valid values.")

if __name__ == '__main__':
    app.run(debug=True)
