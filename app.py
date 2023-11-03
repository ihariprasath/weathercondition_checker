from flask import Flask, render_template, request
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
model = keras.models.load_model('weather.h5')

# Initialize the LabelEncoder
le = LabelEncoder()

# Load your data and fit the LabelEncoder
import pandas as pd
data = pd.read_csv('weather_data.csv')
le.fit(data['Condition'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form['temp_c']),
                      float(request.form['temp_f']),
                      float(request.form['wind_mph']),
                      float(request.form['wind_kph']),
                      float(request.form['humidity'])]
        
        # Perform input validation
        for value in input_data:
            if not (isinstance(value, (int, float)) and value >= 0):
                raise ValueError("Invalid input. Please enter valid numeric values.")

        input_data = np.array([input_data])
        condition_code = model.predict(input_data)
        
        # Convert the condition code back to the original label using the fitted LabelEncoder
        condition = le.inverse_transform(condition_code.astype(int))

        return render_template('result.html', condition=condition[0])
    except Exception as e:
        error_message = str(e) if str(e) != "" else "An error occurred. Please check your input."
        return render_template('result.html', condition=error_message)

if __name__ == '__main__':
    app.run(debug=True)
