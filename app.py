import numpy as np
import pickle
import tensorflow as tf
from flask import Flask, render_template, request
from keras.models import load_model 
from keras.preprocessing.text import Tokenizer # Importing Keras model loading function

# Initialize the Flask App
app = Flask(__name__)

# Load the trained Keras model
model = load_model('model.h5') 
tokenizer = pickle.load(open("tokenizer.pkl", "rb")) # Make sure this file is in your project directory

# Default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')


# To use the predict button in our web-app
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the features from the form
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        
        # Make prediction using the loaded model
        prediction = model.predict(final_features)
        
        # For binary classification, round the result to get either 0 or 1
        output = round(prediction[0][0], 2)  # Assuming output is a single value prediction
        
        # You can adjust the output message based on your model's output
        return render_template('index.html', prediction_text=f'CO2 Emission of the vehicle is: {output}')
    
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True)