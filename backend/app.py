
#Import necessary libraries
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

#Initialize the Flask app
app = Flask("SuperKart")

#Load the trained model
model = joblib.load("backend_files/superkart_model.joblib")

#define the route for the home page
@app.route("/")
def home():
    return "SuperKart API is running!"

#define the route for the predict page
@app.route("/v1/predict", methods=["POST"])
def predict():
    #Get the data from the request
    input_data = request.get_json()
    #convert the data to a pandas dataframe
    data = pd.DataFrame(input_data).
    #make a prediction
    prediction = model.predict(data).tolist()[0]
    return jsonify({"prediction": prediction})

#Define a route to process batch predictions
@app.route("/v1/predictbatch", methods=["POST"])
def predict_batch():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    #read the file
    data = pd.read_csv(file)
    predictions = model.predict(data).tolist()
    #create an output dictionary mapping each row index to its prediction
    #create a dictionary to store the predictions
    # Create an output dictionary mapping row index to predicted sales
    output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}
    return output_dict

#Run the app
if __name__ == "__main__":
    app.run(debug=True)

