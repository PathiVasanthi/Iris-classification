import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

deployment = Flask(__name__)
model = pickle.load(open("ml_model.pkl", "rb"))

@deployment.route("/")
def Home():
    return render_template("index.html")

@deployment.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The flower species is {}".format(prediction))


if __name__ == "__main__":
    deployment.run(debug=True)