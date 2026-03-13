from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model safely
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = pickle.load(open(model_path, "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    population = int(request.form["population"])
    unemployment = int(request.form["unemployment"])
    poverty = int(request.form["poverty"])
    police = int(request.form["police"])

    data = np.array([[population, unemployment, poverty, police]])

    prediction = model.predict(data)

    return render_template("index.html", result=prediction[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
