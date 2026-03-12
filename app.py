from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    population = int(request.form['population'])
    unemployment = int(request.form['unemployment'])
    poverty = int(request.form['poverty'])
    police = int(request.form['police'])

    data = np.array([[population, unemployment, poverty, police]])

    prediction = model.predict(data)

    return render_template('index.html', result=prediction[0])


if __name__ == "__main__":
    app.run(debug=True)