import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template

# Load the iris dataset
iris = load_iris()
X = iris['data']
y = iris['target']

# Train a random forest classifier on the data
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)

# Define a function that takes in the four features and returns the predicted species
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    X_new = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = clf.predict(X_new)[0]
    species = iris['target_names'][prediction]
    image_path=f'static/images/{species}.jpg'
    return species ,image_path

# Set up the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()
    sepal_length = data['sepal_length']
    sepal_width = data['sepal_width']
    petal_length = data['petal_length']
    petal_width = data['petal_width']

    # Use the predict_species() function to make the prediction
    species, image_path = predict_species(sepal_length, sepal_width, petal_length, petal_width)

    # Return the result to the client
    return {'species': species,'image_path':image_path}

if __name__ == '__main__':
    app.run(debug=True)


