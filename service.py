import flask
import joblib
import numpy as np
from flask import jsonify

classifier = joblib.load('classifier.joblib')

app = flask.Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if flask.request.method == 'POST':
        data = flask.request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        print(f"Features: {features}")
        prediction = classifier.predict(features)
        return jsonify(prediction=int(prediction[0]))


@app.route('/health', methods=['GET'])
def health():
    return flask.jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)