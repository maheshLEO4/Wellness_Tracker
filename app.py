from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

pipeline = joblib.load("model/model.pkl")
feature_names = ['age', 'sleep_hours', 'phone_use_hours', 'water_liters', 'gender_Male', 'gender_Other']

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get('features')
    if not features:
        return jsonify({"error": "Missing 'features' in request"}), 400
    if len(features) != len(feature_names):
        return jsonify({"error": f"Expected {len(feature_names)} features, got {len(features)}"}), 400
    X = np.array(features).reshape(1, -1)
    prediction = pipeline.predict(X)[0]
    prediction_percentage = int(round(prediction))
    return jsonify({"wellness_index_percentage": prediction_percentage})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
