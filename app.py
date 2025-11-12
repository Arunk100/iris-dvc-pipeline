from flask import Flask, request, jsonify
import joblib, numpy as np, os

app = Flask(__name__)
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
model = joblib.load(MODEL_PATH)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status":"ok"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({'error': 'Missing features. Send JSON: {"features":[f1,f2,f3,f4]}' }), 400
    try:
        features = np.array(data['features']).reshape(1, -1)
        pred = model.predict(features)
        return jsonify({'prediction': int(pred[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
