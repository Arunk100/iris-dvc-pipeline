import pytest
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

@pytest.fixture
def model():
    """Load the trained model"""
    return joblib.load('model.pkl')

@pytest.fixture
def test_data():
    """Load test data"""
    return pd.read_csv('data/iris.csv')

def test_model_loads(model):
    """Test if model loads successfully"""
    assert model is not None
    print("✅ Model loaded successfully")

def test_model_predictions(model, test_data):
    """Test if model can make predictions"""
    X = test_data.drop('target', axis=1)
    predictions = model.predict(X)
    assert len(predictions) == len(test_data)
    print(f"✅ Model made {len(predictions)} predictions")

def test_model_accuracy(model, test_data):
    """Test if model accuracy is above threshold"""
    X = test_data.drop('target', axis=1)
    y = test_data['target']
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    print(f"📊 Model accuracy: {accuracy:.2%}")
    assert accuracy > 0.90, f"Accuracy {accuracy:.2%} is below threshold 90%"

def test_prediction_range(model, test_data):
    """Test if predictions are in valid range"""
    X = test_data.drop('target', axis=1)
    predictions = model.predict(X)
    unique_predictions = set(predictions)
    assert unique_predictions.issubset({0, 1, 2}), f"Invalid predictions: {unique_predictions}"
    print(f"✅ All predictions in valid range: {unique_predictions}")

def test_data_shape(test_data):
    """Test if data has correct shape"""
    assert test_data.shape[1] == 5, f"Expected 5 columns, got {test_data.shape[1]}"
    print(f"✅ Data shape: {test_data.shape}")

def test_no_missing_values(test_data):
    """Test if data has no missing values"""
    missing_count = test_data.isnull().sum().sum()
    assert missing_count == 0, f"Found {missing_count} missing values"
    print("✅ No missing values found")
