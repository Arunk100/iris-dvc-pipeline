from fastapi.testclient import TestClient
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

# --- Create a single Fake Model ---
class FakeModel:
    def predict(self, data):
        # Always return 0 (for 'setosa')
        return [0]

def test_health_check():
    """
    Tests if the /health endpoint returns a 200 OK.
    """
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

def test_predict_endpoint(mocker): # <--- Add 'mocker' here
    """
    Tests the /predict endpoint by mocking the model load.
    """
    
    # --- MOCKING ---
    # Intercept the 'mlflow.sklearn.load_model' call
    # inside the 'lifespan' function.
    # Instead of erroring, make it return our FakeModel.
    mocker.patch(
        "mlflow.sklearn.load_model",
        return_value=FakeModel()
    )
    # --- END MOCKING ---

    test_data = {
        "sepal_length_cm": 5.1,
        "sepal_width_cm": 3.5,
        "petal_length_cm": 1.4,
        "petal_width_cm": 0.2
    }

    # Now, when we start the TestClient, the 'lifespan'
    # function will run, call our *mocked* load_model,
    # and set app.state.model = FakeModel(), just like we want.
    with TestClient(app) as client:
        response = client.post("/predict", json=test_data)
    
    assert response.status_code == 200
    assert response.json() == {"prediction": 0}
