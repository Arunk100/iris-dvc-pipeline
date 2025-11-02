import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- Model URI ---
# !!! IMPORTANT !!!
# I have updated the bucket path based on your input.
# You MUST replace '...' with the full path to your model:
# e.g., /mlflow-artifacts/[RUN_ID]/artifacts/model
MODEL_URI = "gs://week22_dvc_iris_bucket/..."

# --- Lifespan Event ---
# This code will run when the API server starts up.
# It loads the model and stores it in app.state.
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading model from {MODEL_URI}...")
    try:
        app.state.model = mlflow.sklearn.load_model(MODEL_URI)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR: Model failed to load: {e}")
        app.state.model = None
    yield
    # (You could add cleanup code here if needed)

# --- FastAPI App ---
# We pass the 'lifespan' function to FastAPI
app = FastAPI(title="IRIS Model API", version="1.0.0", lifespan=lifespan)

# --- Input Schema ---
class IrisData(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float

# --- Endpoints ---
@app.get("/health")
def health():
    """Health check endpoint for Kubernetes."""
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: IrisData):
    """Prediction endpoint."""
    
    # Check if the model loaded correctly
    if app.state.model is None:
        return {"error": "Model is not loaded."}

    # Get the model from the app's state
    model = app.state.model

    sample = pd.DataFrame([{
        "sepal length (cm)": data.sepal_length_cm,
        "sepal width (cm)": data.sepal_width_cm,
        "petal length (cm)": data.petal_length_cm,
        "petal width (cm)": data.petal_width_cm
    }])
    
    prediction = model.predict(sample)
    return {"prediction": int(prediction[0])}

# This part is for running locally (e.g., python app.py)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
