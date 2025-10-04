import os
import time
import pickle
import logging
import numpy as np
import mlflow
import mlflow.pyfunc
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("model")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, "Model", "A3_prediction_scalar.model")
LABEL_PATH  = os.path.join(BASE_DIR, "Model", "A3_brand_label.model")

# --------------------------
# Load preprocessing objects
# --------------------------
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(LABEL_PATH, "rb") as f:
    brand_encoder = pickle.load(f)

if hasattr(brand_encoder, "classes_"):
    brand_classes = brand_encoder.classes_.tolist()
elif hasattr(brand_encoder, "categories_"):
    brand_classes = brand_encoder.categories_[0].tolist()
else:
    brand_classes = list(brand_encoder)

# --------------------------
# Feature builder
# --------------------------
def get_X(max_power, mileage, year, brand):
    # match feature order used in training
    feature_names = ["max_power", "mileage", "year"]

    numeric_df = pd.DataFrame([[max_power, mileage, year]], columns=feature_names)

    # scale numeric features
    numeric_scaled = scaler.transform(numeric_df)

    # encode brand
    if hasattr(brand_encoder, "transform"):
        brand_encoded = brand_encoder.transform([brand]).reshape(-1, 1)
    else:
        # brand_encoder is just a list of classes
        brand_encoded = np.array([[brand_encoder.index(brand)]])
    
    # combine all
    X = np.hstack([numeric_scaled, brand_encoded]).astype("float64")
    return X

# --------------------------
# MLflow model loader (using version=1)
# --------------------------
_model = None

def load_model():
    global _model
    if _model is not None:
        return _model

    mlflow.set_tracking_uri("https://admin:password@mlflow.ml.brain.cs.ait.ac.th")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "20")

    MODEL_NAME = "st126222-a3-model"
    VERSION = 1  # <-- Force using version 1
    model_uri = f"models:/{MODEL_NAME}/{VERSION}"

    for attempt in range(3):
        try:
            log.info(f"Loading model from {model_uri} (attempt {attempt+1}/3)")
            _model = mlflow.pyfunc.load_model(model_uri)
            log.info("✅ Model loaded successfully")
            return _model
        except Exception as e:
            log.warning(f"❌ Failed to load model: {e}")
            time.sleep(2)

    raise RuntimeError("❌ Could not load MLflow model after 3 attempts.")

# --------------------------
# Prediction function
# --------------------------
def predict_selling_price(max_power, mileage, year, brand):
    model = load_model()
    X = get_X(max_power, mileage, year, brand)
    raw_pred = model.predict(X)[0]
    class_map = ["Cheap", "Average", "Expensive", "Very Expensive"]
    label = class_map[int(raw_pred)]
    return float(raw_pred), label