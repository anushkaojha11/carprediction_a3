import os
import time
import pickle
import numpy as np
import logging
import mlflow
import mlflow.pyfunc
import pandas as pd

logging.basicConfig(level=logging.INFO)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------
# Paths for scaler and encoders
# --------------------------
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

def get_X(max_power, mileage, year, brand):
    feature_names = ["max_power", "mileage", "year"]
    numeric_df = pd.DataFrame([[max_power, mileage, year]], columns=feature_names)
    numeric_scaled = scaler.transform(numeric_df)
    if hasattr(brand_encoder, "transform"):
        brand_encoded = brand_encoder.transform([brand]).reshape(-1, 1)
    else:
        brand_encoded = np.array([[brand_encoder.index(brand)]])
    X = np.hstack([numeric_scaled, brand_encoded]).astype("float64")
    return X

# --------------------------
# Load MLflow model
# --------------------------
def load_model():
    """
    Load MLflow model using credentials from environment.
    Retries up to 5 times in case of failure.
    """
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.ml.brain.cs.ait.ac.th/")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    # Set credentials in environment before setting tracking URI
    if username and password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        logging.info(f"Using MLflow credentials: {username}/*****")
    else:
        logging.warning("No MLflow credentials found in environment")

    mlflow.set_tracking_uri(mlflow_uri)

    run_id = os.getenv("RUN_ID")
    model_name = os.getenv("MODEL_NAME", "st126222-a3-model")
    model_uri = f"runs:/{run_id}/model" if run_id else f"models:/{model_name}/Production"

    for attempt in range(5):
        try:
            logging.info(f"Loading MLflow model from {model_uri} (attempt {attempt + 1}/5)")
            model = mlflow.pyfunc.load_model(model_uri)
            logging.info("✅ MLflow model loaded successfully")
            return model
        except mlflow.exceptions.MlflowException as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(3)

    raise RuntimeError(f"Failed to load MLflow model after 5 attempts. Tried URI: {model_uri}")

# Load MLflow model at import
try:
    mlflow_model = load_model()
except Exception as e:
    logging.error(f"MLflow model could not be loaded: {e}")
    mlflow_model = None  # Allow app to start but prediction will fail

# --------------------------
# Predict function
# --------------------------
def predict_selling_price(max_power, mileage, year, brand):
    model = load_model()
    X = get_X(max_power, mileage, year, brand)
    raw_pred = model.predict(X)[0]
    class_map = ["Cheap", "Average", "Expensive", "Very Expensive"]
    label = class_map[int(raw_pred)]
    return float(raw_pred), label



# # model.py — MLflow loader with explicit URI + clear logging + caching
# import os, time, pickle, logging, numpy as np, pandas as pd, traceback
# from functools import lru_cache
# import mlflow, mlflow.pyfunc

# logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
# log = logging.getLogger("model")

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# SCALER_PATH = os.path.join(BASE_DIR, "Model", "A3_prediction_scalar.model")
# LABEL_PATH  = os.path.join(BASE_DIR, "Model", "A3_brand_label.model")

# # --------------------------
# # Load preprocessing objects
# # --------------------------
# with open(SCALER_PATH, "rb") as f:
#     scaler = pickle.load(f)
# with open(LABEL_PATH, "rb") as f:
#     brand_encoder = pickle.load(f)

# if hasattr(brand_encoder, "classes_"):
#     brand_classes = brand_encoder.classes_.tolist()
# elif hasattr(brand_encoder, "categories_"):
#     brand_classes = brand_encoder.categories_[0].tolist()
# else:
#     brand_classes = list(brand_encoder)

# def get_X(max_power, mileage, year, brand):
#     feature_names = ["max_power", "mileage", "year"]
#     numeric_df = pd.DataFrame([[max_power, mileage, year]], columns=feature_names)
#     numeric_scaled = scaler.transform(numeric_df)
#     if hasattr(brand_encoder, "transform"):
#         brand_encoded = brand_encoder.transform([brand]).reshape(-1, 1)
#     else:
#         brand_encoded = np.array([[brand_encoder.index(brand)]])
#     X = np.hstack([numeric_scaled, brand_encoded]).astype("float64")
#     return X

# # --------------------------
# # MLflow model loader (env-driven)
# # --------------------------
# def _resolve_model_uri():
#     """
#     Require an explicit MODEL_URI or RUN_ID.
#     Refuse to fall back to a registry default to avoid loading an unintended model.
#     """
#     model_uri = os.getenv("MODEL_URI")
#     if model_uri:
#         return model_uri

#     run_id = os.getenv("RUN_ID")
#     if run_id:
#         return f"runs:/{run_id}/model"

#     raise RuntimeError(
#         "MODEL_URI or RUN_ID must be set; refusing to fall back to models:/<name>/1"
#     )

# def _configure_mlflow():
#     """Configure MLflow from environment without forcing a silent default."""
#     tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
#     if tracking_uri:
#         mlflow.set_tracking_uri(tracking_uri)
#     # Request timeout / TLS settings can be controlled by env variables
#     os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
#     # If testing with a self-signed cert, set in your .env:
#     # MLFLOW_TRACKING_INSECURE_TLS=true

# def _log_runtime_context(model_uri: str):
#     log.info("Resolved MODEL_URI=%r | TRACKING=%r", model_uri, mlflow.get_tracking_uri())
#     # Don’t print secrets; just indicate they’re present
#     if os.getenv("MLFLOW_TRACKING_USERNAME"):
#         log.info("MLFLOW_TRACKING_USERNAME is set")
#     if os.getenv("MLFLOW_TRACKING_INSECURE_TLS"):
#         log.info("MLFLOW_TRACKING_INSECURE_TLS is set")

# @lru_cache(maxsize=1)
# def load_model(max_retries: int = 3, delay: float = 2.0):
#     _configure_mlflow()
#     model_uri = _resolve_model_uri()
#     _log_runtime_context(model_uri)

#     last = None
#     for i in range(1, max_retries + 1):
#         try:
#             log.info("[MLflow] Loading model from %s (attempt %d/%d)", model_uri, i, max_retries)
#             model = mlflow.pyfunc.load_model(model_uri)
#             log.info("✅ Model loaded successfully")
#             return model
#         except Exception as e:
#             last = e
#             log.exception("❌ Load failed (attempt %d). type=%s msg=%s", i, type(e).__name__, e)
#             time.sleep(delay)

#     raise RuntimeError(
#         f"❌ Could not load MLflow model after {max_retries} attempts. "
#         f"MODEL_URI={model_uri!r}, TRACKING={mlflow.get_tracking_uri()!r}"
#     ) from last

# def predict_selling_price(max_power, mileage, year, brand):
#     model = load_model()
#     X = get_X(max_power, mileage, year, brand)
#     raw_pred = model.predict(X)[0]
#     class_map = ["Cheap", "Average", "Expensive", "Very Expensive"]
#     label = class_map[int(raw_pred)]
#     return float(raw_pred), label

# # # model.py  — drop-in replacement for the MLflow parts
# # import os, time, pickle, logging, numpy as np, pandas as pd
# # import mlflow, mlflow.pyfunc, traceback

# # logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
# # log = logging.getLogger("model")

# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # SCALER_PATH = os.path.join(BASE_DIR, "Model", "A3_prediction_scalar.model")
# # LABEL_PATH  = os.path.join(BASE_DIR, "Model", "A3_brand_label.model")

# # # --------------------------
# # # Load preprocessing objects
# # # --------------------------
# # with open(SCALER_PATH, "rb") as f:
# #     scaler = pickle.load(f)
# # with open(LABEL_PATH, "rb") as f:
# #     brand_encoder = pickle.load(f)

# # if hasattr(brand_encoder, "classes_"):
# #     brand_classes = brand_encoder.classes_.tolist()
# # elif hasattr(brand_encoder, "categories_"):
# #     brand_classes = brand_encoder.categories_[0].tolist()
# # else:
# #     brand_classes = list(brand_encoder)

# # def get_X(max_power, mileage, year, brand):
# #     feature_names = ["max_power", "mileage", "year"]
# #     numeric_df = pd.DataFrame([[max_power, mileage, year]], columns=feature_names)
# #     numeric_scaled = scaler.transform(numeric_df)
# #     if hasattr(brand_encoder, "transform"):
# #         brand_encoded = brand_encoder.transform([brand]).reshape(-1, 1)
# #     else:
# #         brand_encoded = np.array([[brand_encoder.index(brand)]])
# #     X = np.hstack([numeric_scaled, brand_encoded]).astype("float64")
# #     return X

# # # --------------------------
# # # MLflow model loader (env-driven)
# # # --------------------------
# # _model = None

# # def _resolve_model_uri():
# #     """Prefer explicit MODEL_URI; else RUN_ID; else registry name/stage/version."""
# #     # 1) Exact URI via env (compose recommended)
# #     model_uri = os.getenv("MODEL_URI")
# #     if model_uri:
# #         return model_uri

# #     # 2) Run-based (from .env RUN_ID)
# #     run_id = os.getenv("RUN_ID")
# #     if run_id:
# #         return f"runs:/{run_id}/model"

# #     # 3) Registry (fallback)
# #     name = os.getenv("MODEL_NAME", "st126222-a3-model")
# #     stage = os.getenv("MODEL_STAGE")  # e.g., Production
# #     version = os.getenv("MODEL_VERSION")  # e.g., 1
# #     if stage:
# #         return f"models:/{name}/{stage}"
# #     if version:
# #         return f"models:/{name}/{version}"
# #     # final fallback if nothing else set:
# #     return f"models:/{name}/1"

# # def _configure_mlflow():
# #     """Use standard env vars; do NOT embed creds in the URI."""
# #     tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.ml.brain.cs.ait.ac.th/")
# #     mlflow.set_tracking_uri(tracking_uri)
# #     # Timeouts & TLS
# #     os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
# #     # If your server uses a self-signed cert and you’re testing, you can set:
# #     # os.environ.setdefault("MLFLOW_TRACKING_INSECURE_TLS", "true")

# # def load_model(max_retries=3, delay=2):
# #     global _model
# #     if _model is not None:
# #         return _model

# #     _configure_mlflow()
# #     model_uri = _resolve_model_uri()
# #     last = None
# #     for i in range(1, max_retries + 1):
# #         try:
# #             log.info(f"[MLflow] Loading model from {model_uri} (attempt {i}/{max_retries})")
# #             _model = mlflow.pyfunc.load_model(model_uri)
# #             log.info("✅ Model loaded successfully")
# #             return _model
# #         except Exception as e:
# #             last = e
# #             log.warning(f"❌ Load failed (attempt {i}): {e}")
# #             traceback.print_exc()
# #             time.sleep(delay)

# #     # include context to expose root cause
# #     raise RuntimeError(
# #         f"❌ Could not load MLflow model after {max_retries} attempts. "
# #         f"MODEL_URI={model_uri!r}, MLFLOW_TRACKING_URI={os.getenv('MLFLOW_TRACKING_URI')!r}"
# #     ) from last

# # def predict_selling_price(max_power, mileage, year, brand):
# #     model = load_model()
# #     X = get_X(max_power, mileage, year, brand)
# #     raw_pred = model.predict(X)[0]
# #     class_map = ["Cheap", "Average", "Expensive", "Very Expensive"]
# #     label = class_map[int(raw_pred)]
# #     return float(raw_pred), label