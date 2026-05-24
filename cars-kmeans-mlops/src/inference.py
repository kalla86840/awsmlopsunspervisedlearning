import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def model_fn(model_dir):
    metadata_path = Path(model_dir) / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "model": joblib.load(Path(model_dir) / "model.joblib"),
        "metadata": metadata,
    }


def input_fn(request_body, request_content_type):
    content_type = request_content_type.split(";")[0].strip().lower()
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    payload = json.loads(request_body)
    instances = payload.get("instances")
    if instances is None:
        raise ValueError("Request JSON must contain an 'instances' array.")
    return instances


def predict_fn(input_data, model_bundle):
    model = model_bundle["model"]
    feature_columns = model_bundle.get("metadata", {}).get("feature_columns", [])

    if input_data and isinstance(input_data[0], dict):
        frame = pd.DataFrame(input_data)
        if feature_columns:
            frame = frame[feature_columns]
        clusters = model.predict(frame)
        distances = model.transform(frame).min(axis=1)
    else:
        array = np.asarray(input_data)
        clusters = model.predict(array)
        distances = model.transform(array).min(axis=1)

    return [
        {"cluster": int(cluster), "distance_to_centroid": float(distance)}
        for cluster, distance in zip(clusters, distances)
    ]


def output_fn(prediction, response_content_type):
    return json.dumps({"predictions": prediction}), "application/json"
