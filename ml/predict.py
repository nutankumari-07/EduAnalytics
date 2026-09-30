"""Model loading and inference used by the student dashboard, admin pages and recommendation engine."""
import json
import os

import joblib
import pandas as pd

from flask import current_app, has_app_context

from config import Config


def _cfg(key):
    """Paths come from the running app's config (so tests/alt configs are isolated), else the defaults."""
    return current_app.config[key] if has_app_context() else getattr(Config, key)


_cache = {"mtime": None, "bundle": None}


def model_available():
    return os.path.exists(_cfg("MODEL_PATH"))


def load_bundle(path=None):
    path = path or _cfg("MODEL_PATH")
    mtime = os.path.getmtime(path)
    if _cache.get("path") != path or _cache["bundle"] is None or _cache["mtime"] != mtime:
        _cache["bundle"] = joblib.load(path)
        _cache["mtime"], _cache["path"] = mtime, path
    return _cache["bundle"]


def load_metrics():
    path = _cfg("METRICS_PATH")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def predict(features):
    """features: dict with every name in bundle['features'] -> {'level', 'confidence', 'proba'}."""
    bundle = load_bundle()
    row = pd.DataFrame([[float(features.get(f, 0) or 0) for f in bundle["features"]]],
                       columns=bundle["features"])
    model = bundle["model"]
    proba = model.predict_proba(row)[0]
    classes = list(model.classes_)
    level = classes[int(proba.argmax())]
    order = bundle["labels"]
    return {"level": level, "confidence": float(proba.max()),
            "proba": {c: float(proba[classes.index(c)]) for c in order if c in classes}}


def explain(features, top=5):
    """Per-student factor analysis: importance-weighted distance from the cohort mean.

    Positive contribution = the value pushes the student towards a better level.
    """
    bundle = load_bundle()
    out = []
    for f in bundle["features"]:
        st = bundle["stats"][f]
        z = (float(features.get(f, 0) or 0) - st["mean"]) / st["std"]
        out.append({"feature": f, "value": float(features.get(f, 0) or 0), "cohort_mean": st["mean"],
                    "z": z, "contribution": z * bundle["importances"][f]})
    out.sort(key=lambda d: -abs(d["contribution"]))
    return out[:top]


def predict_levels(frame):
    """Batch prediction. frame: DataFrame with the model's feature columns -> list of predicted levels."""
    if frame is None or len(frame) == 0:
        return []
    bundle = load_bundle()
    return [str(x) for x in bundle["model"].predict(frame[bundle["features"]].astype(float))]
