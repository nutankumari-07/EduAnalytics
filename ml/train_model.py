"""Train, compare and select the student performance-level model.

Run:  python -m ml.train_model            (from the project root, after seeding the database)

Pipeline (identical for every model)
  1. Build ONE dataset from the database: features from services.features (single definition) for every
     student with sufficient activity, target = student_outcomes.performance_level (Low / Medium / High).
  2. One stratified 80/20 split (random_state=42), one preprocessing step (StandardScaler) for all models.
  3. Each candidate: 5-fold stratified CV on the training part (weighted F1 + accuracy), then fit on the
     training part and evaluate on the held-out test part (accuracy, weighted precision/recall/F1,
     per-class metrics, confusion matrix).
  4. Selection is automatic: highest weighted F1; models within TIE_TOLERANCE of the best are tie-broken
     by cross-validation stability, scored as CV weighted-F1 mean minus its std (a conservative score, so a
     model that is merely "consistently worse" cannot win on low variance alone). Nothing is hard-coded.

Outputs
  data/student_performance.csv    the training dataset exported from the database
  models/performance_model.pkl    selected model bundle (joblib)
  models/model_metrics.json       everything shown on Admin -> ML Analytics
  models/model_comparison.csv     one row per candidate model
"""
import json
import os
import sqlite3
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import Config
from services.features import FEATURE_LABELS, FEATURES, LABELS, feature_frame

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TIE_TOLERANCE = 0.005          # weighted-F1 difference (0.5 pt) treated as a tie
MIN_ROWS, MIN_PER_CLASS = 60, 8

CRITERION = "Highest weighted F1-score (near-ties broken by cross-validation stability: CV F1 mean − std)"


class _XGBWrap(BaseEstimator, ClassifierMixin):
    """XGBoost needs integer class labels; encode/decode internally so it behaves like the others."""

    def fit(self, X, y):
        from xgboost import XGBClassifier
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        self._m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9,
                                eval_metric="mlogloss", random_state=SEED, n_jobs=1)
        self._m.fit(X, self._le.transform(y))
        self.feature_importances_ = self._m.feature_importances_
        return self

    def predict(self, X):
        return self._le.inverse_transform(self._m.predict(X))

    def predict_proba(self, X):
        return self._m.predict_proba(X)


def candidate_models():
    """(name, estimator, required). Same hyper-parameters every run; no per-model tuning."""
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, random_state=SEED), True),
        ("Decision Tree", DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=SEED), True),
        ("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=3,
                                                 random_state=SEED, n_jobs=1), True),
        ("KNN", KNeighborsClassifier(n_neighbors=15), True),
        ("SVM", SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED), True),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=SEED), True),
        ("Naive Bayes", GaussianNB(), True),
        ("Extra Trees", ExtraTreesClassifier(n_estimators=200, min_samples_leaf=3, random_state=SEED, n_jobs=1), False),
    ]
    try:
        import xgboost  # noqa: F401
        models.append(("XGBoost", _XGBWrap(), False))
    except ImportError:
        pass
    return models


def xgboost_available():
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


def build_dataset(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        feats = feature_frame(conn)
        out = pd.DataFrame([dict(r) for r in conn.execute(
            "SELECT o.user_id, u.student_code, o.performance_level FROM student_outcomes o "
            "JOIN users u ON u.id=o.user_id WHERE u.role='student' AND u.is_active=1")])
    finally:
        conn.close()
    if feats.empty or out.empty:
        return pd.DataFrame(columns=["user_id", "student_code"] + FEATURES + ["performance_level"])
    df = feats.merge(out, on="user_id", how="inner")
    return df[["user_id", "student_code"] + FEATURES + ["performance_level"]].round(3)


def _importance(fitted, X_te, y_te):
    est = fitted.steps[-1][1]
    if hasattr(est, "feature_importances_"):
        raw, method = np.asarray(est.feature_importances_, dtype=float), "native (tree feature importance)"
    elif hasattr(est, "coef_"):
        raw, method = np.abs(np.asarray(est.coef_)).mean(axis=0), "native (mean absolute coefficient on scaled features)"
    else:
        r = permutation_importance(fitted, X_te, y_te, scoring="f1_weighted", n_repeats=10,
                                   random_state=SEED, n_jobs=1)
        raw, method = np.clip(r.importances_mean, 0, None), "permutation importance on the test set"
    total = raw.sum() or 1.0
    return raw / total, method


def _evaluate(name, pipe, X_tr, X_te, y_tr, y_te, cv):
    cvres = cross_validate(clone(pipe), X_tr, y_tr, cv=cv, scoring={"f1": "f1_weighted", "acc": "accuracy"})
    fitted = clone(pipe).fit(X_tr, y_tr)
    pred = fitted.predict(X_te)
    prec, rec, f1, sup = precision_recall_fscore_support(y_te, pred, labels=LABELS, zero_division=0)
    return fitted, {
        "name": name,
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_te, pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_te, pred, average="weighted", zero_division=0)),
        "cv_f1_mean": float(cvres["test_f1"].mean()), "cv_f1_std": float(cvres["test_f1"].std()),
        "cv_accuracy_mean": float(cvres["test_acc"].mean()),
        "per_class": [{"label": l, "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                      for l, p, r, f, s in zip(LABELS, prec, rec, f1, sup)],
        "confusion_matrix": confusion_matrix(y_te, pred, labels=LABELS).tolist(),
    }


def select_best(results):
    """Automatic selection: max weighted F1; near-ties (<= TIE_TOLERANCE) -> most stable CV."""
    best_f1 = max(r["f1"] for r in results)
    tied = [r for r in results if best_f1 - r["f1"] <= TIE_TOLERANCE]
    tied.sort(key=lambda r: (-(r["cv_f1_mean"] - r["cv_f1_std"]), -r["f1"]))
    winner = tied[0]
    ranked = sorted(results, key=lambda r: -r["f1"])
    runner = next(r for r in ranked if r["name"] != winner["name"])
    gap = winner["f1"] - winner["cv_f1_mean"]
    if gap > 0.05 or winner["cv_f1_std"] > 0.05:
        stability = (f" Caution: its cross-validated weighted F1 is {winner['cv_f1_mean']*100:.2f}% "
                     f"(std {winner['cv_f1_std']*100:.2f} pts), noticeably below/less steady than the single "
                     f"test-set score, so the test figure may be optimistic.")
    else:
        stability = (f" Cross-validated weighted F1 is {winner['cv_f1_mean']*100:.2f}% "
                     f"(std {winner['cv_f1_std']*100:.2f} pts), consistent with the test-set score.")
    if len(tied) == 1:
        reason = (f"{winner['name']} achieved the highest weighted F1-score on the held-out test set "
                  f"({winner['f1']*100:.2f}%), ahead of the runner-up {runner['name']} "
                  f"({runner['f1']*100:.2f}%)." + stability)
    else:
        def cons(r):
            return f"{(r['cv_f1_mean'] - r['cv_f1_std'])*100:.1f}"
        others = ", ".join(f"{r['name']} (F1 {r['f1']*100:.2f}%, CV score {cons(r)})" for r in tied if r is not winner)
        reason = (f"{len(tied)} models were within {TIE_TOLERANCE*100:.1f} pt of the best weighted F1 "
                  f"({best_f1*100:.2f}%). Tie-break by cross-validation stability (CV weighted-F1 mean minus std): "
                  f"{winner['name']} (F1 {winner['f1']*100:.2f}%) scored {cons(winner)} "
                  f"(CV mean {winner['cv_f1_mean']*100:.2f}%, std {winner['cv_f1_std']*100:.2f} pts) versus {others}."
                  + stability)
    return winner, reason


def train(db_path=None, model_path=None, metrics_path=None, data_path=None, verbose=True):
    db_path = db_path or Config.DATABASE
    model_path = model_path or Config.MODEL_PATH
    metrics_path = metrics_path or Config.METRICS_PATH
    data_path = data_path or Config.DATASET_PATH
    for p in (model_path, metrics_path, data_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    df = build_dataset(db_path)
    counts = df["performance_level"].value_counts().reindex(LABELS).fillna(0).astype(int) if len(df) else None
    if len(df) < MIN_ROWS or counts is None or counts.min() < MIN_PER_CLASS:
        raise ValueError(
            f"Not enough labelled students to train: {len(df)} rows "
            f"(need at least {MIN_ROWS}, and {MIN_PER_CLASS} per class). "
            f"Class counts: {None if counts is None else counts.to_dict()}.")
    df.to_csv(data_path, index=False)

    X, y = df[FEATURES], df["performance_level"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    results, fitted_by_name, skipped = [], {}, []
    for name, est, required in candidate_models():
        try:
            fitted, res = _evaluate(name, make_pipeline(StandardScaler(), est), X_tr, X_te, y_tr, y_te, cv)
        except Exception as exc:                        # optional models may fail; required ones must not
            if required:
                raise
            skipped.append({"name": name, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        results.append(res)
        fitted_by_name[name] = fitted
        if verbose:
            print(f"  {name:20s} acc={res['accuracy']:.4f} f1={res['f1']:.4f} "
                  f"cv_f1={res['cv_f1_mean']:.4f}±{res['cv_f1_std']:.4f}")
    if not xgboost_available():
        skipped.append({"name": "XGBoost", "reason": "xgboost package is not installed in this environment"})

    winner, reason = select_best(results)
    model = fitted_by_name[winner["name"]]
    imp, imp_method = _importance(model, X_te, y_te)
    importances = sorted(({"feature": f, "label": FEATURE_LABELS[f], "importance": round(float(i), 4)}
                          for f, i in zip(FEATURES, imp)), key=lambda d: -d["importance"])
    stats = {f: {"mean": float(df[f].mean()), "std": float(df[f].std() or 1.0),
                 "min": float(df[f].min()), "max": float(df[f].max())} for f in FEATURES}
    trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def r4(v):
        return round(float(v), 4)
    comparison = [{"name": r["name"], "selected": r["name"] == winner["name"],
                   **{k: r4(r[k]) for k in ("accuracy", "precision", "recall", "f1", "cv_f1_mean", "cv_f1_std",
                                             "cv_accuracy_mean")}} for r in sorted(results, key=lambda r: -r["f1"])]
    metrics = {
        "model_name": winner["name"],
        "selection_criterion": CRITERION,
        "selection_reason": reason,
        "library": "scikit-learn " + sklearn.__version__,
        "target": "performance_level (Low / Medium / High) — overall student performance, not per-topic gaps",
        "trained_at": trained_at, "dataset_source": "student database (features) + student_outcomes (target)",
        "dataset_size": int(len(df)), "train_size": int(len(X_tr)), "test_size": int(len(X_te)),
        "cv_folds": CV_FOLDS, "split": f"stratified {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}, random_state={SEED}",
        "preprocessing": "StandardScaler (applied to every model)",
        "features": FEATURES, "feature_labels": FEATURE_LABELS, "labels": LABELS,
        "accuracy": r4(winner["accuracy"]), "precision": r4(winner["precision"]),
        "recall": r4(winner["recall"]), "f1": r4(winner["f1"]),
        "cv_f1_mean": r4(winner["cv_f1_mean"]), "cv_f1_std": r4(winner["cv_f1_std"]),
        "cv_accuracy_mean": r4(winner["cv_accuracy_mean"]),
        "per_class": [{**c, **{k: r4(c[k]) for k in ("precision", "recall", "f1")}} for c in winner["per_class"]],
        "confusion_matrix": winner["confusion_matrix"],
        "class_distribution": {k: int(v) for k, v in counts.items()},
        "feature_importance": importances, "feature_importance_method": imp_method,
        "model_comparison": comparison, "models_skipped": skipped,
        "min_activity_for_prediction": "≥1 quiz, ≥1 practice, ≥1 assessment, ≥3 scored topics, ≥6 attempts",
    }
    joblib.dump({"model": model, "model_name": winner["name"], "features": FEATURES, "labels": LABELS,
                 "stats": stats, "importances": {f: float(i) for f, i in zip(FEATURES, imp)},
                 "trained_at": trained_at}, model_path)
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    pd.DataFrame(comparison).to_csv(os.path.join(os.path.dirname(metrics_path), "model_comparison.csv"), index=False)
    if verbose:
        print(f"dataset: {len(df)} rows -> {data_path}   classes: {metrics['class_distribution']}")
        print(f"SELECTED: {winner['name']}  weighted-F1={winner['f1']:.4f}  cv-F1={winner['cv_f1_mean']:.4f}")
        print("reason:", reason)
        if skipped:
            print("skipped:", skipped)
    return metrics


if __name__ == "__main__":
    try:
        train()
    except ValueError as e:
        sys.exit(str(e))
