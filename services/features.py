"""The ONE feature definition used everywhere: training, student pages, admin pages.

Features are computed from the live database (never imputed with placeholder numbers).
A student only gets an ML prediction once they have enough real activity
(see ACTIVITY_REQUIREMENTS); until then features_for_user() reports what is still missing.
"""
import pandas as pd

FEATURES = [
    "attendance_percentage", "study_hours_per_week", "previous_gpa", "quiz_avg",
    "practice_accuracy", "assessment_avg", "avg_topic_score", "material_completion_pct",
]
FEATURE_LABELS = {
    "attendance_percentage": "Attendance %",
    "study_hours_per_week": "Study hours / week",
    "previous_gpa": "Previous GPA (out of 10)",
    "quiz_avg": "Quiz average %",
    "practice_accuracy": "Practice accuracy %",
    "assessment_avg": "Assessment average %",
    "avg_topic_score": "Average topic score %",
    "material_completion_pct": "Learning-material completion %",
}
LABELS = ["Low", "Medium", "High"]

# A prediction is only shown once a student has completed their profile and has this much real activity.
ACTIVITY_REQUIREMENTS = {"quiz": 1, "practice": 1, "assessment": 1, "topics_scored": 3, "total_attempts": 6}

_ACTIVITY_SQL = """
SELECT u.id AS user_id,
       SUM(CASE WHEN a.kind='quiz' THEN 1 ELSE 0 END)       AS n_quiz,
       SUM(CASE WHEN a.kind='practice' THEN 1 ELSE 0 END)   AS n_practice,
       SUM(CASE WHEN a.kind='assessment' THEN 1 ELSE 0 END) AS n_assessment,
       COUNT(a.id)                                          AS n_total,
       AVG(CASE WHEN a.kind='quiz' THEN a.percentage END)        AS quiz_avg,
       AVG(CASE WHEN a.kind='practice' THEN a.accuracy END)      AS practice_accuracy,
       AVG(CASE WHEN a.kind='assessment' THEN a.percentage END)  AS assessment_avg
FROM users u LEFT JOIN attempts a ON a.user_id=u.id AND a.status='submitted'
WHERE u.role='student' {where}
GROUP BY u.id
"""
_TOPIC_SQL = """
SELECT user_id, AVG(score) AS avg_topic_score, COUNT(score) AS n_topics,
       100.0*SUM(materials_done)/NULLIF(SUM(materials_total),0) AS material_completion_pct
FROM student_progress {where} GROUP BY user_id
"""


def _frame(db, user_id=None):
    where_u = "AND u.id = ?" if user_id else ""
    args = (user_id,) if user_id else ()
    act = pd.DataFrame([dict(r) for r in db.execute(_ACTIVITY_SQL.format(where=where_u), args).fetchall()])
    if act.empty:
        return act
    where_t = "WHERE user_id = ?" if user_id else ""
    top = pd.DataFrame([dict(r) for r in db.execute(_TOPIC_SQL.format(where=where_t), args).fetchall()])
    prof = pd.DataFrame([dict(r) for r in db.execute(
        "SELECT id AS user_id, attendance_percentage, study_hours_per_week, previous_gpa "
        "FROM users WHERE role='student'" + (" AND id = ?" if user_id else ""), args).fetchall()])
    df = act.merge(prof, on="user_id", how="left")
    if top.empty:
        top = pd.DataFrame({"user_id": [], "avg_topic_score": [], "n_topics": [], "material_completion_pct": []})
    df = df.merge(top, on="user_id", how="left")
    df["n_topics"] = df["n_topics"].fillna(0)
    R = ACTIVITY_REQUIREMENTS
    df["profile_ok"] = df[["attendance_percentage", "study_hours_per_week", "previous_gpa"]].notna().all(axis=1)
    df["sufficient"] = (df["profile_ok"] & (df["n_quiz"] >= R["quiz"]) & (df["n_practice"] >= R["practice"]) &
                        (df["n_assessment"] >= R["assessment"]) & (df["n_topics"] >= R["topics_scored"]) &
                        (df["n_total"] >= R["total_attempts"]))
    return df


def feature_frame(db):
    """All students with sufficient activity -> DataFrame[user_id + FEATURES]."""
    df = _frame(db)
    if df.empty:
        return pd.DataFrame(columns=["user_id"] + FEATURES)
    return df[df["sufficient"]][["user_id"] + FEATURES].reset_index(drop=True)


def activity_status(db, user_id):
    """Progress towards the activity needed for a prediction (safe to show to a student)."""
    df = _frame(db, user_id)
    R = ACTIVITY_REQUIREMENTS
    if df.empty:
        counts = {"quiz": 0, "practice": 0, "assessment": 0, "topics_scored": 0, "total_attempts": 0}
        profile_ok = False
    else:
        r = df.iloc[0]
        counts = {"quiz": int(r.n_quiz), "practice": int(r.n_practice), "assessment": int(r.n_assessment),
                  "topics_scored": int(r.n_topics), "total_attempts": int(r.n_total)}
        profile_ok = bool(r.profile_ok)
    missing = []
    if not profile_ok:
        missing.append("your profile details (attendance %, study hours, previous GPA)")
    labels = {"quiz": "quiz attempt(s)", "practice": "practice set(s)", "assessment": "timed assessment(s)",
              "topics_scored": "topic(s) with a score", "total_attempts": "total submitted attempt(s)"}
    for k, need in R.items():
        if counts[k] < need:
            missing.append(f"{need - counts[k]} more {labels[k]}")
    return {"sufficient": not missing, "profile_ok": profile_ok, "counts": counts, "required": dict(R), "missing": missing}


def features_for_user(db, user_id):
    """Returns (features_dict or None, activity_status). None => not enough activity, no prediction."""
    status = activity_status(db, user_id)
    if not status["sufficient"]:
        return None, status
    df = _frame(db, user_id)
    row = df.iloc[0]
    return {f: float(row[f]) for f in FEATURES}, status
