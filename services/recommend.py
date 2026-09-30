"""Turns active learning gaps (and healthy-but-unexplored topics) into concrete,
clickable recommendations that always point at a real in-app material, topic or practice set."""
from db import now
from ml import predict as ml_predict
from services import features as features_svc

SEVERITY_PRIORITY = {"High": 90, "Medium": 60, "Low": 40}


def _best_material(db, topic_id, prefer_type=None):
    if prefer_type:
        row = db.execute("SELECT id, title, type FROM materials WHERE topic_id = ? AND type = ? LIMIT 1",
                         (topic_id, prefer_type)).fetchone()
        if row:
            return row
    return db.execute(
        """SELECT id, title, type FROM materials WHERE topic_id = ?
           ORDER BY is_recommended DESC, position ASC LIMIT 1""", (topic_id,)).fetchone()


def performance_level(db, user_id):
    """The ML performance level (Low/Medium/High), or None when the student has not yet got enough
    real activity / profile data or no model is trained. Never invents a level."""
    if not ml_predict.model_available():
        return None
    feats, _ = features_svc.features_for_user(db, user_id)
    return ml_predict.predict(feats)["level"] if feats is not None else None


LEVEL_NOTE = {
    "Low": " Your overall performance level is Low, so foundations come first.",
    "High": " You are performing strongly overall, so a short targeted review should close this.",
}


def generate_for_user(db, user_id, limit=6):
    """Recommendations combine two separate signals: threshold-based learning gaps (topic level) and the
    ML-predicted overall performance level (only when available). The ML level never creates or removes a
    gap; it only re-prioritises recommendations and picks the kind of study material."""
    level = performance_level(db, user_id)
    gap_limit = limit - 1 if level in ("Low", "High") else limit      # keep one slot for the level-driven pick
    db.execute("UPDATE recommendations SET is_active = 0 WHERE user_id = ? AND source = 'system'", (user_id,))

    gaps = db.execute(
        """SELECT lg.*, t.name AS topic_name, t.subject_id, s.name AS subject_name
           FROM learning_gaps lg JOIN topics t ON t.id = lg.topic_id JOIN subjects s ON s.id = t.subject_id
           WHERE lg.user_id = ? AND lg.status = 'active'
           ORDER BY CASE lg.severity WHEN 'High' THEN 0 WHEN 'Medium' THEN 1 ELSE 2 END, lg.score ASC
           LIMIT ?""", (user_id, gap_limit)).fetchall()

    created = 0
    for g in gaps:
        material = _best_material(db, g["topic_id"], "practice" if level == "High" else None)
        priority = SEVERITY_PRIORITY.get(g["severity"], 50) + (10 if level == "Low" else 0)
        db.execute(
            """INSERT INTO recommendations (user_id, topic_id, material_id, kind, title, reason,
                                            action_label, action_url, priority, source, is_active, created_at)
               VALUES (?,?,?,?,?,?,?,?,?, 'system', 1, ?)""",
            (user_id, g["topic_id"], material["id"] if material else None, "weak_topic",
             f"Strengthen {g['topic_name']}", g["observed_issue"] + LEVEL_NOTE.get(level, ""), "Study now",
             f"/student/topics/{g['topic_id']}", priority, now()))
        created += 1

    if level in ("Low", "High"):
        created += _level_recommendation(db, user_id, level)

    if created < limit:
        unexplored = db.execute(
            """SELECT t.id, t.name FROM topics t
               JOIN subjects s ON s.id = t.subject_id
               JOIN enrollments e ON e.subject_id = s.id AND e.user_id = ?
               WHERE t.id NOT IN (
                 SELECT DISTINCT q.topic_id FROM questions q
                 JOIN attempt_answers aa ON aa.question_id = q.id
                 JOIN attempts a ON a.id = aa.attempt_id WHERE a.user_id = ? AND a.status='submitted')
               ORDER BY t.position LIMIT ?""",
            (user_id, user_id, limit - created)).fetchall()
        for t in unexplored:
            material = _best_material(db, t["id"])
            db.execute(
                """INSERT INTO recommendations (user_id, topic_id, material_id, kind, title, reason,
                                                action_label, action_url, priority, source, is_active, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?, 'system', 1, ?)""",
                (user_id, t["id"], material["id"] if material else None, "explore",
                 f"Get started with {t['name']}", "You haven't attempted any questions on this topic yet.",
                 "Start learning", f"/student/topics/{t['id']}", 20, now()))
    db.commit()


def _level_recommendation(db, user_id, level):
    """One extra recommendation driven by the ML level: Low -> a Beginner topic to build foundations,
    High -> an Advanced topic to stretch. Always links to a real topic the student is enrolled in."""
    difficulty, kind, title, reason, label = (
        ("Beginner", "foundation", "Build your foundations", "Your overall performance level is Low: start "
         "with a Beginner topic and finish its notes and practice set before moving on.", "Start basics")
        if level == "Low" else
        ("Advanced", "challenge", "Challenge yourself", "Your overall performance level is High: try an "
         "Advanced topic to keep progressing.", "Take the challenge"))
    topic = db.execute(
        """SELECT t.id, t.name FROM topics t JOIN enrollments e ON e.subject_id = t.subject_id AND e.user_id = ?
           LEFT JOIN student_progress sp ON sp.topic_id = t.id AND sp.user_id = ?
           WHERE t.difficulty = ? AND COALESCE(sp.progress_pct, 0) < 100
                 AND t.id NOT IN (SELECT topic_id FROM learning_gaps WHERE user_id = ? AND status = 'active')
           ORDER BY COALESCE(sp.progress_pct, 0) ASC, t.position LIMIT 1""",
        (user_id, user_id, difficulty, user_id)).fetchone()
    if not topic:
        return 0
    material = _best_material(db, topic["id"], "practice" if level == "High" else None)
    db.execute(
        """INSERT INTO recommendations (user_id, topic_id, material_id, kind, title, reason, action_label,
                                        action_url, priority, source, is_active, created_at)
           VALUES (?,?,?,?,?,?,?,?,?, 'system', 1, ?)""",
        (user_id, topic["id"], material["id"] if material else None, kind, f"{title}: {topic['name']}", reason,
         label, f"/student/topics/{topic['id']}", 55, now()))
    return 1
