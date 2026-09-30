from flask import Blueprint, abort, flash, redirect, render_template, request, session, url_for

from db import get_db, jload, now
from ml import predict as ml_predict
from services import features as features_svc
from services import gaps as gaps_svc
from services import progress as progress_svc
from services import recommend as recommend_svc
from services.auth_helpers import current_user, role_required
from services.scoring import submit_attempt

student_bp = Blueprint("student", __name__)


def _uid():
    return session["user_id"]


# ---------------------------------------------------------------- dashboard ----

@student_bp.route("/dashboard")
@role_required("student")
def dashboard():
    db = get_db()
    user = current_user()

    subjects = db.execute(
        """SELECT s.*,
             (SELECT COUNT(*) FROM topics t WHERE t.subject_id = s.id) AS topic_count,
             (SELECT COALESCE(AVG(sp.progress_pct), 0) FROM student_progress sp
                JOIN topics t2 ON t2.id = sp.topic_id WHERE t2.subject_id = s.id AND sp.user_id = ?) AS progress_pct
           FROM subjects s JOIN enrollments e ON e.subject_id = s.id
           WHERE e.user_id = ? ORDER BY progress_pct DESC, s.name""", (user["id"], user["id"])).fetchall()

    attempts_done = db.execute("SELECT COUNT(*) c FROM attempts WHERE user_id=? AND status='submitted'",
                               (user["id"],)).fetchone()["c"]
    avg_score = db.execute(
        "SELECT ROUND(AVG(percentage),1) a FROM attempts WHERE user_id=? AND status='submitted'",
        (user["id"],)).fetchone()["a"] or 0
    active_gaps = db.execute("SELECT COUNT(*) c FROM learning_gaps WHERE user_id=? AND status='active'",
                             (user["id"],)).fetchone()["c"]
    completed_topics = db.execute("SELECT COUNT(*) c FROM student_progress WHERE user_id=? AND completed=1",
                                  (user["id"],)).fetchone()["c"]

    recent = db.execute(
        """SELECT a.*, s.name AS subject_name FROM attempts a LEFT JOIN subjects s ON s.id=a.subject_id
           WHERE a.user_id=? AND a.status='submitted' ORDER BY a.submitted_at DESC LIMIT 6""",
        (user["id"],)).fetchall()
    recs = db.execute(
        """SELECT r.*, t.name AS topic_name FROM recommendations r LEFT JOIN topics t ON t.id=r.topic_id
           WHERE r.user_id=? AND r.is_active=1 ORDER BY r.priority DESC LIMIT 4""", (user["id"],)).fetchall()
    notifications = db.execute(
        "SELECT * FROM notifications WHERE user_id=? ORDER BY created_at DESC LIMIT 5", (user["id"],)).fetchall()
    unread_count = db.execute("SELECT COUNT(*) c FROM notifications WHERE user_id=? AND is_read=0",
                             (user["id"],)).fetchone()["c"]

    prediction, activity = _prediction_for(db, user)

    return render_template("student/dashboard.html", user=user, subjects=subjects, attempts_done=attempts_done,
                          avg_score=avg_score, active_gaps=active_gaps, completed_topics=completed_topics,
                          recent=recent, recs=recs, notifications=notifications, unread_count=unread_count,
                          prediction=prediction, activity=activity)


def _prediction_for(db, user):
    """(prediction or None, activity_status). No prediction — and no placeholder numbers — until the
    student has completed their profile and has enough real activity."""
    feats, activity = features_svc.features_for_user(db, user["id"])
    if feats is None or not ml_predict.model_available():
        return None, activity
    return ml_predict.predict(feats), activity


# ---------------------------------------------------------------- subjects/topics ----

@student_bp.route("/subjects")
@role_required("student")
def subjects():
    db = get_db()
    rows = db.execute(
        """SELECT s.*, EXISTS(SELECT 1 FROM enrollments e WHERE e.subject_id=s.id AND e.user_id=?) AS enrolled,
             (SELECT COUNT(*) FROM topics t WHERE t.subject_id=s.id) AS topic_count
           FROM subjects s ORDER BY s.name""", (_uid(),)).fetchall()
    return render_template("student/subjects.html", subjects=rows)


@student_bp.route("/subjects/<int:subject_id>/enroll", methods=["POST"])
@role_required("student")
def enroll(subject_id):
    db = get_db()
    db.execute("INSERT OR IGNORE INTO enrollments (user_id, subject_id, enrolled_at) VALUES (?,?,?)",
              (_uid(), subject_id, now()))
    db.commit()
    flash("Enrolled! Your new subject is on your dashboard.", "success")
    return redirect(url_for("student.subject_detail", subject_id=subject_id))


@student_bp.route("/subjects/<int:subject_id>")
@role_required("student")
def subject_detail(subject_id):
    db = get_db()
    subject = db.execute("SELECT * FROM subjects WHERE id=?", (subject_id,)).fetchone()
    if not subject:
        abort(404)
    enrolled = db.execute("SELECT 1 FROM enrollments WHERE user_id=? AND subject_id=?",
                          (_uid(), subject_id)).fetchone() is not None
    topics = db.execute(
        """SELECT t.*, COALESCE(sp.progress_pct,0) AS progress_pct, COALESCE(sp.completed,0) AS completed,
             (SELECT COUNT(*) FROM questions q WHERE q.topic_id=t.id) AS question_count
           FROM topics t LEFT JOIN student_progress sp ON sp.topic_id=t.id AND sp.user_id=?
           WHERE t.subject_id=? ORDER BY t.position""", (_uid(), subject_id)).fetchall()
    assessment = db.execute("SELECT * FROM assessments WHERE subject_id=? AND is_published=1", (subject_id,)).fetchone()
    return render_template("student/subject_detail.html", subject=subject, topics=topics, enrolled=enrolled,
                          assessment=assessment)


@student_bp.route("/topics/<int:topic_id>")
@role_required("student")
def topic_detail(topic_id):
    db = get_db()
    topic = db.execute("SELECT t.*, s.name AS subject_name, s.id AS subject_id FROM topics t "
                       "JOIN subjects s ON s.id=t.subject_id WHERE t.id=?", (topic_id,)).fetchone()
    if not topic:
        abort(404)
    materials = db.execute(
        """SELECT m.*, COALESCE(mp.completed,0) AS done FROM materials m
           LEFT JOIN material_progress mp ON mp.material_id=m.id AND mp.user_id=?
           WHERE m.topic_id=? ORDER BY m.position""", (_uid(), topic_id)).fetchall()
    quiz = db.execute("SELECT * FROM quizzes WHERE topic_id=? AND is_published=1", (topic_id,)).fetchone()
    progress = db.execute("SELECT * FROM student_progress WHERE user_id=? AND topic_id=?",
                          (_uid(), topic_id)).fetchone()
    gap = db.execute("SELECT * FROM learning_gaps WHERE user_id=? AND topic_id=? AND status='active'",
                     (_uid(), topic_id)).fetchone()
    concepts = jload(topic["concepts"], [])
    return render_template("student/topic_detail.html", topic=topic, materials=materials, quiz=quiz,
                          progress=progress, gap=gap, concepts=concepts,
                          suggested_actions=jload(gap["suggested_actions"], []) if gap else [])


@student_bp.route("/materials/<int:material_id>/view")
@role_required("student")
def material_view(material_id):
    db = get_db()
    material = db.execute("SELECT m.*, t.name AS topic_name, t.id AS topic_id FROM materials m "
                          "JOIN topics t ON t.id=m.topic_id WHERE m.id=?", (material_id,)).fetchone()
    if not material:
        abort(404)
    progress_svc.touch_material(db, _uid(), material_id)
    return render_template("student/material_view.html", material=material)


@student_bp.route("/materials/<int:material_id>/complete", methods=["POST"])
@role_required("student")
def material_complete(material_id):
    db = get_db()
    material = db.execute("SELECT topic_id FROM materials WHERE id=?", (material_id,)).fetchone()
    if not material:
        abort(404)
    progress_svc.mark_material_done(db, _uid(), material_id)
    flash("Marked as complete.", "success")
    return redirect(url_for("student.topic_detail", topic_id=material["topic_id"]))


# ---------------------------------------------------------------- attempts ----

def _start_attempt(db, kind, question_ids, **extra):
    if not question_ids:
        abort(404)
    started = now()
    cur = db.execute(
        """INSERT INTO attempts (user_id, kind, quiz_id, assessment_id, subject_id, topic_id, title, difficulty,
                                 started_at, status, time_limit_sec)
           VALUES (?,?,?,?,?,?,?,?,?, 'in_progress', ?)""",
        (_uid(), kind, extra.get("quiz_id"), extra.get("assessment_id"), extra.get("subject_id"),
         extra.get("topic_id"), extra.get("title"), extra.get("difficulty", "Mixed"), started,
         extra.get("time_limit_sec")))
    attempt_id = cur.lastrowid
    for pos, qid in enumerate(question_ids):
        db.execute("INSERT INTO attempt_answers (attempt_id, question_id, position, correct_option) "
                  "SELECT ?, id, ?, correct_option FROM questions WHERE id=?", (attempt_id, pos, qid))
    db.commit()
    return attempt_id


@student_bp.route("/practice/<int:topic_id>/start", methods=["POST"])
@role_required("student")
def practice_start(topic_id):
    db = get_db()
    topic = db.execute("SELECT * FROM topics WHERE id=?", (topic_id,)).fetchone()
    if not topic:
        abort(404)
    qids = [r["id"] for r in db.execute("SELECT id FROM questions WHERE topic_id=? ORDER BY id", (topic_id,)).fetchall()]
    attempt_id = _start_attempt(db, "practice", qids, subject_id=topic["subject_id"], topic_id=topic_id,
                               title=f"{topic['name']} Practice")
    return redirect(url_for("student.attempt_take", attempt_id=attempt_id))


@student_bp.route("/quiz/<int:quiz_id>/start", methods=["POST"])
@role_required("student")
def quiz_start(quiz_id):
    db = get_db()
    quiz = db.execute("SELECT * FROM quizzes WHERE id=?", (quiz_id,)).fetchone()
    if not quiz:
        abort(404)
    qids = [r["question_id"] for r in db.execute(
        "SELECT question_id FROM quiz_questions WHERE quiz_id=? ORDER BY position", (quiz_id,)).fetchall()]
    attempt_id = _start_attempt(db, "quiz", qids, quiz_id=quiz_id, subject_id=quiz["subject_id"],
                               topic_id=quiz["topic_id"], title=quiz["title"], difficulty=quiz["difficulty"])
    return redirect(url_for("student.attempt_take", attempt_id=attempt_id))


@student_bp.route("/assessment/<int:assessment_id>/start", methods=["POST"])
@role_required("student")
def assessment_start(assessment_id):
    db = get_db()
    a = db.execute("SELECT * FROM assessments WHERE id=?", (assessment_id,)).fetchone()
    if not a:
        abort(404)
    qids = [r["question_id"] for r in db.execute(
        "SELECT question_id FROM assessment_questions WHERE assessment_id=? ORDER BY position",
        (assessment_id,)).fetchall()]
    attempt_id = _start_attempt(db, "assessment", qids, assessment_id=assessment_id, subject_id=a["subject_id"],
                               title=a["title"], difficulty=a["difficulty"], time_limit_sec=a["time_limit_min"] * 60)
    return redirect(url_for("student.attempt_take", attempt_id=attempt_id))


@student_bp.route("/attempt/<int:attempt_id>", methods=["GET", "POST"])
@role_required("student")
def attempt_take(attempt_id):
    db = get_db()
    attempt = db.execute("SELECT * FROM attempts WHERE id=? AND user_id=?", (attempt_id, _uid())).fetchone()
    if not attempt:
        abort(404)
    if attempt["status"] == "submitted":
        return redirect(url_for("student.attempt_result", attempt_id=attempt_id))

    if request.method == "POST":
        answers = {}
        for key, val in request.form.items():
            if key.startswith("q_") and val:
                answers[int(key[2:])] = val
        elapsed = int(request.form.get("elapsed_sec", 0) or 0)
        submit_attempt(db, attempt_id, answers, duration_sec=elapsed)
        progress_svc.recompute_topic_progress(db, _uid(), attempt["topic_id"]) if attempt["topic_id"] else None
        if attempt["kind"] == "assessment":
            for t in db.execute(
                "SELECT DISTINCT topic_id FROM questions WHERE id IN "
                "(SELECT question_id FROM attempt_answers WHERE attempt_id=?)", (attempt_id,)).fetchall():
                progress_svc.recompute_topic_progress(db, _uid(), t["topic_id"])
        gaps_svc.detect_all_for_user(db, _uid())
        recommend_svc.generate_for_user(db, _uid())
        return redirect(url_for("student.attempt_result", attempt_id=attempt_id))

    questions = db.execute(
        """SELECT q.*, aa.selected_option FROM attempt_answers aa JOIN questions q ON q.id=aa.question_id
           WHERE aa.attempt_id=? ORDER BY aa.position""", (attempt_id,)).fetchall()
    return render_template("student/attempt_take.html", attempt=attempt, questions=questions)


@student_bp.route("/results/<int:attempt_id>")
@role_required("student")
def attempt_result(attempt_id):
    db = get_db()
    attempt = db.execute("SELECT * FROM attempts WHERE id=? AND user_id=?", (attempt_id, _uid())).fetchone()
    if not attempt:
        abort(404)
    if attempt["status"] != "submitted":
        return redirect(url_for("student.attempt_take", attempt_id=attempt_id))
    answers = db.execute(
        """SELECT aa.*, q.text, q.option_a, q.option_b, q.option_c, q.option_d, q.explanation, q.difficulty
           FROM attempt_answers aa JOIN questions q ON q.id=aa.question_id
           WHERE aa.attempt_id=? ORDER BY aa.position""", (attempt_id,)).fetchall()
    return render_template("student/result.html", attempt=attempt, answers=answers)


# ---------------------------------------------------------------- performance / gaps / recs / progress ----

@student_bp.route("/performance")
@role_required("student")
def performance():
    db = get_db()
    user = current_user()
    subject_scores = db.execute(
        """SELECT s.name, ROUND(AVG(sp.score),1) avg_score FROM student_progress sp
           JOIN topics t ON t.id=sp.topic_id JOIN subjects s ON s.id=t.subject_id
           WHERE sp.user_id=? AND sp.score IS NOT NULL GROUP BY s.id ORDER BY s.name""", (user["id"],)).fetchall()
    prediction, activity = _prediction_for(db, user)
    explanation = None
    if prediction:
        explanation = ml_predict.explain(features_svc.features_for_user(db, user["id"])[0])
    return render_template("student/performance.html", user=user, subject_scores=subject_scores,
                          prediction=prediction, explanation=explanation, activity=activity)


@student_bp.route("/learning-gaps")
@role_required("student")
def learning_gaps():
    db = get_db()
    gaps = db.execute(
        """SELECT lg.*, t.name AS topic_name, s.name AS subject_name, s.id AS subject_id
           FROM learning_gaps lg JOIN topics t ON t.id=lg.topic_id JOIN subjects s ON s.id=t.subject_id
           WHERE lg.user_id=? AND lg.status='active'
           ORDER BY CASE lg.severity WHEN 'High' THEN 0 WHEN 'Medium' THEN 1 ELSE 2 END, lg.score ASC""",
        (_uid(),)).fetchall()
    return render_template("student/learning_gaps.html",
                          gaps=[dict(g, suggested_actions=jload(g["suggested_actions"], [])) for g in gaps])


@student_bp.route("/recommendations")
@role_required("student")
def recommendations():
    db = get_db()
    recs = db.execute(
        """SELECT r.*, t.name AS topic_name FROM recommendations r LEFT JOIN topics t ON t.id=r.topic_id
           WHERE r.user_id=? AND r.is_active=1 ORDER BY r.priority DESC""", (_uid(),)).fetchall()
    return render_template("student/recommendations.html", recs=recs)


@student_bp.route("/progress")
@role_required("student")
def progress():
    db = get_db()
    rows = db.execute(
        """SELECT sp.*, t.name AS topic_name, s.name AS subject_name FROM student_progress sp
           JOIN topics t ON t.id=sp.topic_id JOIN subjects s ON s.id=t.subject_id
           WHERE sp.user_id=? ORDER BY s.name, t.position""", (_uid(),)).fetchall()
    return render_template("student/progress.html", rows=rows)


@student_bp.route("/history")
@role_required("student")
def history():
    db = get_db()
    kind = request.args.get("kind", "")
    q = "SELECT a.*, s.name AS subject_name FROM attempts a LEFT JOIN subjects s ON s.id=a.subject_id WHERE a.user_id=? AND a.status='submitted'"
    args = [_uid()]
    if kind in ("practice", "quiz", "assessment"):
        q += " AND a.kind=?"
        args.append(kind)
    q += " ORDER BY a.submitted_at DESC"
    attempts = db.execute(q, args).fetchall()
    return render_template("student/history.html", attempts=attempts, kind=kind)


@student_bp.route("/profile", methods=["GET", "POST"])
@role_required("student")
def profile():
    db = get_db()
    user = current_user()
    if request.method == "POST":
        phone = request.form.get("phone", "").strip()

        def num(field, lo, hi, current):
            raw = request.form.get(field, "").strip()
            if not raw:
                return current
            try:
                return max(lo, min(hi, float(raw)))
            except ValueError:
                return current
        hours = num("study_hours_per_week", 0, 80, user["study_hours_per_week"])
        attendance = num("attendance_percentage", 0, 100, user["attendance_percentage"])
        gpa = num("previous_gpa", 0, 10, user["previous_gpa"])
        db.execute("UPDATE users SET phone=?, study_hours_per_week=?, attendance_percentage=?, previous_gpa=? WHERE id=?",
                   (phone, hours, attendance, gpa, user["id"]))
        db.commit()
        flash("Profile updated.", "success")
        return redirect(url_for("student.profile"))
    return render_template("student/profile.html", user=user)


@student_bp.route("/notifications/<int:notif_id>/read", methods=["POST"])
@role_required("student")
def notification_read(notif_id):
    db = get_db()
    db.execute("UPDATE notifications SET is_read=1 WHERE id=? AND user_id=?", (notif_id, _uid()))
    db.commit()
    return redirect(request.referrer or url_for("student.dashboard"))


@student_bp.route("/search")
@role_required("student")
def search():
    db = get_db()
    query = request.args.get("q", "").strip()
    subjects = topics = materials = []
    if query:
        like = f"%{query}%"
        subjects = db.execute("SELECT * FROM subjects WHERE name LIKE ? OR description LIKE ?", (like, like)).fetchall()
        topics = db.execute(
            """SELECT t.*, s.name AS subject_name FROM topics t JOIN subjects s ON s.id=t.subject_id
               WHERE t.name LIKE ? OR t.description LIKE ?""", (like, like)).fetchall()
        materials = db.execute(
            """SELECT m.*, t.name AS topic_name FROM materials m JOIN topics t ON t.id=m.topic_id
               WHERE m.title LIKE ? OR m.description LIKE ?""", (like, like)).fetchall()
    return render_template("student/search.html", query=query, subjects=subjects, topics=topics, materials=materials)
