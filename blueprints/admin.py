import os
import uuid

from flask import (Blueprint, abort, current_app, flash, redirect, render_template, request,
                   session, url_for)
from werkzeug.security import generate_password_hash
from werkzeug.utils import secure_filename

from db import get_db, jload, now
from ml import predict as ml_predict
from services import features as features_svc
from services.auth_helpers import current_user, role_required

admin_bp = Blueprint("admin", __name__)


# ---------------------------------------------------------------- dashboard ----

@admin_bp.route("/dashboard")
@role_required("admin")
def dashboard():
    db = get_db()
    stats = {
        "students": db.execute("SELECT COUNT(*) c FROM users WHERE role='student'").fetchone()["c"],
        "subjects": db.execute("SELECT COUNT(*) c FROM subjects").fetchone()["c"],
        "questions": db.execute("SELECT COUNT(*) c FROM questions").fetchone()["c"],
        "attempts": db.execute("SELECT COUNT(*) c FROM attempts WHERE status='submitted'").fetchone()["c"],
        "active_gaps": db.execute("SELECT COUNT(*) c FROM learning_gaps WHERE status='active'").fetchone()["c"],
        "avg_score": db.execute("SELECT ROUND(AVG(percentage),1) a FROM attempts WHERE status='submitted'")
                       .fetchone()["a"] or 0,
    }
    level_dist = _performance_distribution(db)
    weak_topics = db.execute(
        """SELECT t.name, s.name AS subject_name, COUNT(*) c, ROUND(AVG(lg.score),1) avg_score
           FROM learning_gaps lg JOIN topics t ON t.id=lg.topic_id JOIN subjects s ON s.id=t.subject_id
           WHERE lg.status='active' GROUP BY t.id ORDER BY c DESC, avg_score ASC LIMIT 8""").fetchall()
    recent_students = db.execute(
        "SELECT * FROM users WHERE role='student' ORDER BY created_at DESC LIMIT 6").fetchall()
    pending_gaps = db.execute("SELECT COUNT(*) c FROM learning_gaps WHERE status='active' AND severity='High'") \
                     .fetchone()["c"]
    return render_template("admin/dashboard.html", stats=stats, level_dist=level_dist, weak_topics=weak_topics,
                          recent_students=recent_students, pending_gaps=pending_gaps)


def _performance_distribution(db):
    """Low/Medium/High counts from the trained model, predicted in one batch from live DB features.
    Students without enough activity/profile data get no prediction and are counted as 'Pending'."""
    dist = {"Low": 0, "Medium": 0, "High": 0, "Pending": 0}
    total = db.execute("SELECT COUNT(*) c FROM users WHERE role='student' AND is_active=1").fetchone()["c"]
    if not ml_predict.model_available():
        dist["Pending"] = total
        return dist
    frame = features_svc.feature_frame(db)
    for level in ml_predict.predict_levels(frame):
        dist[level] = dist.get(level, 0) + 1
    dist["Pending"] = max(0, total - len(frame))
    return dist


# ---------------------------------------------------------------- students ----

@admin_bp.route("/students")
@role_required("admin")
def students():
    db = get_db()
    q = request.args.get("q", "").strip()
    sql = "SELECT * FROM users WHERE role='student'"
    args = []
    if q:
        sql += " AND (name LIKE ? OR email LIKE ? OR student_code LIKE ?)"
        args += [f"%{q}%"] * 3
    sql += " ORDER BY name"
    rows = db.execute(sql, args).fetchall()
    return render_template("admin/students.html", students=rows, q=q)


@admin_bp.route("/students/<int:user_id>")
@role_required("admin")
def student_detail(user_id):
    db = get_db()
    student = db.execute("SELECT * FROM users WHERE id=? AND role='student'", (user_id,)).fetchone()
    if not student:
        abort(404)
    subject_scores = db.execute(
        """SELECT s.name, ROUND(AVG(sp.score),1) avg_score FROM student_progress sp
           JOIN topics t ON t.id=sp.topic_id JOIN subjects s ON s.id=t.subject_id
           WHERE sp.user_id=? AND sp.score IS NOT NULL GROUP BY s.id ORDER BY s.name""", (user_id,)).fetchall()
    gaps = db.execute(
        """SELECT lg.*, t.name AS topic_name FROM learning_gaps lg JOIN topics t ON t.id=lg.topic_id
           WHERE lg.user_id=? AND lg.status='active' ORDER BY lg.score ASC""", (user_id,)).fetchall()
    recs = db.execute(
        """SELECT r.*, t.name AS topic_name FROM recommendations r LEFT JOIN topics t ON t.id=r.topic_id
           WHERE r.user_id=? AND r.is_active=1 ORDER BY r.priority DESC""", (user_id,)).fetchall()
    prediction = explanation = None
    feats, activity = features_svc.features_for_user(db, user_id)
    if feats is not None and ml_predict.model_available():
        prediction = ml_predict.predict(feats)
        explanation = ml_predict.explain(feats)
    attempts = db.execute("SELECT * FROM attempts WHERE user_id=? AND status='submitted' ORDER BY submitted_at DESC LIMIT 10",
                          (user_id,)).fetchall()
    return render_template("admin/student_detail.html", student=student, subject_scores=subject_scores, gaps=gaps,
                          recs=recs, prediction=prediction, explanation=explanation, attempts=attempts,
                          activity=activity)


@admin_bp.route("/students/<int:user_id>/toggle-active", methods=["POST"])
@role_required("admin")
def student_toggle_active(user_id):
    db = get_db()
    row = db.execute("SELECT is_active FROM users WHERE id=? AND role='student'", (user_id,)).fetchone()
    if not row:
        abort(404)
    db.execute("UPDATE users SET is_active=? WHERE id=?", (0 if row["is_active"] else 1, user_id))
    db.commit()
    flash("Student account updated.", "success")
    return redirect(url_for("admin.student_detail", user_id=user_id))


# ---------------------------------------------------------------- subjects ----

def _allowed_image(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in current_app.config["ALLOWED_IMAGE_EXTENSIONS"]


@admin_bp.route("/subjects")
@role_required("admin")
def subjects():
    db = get_db()
    rows = db.execute(
        """SELECT s.*, (SELECT COUNT(*) FROM topics t WHERE t.subject_id=s.id) topic_count,
             (SELECT COUNT(*) FROM enrollments e WHERE e.subject_id=s.id) enrolled_count
           FROM subjects s ORDER BY s.name""").fetchall()
    return render_template("admin/subjects.html", subjects=rows)


@admin_bp.route("/subjects/new", methods=["GET", "POST"])
@role_required("admin")
def subject_new():
    if request.method == "POST":
        db = get_db()
        name = request.form.get("name", "").strip()
        category = request.form.get("category", "").strip()
        description = request.form.get("description", "").strip()
        if not name:
            flash("Subject name is required.", "error")
            return render_template("admin/subject_form.html", subject=None)
        image_path = "images/courses/default.jpg"
        file = request.files.get("image")
        if file and file.filename and _allowed_image(file.filename):
            fname = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            file.save(os.path.join(current_app.config["UPLOAD_FOLDER"], fname))
            image_path = f"uploads/{fname}"
        db.execute("INSERT INTO subjects (name, category, description, image_path, created_at) VALUES (?,?,?,?,?)",
                  (name, category, description, image_path, now()))
        db.commit()
        flash("Subject created.", "success")
        return redirect(url_for("admin.subjects"))
    return render_template("admin/subject_form.html", subject=None)


@admin_bp.route("/subjects/<int:subject_id>/edit", methods=["GET", "POST"])
@role_required("admin")
def subject_edit(subject_id):
    db = get_db()
    subject = db.execute("SELECT * FROM subjects WHERE id=?", (subject_id,)).fetchone()
    if not subject:
        abort(404)
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        category = request.form.get("category", "").strip()
        description = request.form.get("description", "").strip()
        image_path = subject["image_path"]
        file = request.files.get("image")
        if file and file.filename and _allowed_image(file.filename):
            fname = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            file.save(os.path.join(current_app.config["UPLOAD_FOLDER"], fname))
            image_path = f"uploads/{fname}"
        db.execute("UPDATE subjects SET name=?, category=?, description=?, image_path=? WHERE id=?",
                  (name, category, description, image_path, subject_id))
        db.commit()
        flash("Subject updated.", "success")
        return redirect(url_for("admin.subjects"))
    return render_template("admin/subject_form.html", subject=subject)


@admin_bp.route("/subjects/<int:subject_id>/delete", methods=["POST"])
@role_required("admin")
def subject_delete(subject_id):
    db = get_db()
    db.execute("DELETE FROM subjects WHERE id=?", (subject_id,))
    db.commit()
    flash("Subject and all its topics/questions were deleted.", "info")
    return redirect(url_for("admin.subjects"))


# ---------------------------------------------------------------- topics ----

@admin_bp.route("/subjects/<int:subject_id>/topics")
@role_required("admin")
def topics(subject_id):
    db = get_db()
    subject = db.execute("SELECT * FROM subjects WHERE id=?", (subject_id,)).fetchone()
    if not subject:
        abort(404)
    rows = db.execute(
        """SELECT t.*, (SELECT COUNT(*) FROM questions q WHERE q.topic_id=t.id) question_count,
             (SELECT COUNT(*) FROM materials m WHERE m.topic_id=t.id) material_count
           FROM topics t WHERE t.subject_id=? ORDER BY t.position""", (subject_id,)).fetchall()
    return render_template("admin/topics.html", subject=subject, topics=rows)


@admin_bp.route("/subjects/<int:subject_id>/topics/new", methods=["GET", "POST"])
@role_required("admin")
def topic_new(subject_id):
    db = get_db()
    subject = db.execute("SELECT * FROM subjects WHERE id=?", (subject_id,)).fetchone()
    if not subject:
        abort(404)
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        description = request.form.get("description", "").strip()
        difficulty = request.form.get("difficulty", "Intermediate")
        if not name:
            flash("Topic name is required.", "error")
            return render_template("admin/topic_form.html", subject=subject, topic=None)
        pos = db.execute("SELECT COALESCE(MAX(position),-1)+1 p FROM topics WHERE subject_id=?",
                         (subject_id,)).fetchone()["p"]
        db.execute("INSERT INTO topics (subject_id, name, description, difficulty, position, concepts) "
                  "VALUES (?,?,?,?,?,'[]')", (subject_id, name, description, difficulty, pos))
        db.commit()
        flash("Topic created.", "success")
        return redirect(url_for("admin.topics", subject_id=subject_id))
    return render_template("admin/topic_form.html", subject=subject, topic=None)


@admin_bp.route("/topics/<int:topic_id>/edit", methods=["GET", "POST"])
@role_required("admin")
def topic_edit(topic_id):
    db = get_db()
    topic = db.execute("SELECT * FROM topics WHERE id=?", (topic_id,)).fetchone()
    if not topic:
        abort(404)
    subject = db.execute("SELECT * FROM subjects WHERE id=?", (topic["subject_id"],)).fetchone()
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        description = request.form.get("description", "").strip()
        difficulty = request.form.get("difficulty", "Intermediate")
        db.execute("UPDATE topics SET name=?, description=?, difficulty=? WHERE id=?",
                  (name, description, difficulty, topic_id))
        db.commit()
        flash("Topic updated.", "success")
        return redirect(url_for("admin.topics", subject_id=topic["subject_id"]))
    return render_template("admin/topic_form.html", subject=subject, topic=topic)


@admin_bp.route("/topics/<int:topic_id>/delete", methods=["POST"])
@role_required("admin")
def topic_delete(topic_id):
    db = get_db()
    topic = db.execute("SELECT subject_id FROM topics WHERE id=?", (topic_id,)).fetchone()
    if not topic:
        abort(404)
    db.execute("DELETE FROM topics WHERE id=?", (topic_id,))
    db.commit()
    flash("Topic deleted.", "info")
    return redirect(url_for("admin.topics", subject_id=topic["subject_id"]))


# ---------------------------------------------------------------- materials ----

@admin_bp.route("/topics/<int:topic_id>/materials")
@role_required("admin")
def materials(topic_id):
    db = get_db()
    topic = db.execute("SELECT t.*, s.name AS subject_name FROM topics t JOIN subjects s ON s.id=t.subject_id "
                       "WHERE t.id=?", (topic_id,)).fetchone()
    if not topic:
        abort(404)
    rows = db.execute("SELECT * FROM materials WHERE topic_id=? ORDER BY position", (topic_id,)).fetchall()
    return render_template("admin/materials.html", topic=topic, materials=rows)


@admin_bp.route("/topics/<int:topic_id>/materials/new", methods=["GET", "POST"])
@role_required("admin")
def material_new(topic_id):
    db = get_db()
    topic = db.execute("SELECT * FROM topics WHERE id=?", (topic_id,)).fetchone()
    if not topic:
        abort(404)
    if request.method == "POST":
        ok, payload = _material_form(request)
        if not ok:
            flash(payload, "error")
            return render_template("admin/material_form.html", topic=topic, material=None)
        pos = db.execute("SELECT COALESCE(MAX(position),-1)+1 p FROM materials WHERE topic_id=?",
                         (topic_id,)).fetchone()["p"]
        db.execute(
            """INSERT INTO materials (topic_id, title, description, type, url, content, est_minutes,
                                      is_recommended, position, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (topic_id, payload["title"], payload["description"], payload["type"], payload["url"],
             payload["content"], payload["est_minutes"], payload["is_recommended"], pos, now()))
        db.commit()
        flash("Material added.", "success")
        return redirect(url_for("admin.materials", topic_id=topic_id))
    return render_template("admin/material_form.html", topic=topic, material=None)


def _material_form(req):
    mtype = req.form.get("type", "notes")
    title = req.form.get("title", "").strip()
    url = req.form.get("url", "").strip() or None
    if not title:
        return False, "Title is required."
    if mtype in ("video", "pdf", "article") and not url:
        return False, f"A valid URL is required for a {mtype} material — no placeholder links allowed."
    try:
        est = max(1, int(req.form.get("est_minutes", 10) or 10))
    except ValueError:
        est = 10
    return True, {"title": title, "description": req.form.get("description", "").strip(), "type": mtype,
                 "url": url, "content": req.form.get("content", "").strip() or None, "est_minutes": est,
                 "is_recommended": 1 if req.form.get("is_recommended") else 0}


@admin_bp.route("/materials/<int:material_id>/edit", methods=["GET", "POST"])
@role_required("admin")
def material_edit(material_id):
    db = get_db()
    material = db.execute("SELECT * FROM materials WHERE id=?", (material_id,)).fetchone()
    if not material:
        abort(404)
    topic = db.execute("SELECT * FROM topics WHERE id=?", (material["topic_id"],)).fetchone()
    if request.method == "POST":
        ok, payload = _material_form(request)
        if not ok:
            flash(payload, "error")
            return render_template("admin/material_form.html", topic=topic, material=material)
        db.execute(
            """UPDATE materials SET title=?, description=?, type=?, url=?, content=?, est_minutes=?,
                                   is_recommended=?, updated_at=? WHERE id=?""",
            (payload["title"], payload["description"], payload["type"], payload["url"], payload["content"],
             payload["est_minutes"], payload["is_recommended"], now(), material_id))
        db.commit()
        flash("Material updated.", "success")
        return redirect(url_for("admin.materials", topic_id=topic["id"]))
    return render_template("admin/material_form.html", topic=topic, material=material)


@admin_bp.route("/materials/<int:material_id>/delete", methods=["POST"])
@role_required("admin")
def material_delete(material_id):
    db = get_db()
    material = db.execute("SELECT topic_id FROM materials WHERE id=?", (material_id,)).fetchone()
    if not material:
        abort(404)
    db.execute("DELETE FROM materials WHERE id=?", (material_id,))
    db.commit()
    flash("Material deleted.", "info")
    return redirect(url_for("admin.materials", topic_id=material["topic_id"]))


# ---------------------------------------------------------------- question bank ----

@admin_bp.route("/questions")
@role_required("admin")
def questions():
    db = get_db()
    subject_id = request.args.get("subject_id", type=int)
    topic_id = request.args.get("topic_id", type=int)
    difficulty = request.args.get("difficulty", "")
    q = request.args.get("q", "").strip()

    sql = """SELECT q.*, t.name AS topic_name, s.id AS subject_id, s.name AS subject_name
              FROM questions q JOIN topics t ON t.id=q.topic_id JOIN subjects s ON s.id=t.subject_id WHERE 1=1"""
    args = []
    if subject_id:
        sql += " AND s.id=?"; args.append(subject_id)
    if topic_id:
        sql += " AND t.id=?"; args.append(topic_id)
    if difficulty in ("Easy", "Medium", "Hard"):
        sql += " AND q.difficulty=?"; args.append(difficulty)
    if q:
        sql += " AND q.text LIKE ?"; args.append(f"%{q}%")
    sql += " ORDER BY s.name, t.position, q.id"
    rows = db.execute(sql, args).fetchall()
    subjects = db.execute("SELECT * FROM subjects ORDER BY name").fetchall()
    topic_rows = db.execute("SELECT * FROM topics WHERE subject_id=? ORDER BY position", (subject_id,)).fetchall() \
        if subject_id else []
    return render_template("admin/questions.html", questions=rows, subjects=subjects, topics=topic_rows,
                          subject_id=subject_id, topic_id=topic_id, difficulty=difficulty, q=q,
                          total=len(rows))


def _question_form(req):
    text = req.form.get("text", "").strip()
    opts = {k: req.form.get(k, "").strip() for k in ("option_a", "option_b", "option_c", "option_d")}
    correct = req.form.get("correct_option", "")
    explanation = req.form.get("explanation", "").strip()
    difficulty = req.form.get("difficulty", "Medium")
    if not text or not all(opts.values()) or correct not in ("A", "B", "C", "D") or not explanation:
        return False, "All fields (question, four options, correct answer, explanation) are required."
    if len({v.strip().lower() for v in opts.values()}) < 4:
        return False, "The four options must be distinct."
    marks = {"Easy": 1, "Medium": 2, "Hard": 3}.get(difficulty, 1)
    return True, {**opts, "text": text, "correct_option": correct, "explanation": explanation,
                 "difficulty": difficulty, "marks": marks}


@admin_bp.route("/topics/<int:topic_id>/questions/new", methods=["GET", "POST"])
@role_required("admin")
def question_new(topic_id):
    db = get_db()
    topic = db.execute("SELECT * FROM topics WHERE id=?", (topic_id,)).fetchone()
    if not topic:
        abort(404)
    if request.method == "POST":
        ok, payload = _question_form(request)
        if not ok:
            flash(payload, "error")
            return render_template("admin/question_form.html", topic=topic, question=None)
        db.execute(
            """INSERT INTO questions (topic_id, text, option_a, option_b, option_c, option_d, correct_option,
                                      explanation, difficulty, marks, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (topic_id, payload["text"], payload["option_a"], payload["option_b"], payload["option_c"],
             payload["option_d"], payload["correct_option"], payload["explanation"], payload["difficulty"],
             payload["marks"], now()))
        db.commit()
        flash("Question added to the bank.", "success")
        return redirect(url_for("admin.questions", topic_id=topic_id))
    return render_template("admin/question_form.html", topic=topic, question=None)


@admin_bp.route("/questions/<int:question_id>/edit", methods=["GET", "POST"])
@role_required("admin")
def question_edit(question_id):
    db = get_db()
    question = db.execute("SELECT * FROM questions WHERE id=?", (question_id,)).fetchone()
    if not question:
        abort(404)
    topic = db.execute("SELECT * FROM topics WHERE id=?", (question["topic_id"],)).fetchone()
    if request.method == "POST":
        ok, payload = _question_form(request)
        if not ok:
            flash(payload, "error")
            return render_template("admin/question_form.html", topic=topic, question=question)
        db.execute(
            """UPDATE questions SET text=?, option_a=?, option_b=?, option_c=?, option_d=?, correct_option=?,
                                   explanation=?, difficulty=?, marks=? WHERE id=?""",
            (payload["text"], payload["option_a"], payload["option_b"], payload["option_c"], payload["option_d"],
             payload["correct_option"], payload["explanation"], payload["difficulty"], payload["marks"], question_id))
        db.commit()
        flash("Question updated.", "success")
        return redirect(url_for("admin.questions", topic_id=topic["id"]))
    return render_template("admin/question_form.html", topic=topic, question=question)


@admin_bp.route("/questions/<int:question_id>/delete", methods=["POST"])
@role_required("admin")
def question_delete(question_id):
    db = get_db()
    question = db.execute("SELECT topic_id FROM questions WHERE id=?", (question_id,)).fetchone()
    if not question:
        abort(404)
    db.execute("DELETE FROM questions WHERE id=?", (question_id,))
    db.commit()
    flash("Question removed from the bank.", "info")
    return redirect(url_for("admin.questions", topic_id=question["topic_id"]))


# ---------------------------------------------------------------- quizzes / assessments ----

@admin_bp.route("/quizzes")
@role_required("admin")
def quizzes():
    db = get_db()
    rows = db.execute(
        """SELECT qz.*, s.name AS subject_name, t.name AS topic_name,
             (SELECT COUNT(*) FROM quiz_questions qq WHERE qq.quiz_id=qz.id) q_count,
             (SELECT COUNT(*) FROM attempts a WHERE a.quiz_id=qz.id AND a.status='submitted') attempt_count
           FROM quizzes qz JOIN subjects s ON s.id=qz.subject_id LEFT JOIN topics t ON t.id=qz.topic_id
           ORDER BY s.name, t.position""").fetchall()
    return render_template("admin/quizzes.html", quizzes=rows)


@admin_bp.route("/quizzes/<int:quiz_id>/toggle", methods=["POST"])
@role_required("admin")
def quiz_toggle(quiz_id):
    db = get_db()
    row = db.execute("SELECT is_published FROM quizzes WHERE id=?", (quiz_id,)).fetchone()
    if not row:
        abort(404)
    db.execute("UPDATE quizzes SET is_published=? WHERE id=?", (0 if row["is_published"] else 1, quiz_id))
    db.commit()
    flash("Quiz visibility updated.", "success")
    return redirect(url_for("admin.quizzes"))


@admin_bp.route("/assessments")
@role_required("admin")
def assessments():
    db = get_db()
    rows = db.execute(
        """SELECT a.*, s.name AS subject_name, (SELECT COUNT(*) FROM assessment_questions aq WHERE aq.assessment_id=a.id) q_count,
             (SELECT COUNT(*) FROM attempts at WHERE at.assessment_id=a.id AND at.status='submitted') attempt_count,
             (SELECT ROUND(AVG(percentage),1) FROM attempts at WHERE at.assessment_id=a.id AND at.status='submitted') avg_score
           FROM assessments a JOIN subjects s ON s.id=a.subject_id ORDER BY s.name""").fetchall()
    return render_template("admin/assessments.html", assessments=rows)


@admin_bp.route("/assessments/<int:assessment_id>/toggle", methods=["POST"])
@role_required("admin")
def assessment_toggle(assessment_id):
    db = get_db()
    row = db.execute("SELECT is_published FROM assessments WHERE id=?", (assessment_id,)).fetchone()
    if not row:
        abort(404)
    db.execute("UPDATE assessments SET is_published=? WHERE id=?", (0 if row["is_published"] else 1, assessment_id))
    db.commit()
    flash("Assessment visibility updated.", "success")
    return redirect(url_for("admin.assessments"))


# ---------------------------------------------------------------- results & analytics ----

@admin_bp.route("/analytics")
@role_required("admin")
def analytics():
    db = get_db()
    level_dist = _performance_distribution(db)
    by_subject = db.execute(
        """SELECT s.name, ROUND(AVG(a.percentage),1) avg_score, COUNT(*) attempts
           FROM attempts a JOIN subjects s ON s.id=a.subject_id WHERE a.status='submitted'
           GROUP BY s.id ORDER BY avg_score ASC""").fetchall()
    by_kind = db.execute(
        """SELECT kind, COUNT(*) c, ROUND(AVG(percentage),1) avg_score FROM attempts
           WHERE status='submitted' GROUP BY kind""").fetchall()
    weak_topics = db.execute(
        """SELECT t.name, s.name AS subject_name, COUNT(*) c, ROUND(AVG(lg.score),1) avg_score
           FROM learning_gaps lg JOIN topics t ON t.id=lg.topic_id JOIN subjects s ON s.id=t.subject_id
           WHERE lg.status='active' GROUP BY t.id ORDER BY c DESC LIMIT 12""").fetchall()
    trend = db.execute(
        """SELECT substr(submitted_at,1,10) day, ROUND(AVG(percentage),1) avg_score, COUNT(*) c
           FROM attempts WHERE status='submitted' GROUP BY day ORDER BY day DESC LIMIT 14""").fetchall()
    return render_template("admin/analytics.html", level_dist=level_dist, by_subject=by_subject, by_kind=by_kind,
                          weak_topics=weak_topics, trend=list(reversed(trend)))


@admin_bp.route("/learning-gaps")
@role_required("admin")
def learning_gaps():
    db = get_db()
    severity = request.args.get("severity", "")
    sql = """SELECT lg.*, u.name AS student_name, u.student_code, t.name AS topic_name, s.name AS subject_name
              FROM learning_gaps lg JOIN users u ON u.id=lg.user_id JOIN topics t ON t.id=lg.topic_id
              JOIN subjects s ON s.id=t.subject_id WHERE lg.status='active'"""
    args = []
    if severity in ("High", "Medium", "Low"):
        sql += " AND lg.severity=?"; args.append(severity)
    sql += " ORDER BY CASE lg.severity WHEN 'High' THEN 0 WHEN 'Medium' THEN 1 ELSE 2 END, lg.score ASC LIMIT 200"
    rows = db.execute(sql, args).fetchall()
    return render_template("admin/learning_gaps.html", gaps=rows, severity=severity)


@admin_bp.route("/recommendations")
@role_required("admin")
def recommendations():
    db = get_db()
    rows = db.execute(
        """SELECT r.*, u.name AS student_name, u.student_code, t.name AS topic_name
           FROM recommendations r JOIN users u ON u.id=r.user_id LEFT JOIN topics t ON t.id=r.topic_id
           WHERE r.is_active=1 ORDER BY r.priority DESC LIMIT 200""").fetchall()
    return render_template("admin/recommendations.html", recs=rows)


# ---------------------------------------------------------------- ML analytics ----

@admin_bp.route("/ml-analytics")
@role_required("admin")
def ml_analytics():
    metrics = ml_predict.load_metrics()
    return render_template("admin/ml_analytics.html", metrics=metrics)


@admin_bp.route("/ml-analytics/retrain", methods=["POST"])
@role_required("admin")
def ml_retrain():
    from ml.train_model import train
    cfg = current_app.config
    try:
        m = train(db_path=cfg["DATABASE"], model_path=cfg["MODEL_PATH"], metrics_path=cfg["METRICS_PATH"],
                  data_path=cfg["DATASET_PATH"], verbose=False)
    except ValueError as exc:
        flash(f"Retraining failed: {exc}", "error")
        return redirect(url_for("admin.ml_analytics"))
    flash(f"Retrained {len(m['model_comparison'])} models on {m['dataset_size']} students from the database; "
          f"selected {m['model_name']} (weighted F1 {m['f1']*100:.1f}%).", "success")
    return redirect(url_for("admin.ml_analytics"))


# ---------------------------------------------------------------- account management ----

@admin_bp.route("/accounts")
@role_required("admin")
def accounts():
    db = get_db()
    admins = db.execute("SELECT * FROM users WHERE role='admin' ORDER BY name").fetchall()
    return render_template("admin/accounts.html", admins=admins, me=current_user())


@admin_bp.route("/accounts/new", methods=["GET", "POST"])
@role_required("admin")
def account_new():
    if request.method == "POST":
        db = get_db()
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if len(name) < 2 or not email or len(password) < 6:
            flash("Please provide a valid name, email and a password of at least 6 characters.", "error")
            return render_template("admin/account_form.html")
        if db.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone():
            flash("An account with that email already exists.", "error")
            return render_template("admin/account_form.html")
        db.execute("INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?,?,?,'admin',?)",
                  (name, email, generate_password_hash(password), now()))
        db.commit()
        flash("Admin account created.", "success")
        return redirect(url_for("admin.accounts"))
    return render_template("admin/account_form.html")


@admin_bp.route("/accounts/<int:user_id>/deactivate", methods=["POST"])
@role_required("admin")
def account_deactivate(user_id):
    if user_id == session["user_id"]:
        flash("You cannot deactivate your own account.", "error")
        return redirect(url_for("admin.accounts"))
    db = get_db()
    db.execute("UPDATE users SET is_active=0 WHERE id=? AND role='admin'", (user_id,))
    db.commit()
    flash("Admin account deactivated.", "info")
    return redirect(url_for("admin.accounts"))
