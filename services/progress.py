"""Aggregates a student's materials + attempts into the per-topic student_progress row."""
from db import now


def recompute_topic_progress(db, user_id, topic_id):
    materials_total = db.execute("SELECT COUNT(*) c FROM materials WHERE topic_id = ?",
                                 (topic_id,)).fetchone()["c"]
    materials_done = db.execute(
        """SELECT COUNT(*) c FROM material_progress mp JOIN materials m ON m.id = mp.material_id
           WHERE mp.user_id = ? AND m.topic_id = ? AND mp.completed = 1""",
        (user_id, topic_id)).fetchone()["c"]
    questions_total = db.execute("SELECT COUNT(*) c FROM questions WHERE topic_id = ?",
                                 (topic_id,)).fetchone()["c"]

    ans = db.execute(
        """SELECT aa.is_correct FROM attempt_answers aa
           JOIN attempts a ON a.id = aa.attempt_id
           JOIN questions q ON q.id = aa.question_id
           WHERE a.user_id = ? AND q.topic_id = ? AND a.status = 'submitted' AND aa.is_correct IS NOT NULL""",
        (user_id, topic_id)).fetchall()
    graded = len(ans)
    correct = sum(1 for a in ans if a["is_correct"])
    attempted_qids = db.execute(
        """SELECT COUNT(DISTINCT aa.question_id) c FROM attempt_answers aa
           JOIN attempts a ON a.id = aa.attempt_id JOIN questions q ON q.id = aa.question_id
           WHERE a.user_id = ? AND q.topic_id = ? AND a.status = 'submitted' AND aa.selected_option IS NOT NULL""",
        (user_id, topic_id)).fetchone()["c"]
    # "questions mastered" = distinct questions the student has ever gotten right (a retake fixing
    # a past mistake counts; repeating a correct answer on retake must never inflate the total past 100%)
    distinct_correct = db.execute(
        """SELECT COUNT(DISTINCT aa.question_id) c FROM attempt_answers aa
           JOIN attempts a ON a.id = aa.attempt_id JOIN questions q ON q.id = aa.question_id
           WHERE a.user_id = ? AND q.topic_id = ? AND a.status = 'submitted' AND aa.is_correct = 1""",
        (user_id, topic_id)).fetchone()["c"]

    score = round((correct / graded) * 100, 2) if graded else None
    material_pct = (materials_done / materials_total) if materials_total else 0
    question_pct = (distinct_correct / questions_total) if questions_total else 0
    progress_pct = round(min(1.0, (material_pct * 0.35) + (question_pct * 0.65)) * 100, 2)
    completed = 1 if (materials_total and materials_done >= materials_total and questions_total
                      and distinct_correct >= questions_total) else 0

    db.execute(
        """INSERT INTO student_progress
             (user_id, topic_id, materials_done, materials_total, questions_attempted, questions_total,
              answers_graded, answers_correct, score, progress_pct, completed, last_activity)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(user_id, topic_id) DO UPDATE SET
             materials_done=excluded.materials_done, materials_total=excluded.materials_total,
             questions_attempted=excluded.questions_attempted, questions_total=excluded.questions_total,
             answers_graded=excluded.answers_graded, answers_correct=excluded.answers_correct,
             score=excluded.score, progress_pct=excluded.progress_pct, completed=excluded.completed,
             last_activity=excluded.last_activity""",
        (user_id, topic_id, materials_done, materials_total, attempted_qids, questions_total,
         graded, correct, score, progress_pct, completed, now()))
    db.commit()


def mark_material_done(db, user_id, material_id):
    row = db.execute("SELECT topic_id FROM materials WHERE id = ?", (material_id,)).fetchone()
    db.execute(
        """INSERT INTO material_progress (user_id, material_id, completed, first_accessed, last_accessed, completed_at)
           VALUES (?, ?, 1, ?, ?, ?)
           ON CONFLICT(user_id, material_id) DO UPDATE SET
             completed = 1, last_accessed = excluded.last_accessed,
             completed_at = COALESCE(material_progress.completed_at, excluded.completed_at)""",
        (user_id, material_id, now(), now(), now()))
    db.commit()
    if row:
        recompute_topic_progress(db, user_id, row["topic_id"])


def touch_material(db, user_id, material_id):
    row = db.execute("SELECT topic_id FROM materials WHERE id = ?", (material_id,)).fetchone()
    db.execute(
        """INSERT INTO material_progress (user_id, material_id, completed, first_accessed, last_accessed)
           VALUES (?, ?, 0, ?, ?)
           ON CONFLICT(user_id, material_id) DO UPDATE SET last_accessed = excluded.last_accessed""",
        (user_id, material_id, now(), now()))
    db.commit()
