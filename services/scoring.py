"""Attempt scoring: turns submitted answers into graded attempt_answers + attempt totals."""
from db import now


def grade_attempt(db, attempt_id):
    """Grade every answered row of an attempt against its question's correct_option,
    then roll the totals up onto the attempts row. Safe to call multiple times."""
    rows = db.execute(
        """SELECT aa.id, aa.selected_option, q.correct_option, q.marks
           FROM attempt_answers aa JOIN questions q ON q.id = aa.question_id
           WHERE aa.attempt_id = ?""", (attempt_id,)).fetchall()

    total_marks = 0.0
    obtained = 0.0
    answered = 0
    correct = 0
    for r in rows:
        total_marks += r["marks"]
        if r["selected_option"] is None:
            db.execute("UPDATE attempt_answers SET is_correct = NULL, marks_awarded = 0 WHERE id = ?", (r["id"],))
            continue
        answered += 1
        is_ok = 1 if r["selected_option"] == r["correct_option"] else 0
        marks_awarded = r["marks"] if is_ok else 0
        if is_ok:
            correct += 1
        db.execute("UPDATE attempt_answers SET is_correct = ?, marks_awarded = ? WHERE id = ?",
                   (is_ok, marks_awarded, r["id"]))
        obtained += marks_awarded

    percentage = round((obtained / total_marks) * 100, 2) if total_marks else 0.0
    accuracy = round((correct / answered) * 100, 2) if answered else 0.0

    db.execute(
        """UPDATE attempts SET total_marks = ?, obtained_marks = ?, percentage = ?, accuracy = ?
           WHERE id = ?""", (total_marks, obtained, percentage, accuracy, attempt_id))
    db.commit()
    return {"total_marks": total_marks, "obtained_marks": obtained, "percentage": percentage,
            "accuracy": accuracy, "answered": answered, "correct": correct, "count": len(rows)}


def submit_attempt(db, attempt_id, answers, duration_sec=0):
    """answers: {question_id: 'A'/'B'/'C'/'D'}. Writes selections, grades, closes the attempt."""
    for qid, opt in answers.items():
        if opt not in ("A", "B", "C", "D"):
            continue
        db.execute(
            """UPDATE attempt_answers SET selected_option = ?, answered_at = ?
               WHERE attempt_id = ? AND question_id = ?""", (opt, now(), attempt_id, qid))
    result = grade_attempt(db, attempt_id)
    db.execute("UPDATE attempts SET status = 'submitted', submitted_at = ?, duration_sec = ? WHERE id = ?",
               (now(), duration_sec, attempt_id))
    db.commit()
    return result
