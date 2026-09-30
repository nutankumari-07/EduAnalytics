"""Learning-gap detection.

Looks at a student's actual graded answers per topic — overall accuracy, accuracy broken
down by question difficulty, how many attempts touched the topic, and questions the
student has gotten wrong more than once — and turns that into a severity + a plain-English
explanation + concrete suggested actions. This is the analysis surfaced on the student's
"Learning Gaps" page and the admin's gap analytics.
"""
import json

from db import jload, now


def _topic_stats(db, user_id, topic_id):
    rows = db.execute(
        """SELECT aa.question_id, aa.is_correct, q.difficulty, a.submitted_at
           FROM attempt_answers aa
           JOIN attempts a ON a.id = aa.attempt_id
           JOIN questions q ON q.id = aa.question_id
           WHERE a.user_id = ? AND q.topic_id = ? AND a.status = 'submitted'
                 AND aa.selected_option IS NOT NULL
           ORDER BY a.submitted_at""", (user_id, topic_id)).fetchall()
    if not rows:
        return None

    total = len(rows)
    correct = sum(1 for r in rows if r["is_correct"])
    accuracy = round((correct / total) * 100, 1)

    by_diff = {}
    for diff in ("Easy", "Medium", "Hard"):
        sub = [r for r in rows if r["difficulty"] == diff]
        if sub:
            by_diff[diff] = round((sum(1 for r in sub if r["is_correct"]) / len(sub)) * 100, 1)

    wrong_counts = {}
    for r in rows:
        if not r["is_correct"]:
            wrong_counts[r["question_id"]] = wrong_counts.get(r["question_id"], 0) + 1
    repeated_misses = sum(1 for c in wrong_counts.values() if c >= 2)

    attempts_count = db.execute(
        """SELECT COUNT(DISTINCT a.id) c FROM attempts a JOIN attempt_answers aa ON aa.attempt_id = a.id
           JOIN questions q ON q.id = aa.question_id
           WHERE a.user_id = ? AND q.topic_id = ? AND a.status = 'submitted'""",
        (user_id, topic_id)).fetchone()["c"]

    return {"accuracy": accuracy, "by_diff": by_diff, "attempts_count": attempts_count,
            "repeated_misses": repeated_misses, "total_answers": total,
            "last_submitted": rows[-1]["submitted_at"]}


def _severity(stats):
    acc = stats["accuracy"]
    hard_weak = stats["by_diff"].get("Hard", 100) < 40 or stats["by_diff"].get("Medium", 100) < 45
    if acc < 50 or (acc < 60 and stats["repeated_misses"] >= 2):
        return "High"
    if acc < 72 or (acc < 80 and hard_weak):
        return "Medium"
    if acc < 82:
        return "Low"
    return None


def _explain(topic_name, stats):
    parts = [f"Scoring {stats['accuracy']}% on {topic_name} across {stats['attempts_count']} "
             f"attempt(s) ({stats['total_answers']} questions answered)."]
    weak_diffs = [d for d, a in stats["by_diff"].items() if a < 55]
    if weak_diffs:
        detail = ", ".join(f"{d} {stats['by_diff'][d]}%" for d in weak_diffs)
        parts.append(f"Particularly weak on {detail}-difficulty questions.")
    if stats["repeated_misses"]:
        parts.append(f"{stats['repeated_misses']} question(s) have been answered incorrectly more than once, "
                     "suggesting the underlying concept hasn't been fixed yet.")
    return " ".join(parts)


def detect_for_topic(db, user_id, topic_id, topic_name, concepts):
    stats = _topic_stats(db, user_id, topic_id)
    if stats is None:
        return None
    severity = _severity(stats)
    if severity is None:
        db.execute("UPDATE learning_gaps SET status = 'resolved', updated_at = ? WHERE user_id = ? AND topic_id = ?",
                  (now(), user_id, topic_id))
        db.commit()
        return None

    concept_names = [c[0] for c in concepts] if concepts else []
    actions = [f"Revisit the '{c}' concept notes" for c in concept_names[:2]]
    actions.append("Retake a short practice set focused on this topic")
    if stats["by_diff"].get("Hard", 100) < 45:
        actions.append("Attempt Hard-difficulty questions only, to target the weakest band")

    db.execute(
        """INSERT INTO learning_gaps (user_id, topic_id, severity, score, attempts_count, observed_issue,
                                      suggested_actions, status, detected_at, updated_at)
           VALUES (?,?,?,?,?,?,?, 'active', ?, ?)
           ON CONFLICT(user_id, topic_id) DO UPDATE SET
             severity=excluded.severity, score=excluded.score, attempts_count=excluded.attempts_count,
             observed_issue=excluded.observed_issue, suggested_actions=excluded.suggested_actions,
             status='active', updated_at=excluded.updated_at""",
        (user_id, topic_id, severity, stats["accuracy"], stats["attempts_count"],
         _explain(topic_name, stats), json.dumps(actions), now(), now()))
    db.commit()
    return severity


def detect_all_for_user(db, user_id):
    """Re-run gap detection for every topic the user has ever answered a question in."""
    topics = db.execute(
        """SELECT DISTINCT t.id, t.name, t.concepts FROM topics t
           JOIN questions q ON q.topic_id = t.id
           JOIN attempt_answers aa ON aa.question_id = q.id
           JOIN attempts a ON a.id = aa.attempt_id
           WHERE a.user_id = ? AND a.status = 'submitted'""", (user_id,)).fetchall()
    found = []
    for t in topics:
        concepts = jload(t["concepts"], [])
        sev = detect_for_topic(db, user_id, t["id"], t["name"], concepts)
        if sev:
            found.append((t["id"], sev))
    return found
