"""Populate the database with real curriculum content, a real 395-question bank, demo
accounts and a batch of simulated student activity so no dashboard is ever empty.

Run:  python -m scripts.seed_data           (from the project root, after init_db)
"""
import json
import os
import random
import sys
from datetime import datetime, timedelta
from urllib.parse import quote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from werkzeug.security import generate_password_hash

from config import Config
from db import connect, init_schema, now
from scripts.data.curriculum import SUBJECTS
from scripts import seed_students
from scripts.data import q_dsa, q_oop, q_dbms, q_cn, q_os, q_se, q_aiml, q_math, q_tc
from services import gaps as gaps_svc
from services import progress as progress_svc
from services import recommend as recommend_svc
from services.scoring import grade_attempt

QMAP = {"dsa": q_dsa.QUESTIONS, "oop": q_oop.QUESTIONS, "dbms": q_dbms.QUESTIONS, "cn": q_cn.QUESTIONS,
        "os": q_os.QUESTIONS, "se": q_se.QUESTIONS, "aiml": q_aiml.QUESTIONS, "math": q_math.QUESTIONS,
        "tc": q_tc.QUESTIONS}
MARKS_BY_DIFF = {"Easy": 1, "Medium": 2, "Hard": 3}
RNG = random.Random(2024)

FIRST_NAMES = ["Aarav", "Ishita", "Rohan", "Sneha", "Vikram", "Ananya", "Karthik", "Meera", "Arjun",
               "Divya", "Rahul", "Priya", "Siddharth", "Neha", "Aditya", "Pooja", "Kunal", "Riya"]
LAST_NAMES = ["Sharma", "Verma", "Iyer", "Reddy", "Nair", "Gupta", "Menon", "Rao", "Kapoor", "Das",
              "Singh", "Pillai", "Joshi", "Bhat", "Mehta", "Chatterjee", "Naidu", "Kulkarni"]


def ts_days_ago(days, hour=None):
    d = datetime.now() - timedelta(days=days, hours=RNG.randint(0, 20) if hour is None else 0)
    return d.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- curriculum ----

def seed_curriculum(db):
    """Returns {subject_key: {'id':.., 'topics': {topic_name: {'id':.., 'concepts':[...]}}}}"""
    ref = {}
    for pos, s in enumerate(SUBJECTS):
        cur = db.execute(
            "INSERT INTO subjects (name, category, description, image_path, created_at) VALUES (?,?,?,?,?)",
            (s["name"], s["category"], s["description"], f"images/courses/{s['key']}.jpg", now()))
        subject_id = cur.lastrowid
        ref[s["key"]] = {"id": subject_id, "name": s["name"], "topics": {}}

        qbank = QMAP[s["key"]]
        for t_pos, (t_name, diff, summary, concepts, wiki_term) in enumerate(s["topics"]):
            cur = db.execute(
                """INSERT INTO topics (subject_id, name, description, difficulty, position, concepts)
                   VALUES (?,?,?,?,?,?)""",
                (subject_id, t_name, summary, diff, t_pos, json.dumps(concepts)))
            topic_id = cur.lastrowid
            ref[s["key"]]["topics"][t_name] = {"id": topic_id, "concepts": concepts}

            # -- materials: in-app notes (no URL, zero 404 risk) + a real Wikipedia article + a practice CTA
            notes_body = summary + "\n\n" + "\n\n".join(f"**{c}** — {expl}" for c, expl in concepts)
            db.execute(
                """INSERT INTO materials (topic_id, title, description, type, url, content, est_minutes,
                                          is_recommended, position, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (topic_id, f"{t_name} — Study Notes", f"Core concepts of {t_name}, written for quick review.",
                 "notes", None, notes_body, 12, 1, 0, now()))
            db.execute(
                """INSERT INTO materials (topic_id, title, description, type, url, content, est_minutes,
                                          is_recommended, position, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (topic_id, f"{t_name} — Reference Reading", f"Background reading on {t_name} from Wikipedia.",
                 "article", f"https://en.wikipedia.org/wiki/{quote(wiki_term.replace(' ', '_'))}", None, 10,
                 0, 1, now()))
            db.execute(
                """INSERT INTO materials (topic_id, title, description, type, url, content, est_minutes,
                                          is_recommended, position, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (topic_id, f"{t_name} — Practice Set", "Five graded questions to test what you've just read.",
                 "practice", None, None, 8, 0, 2, now()))

            # -- questions, options shuffled deterministically per question
            qs = qbank[t_name]
            for qi, (diff_q, text, correct, wrongs, expl) in enumerate(qs):
                opts = [correct] + list(wrongs)
                order = list(range(4))
                RNG.shuffle(order)
                shuffled = [opts[i] for i in order]
                correct_letter = "ABCD"[shuffled.index(correct)]
                db.execute(
                    """INSERT INTO questions (topic_id, text, option_a, option_b, option_c, option_d,
                                              correct_option, explanation, difficulty, marks, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (topic_id, text, shuffled[0], shuffled[1], shuffled[2], shuffled[3], correct_letter,
                     expl, diff_q, MARKS_BY_DIFF[diff_q], now()))
    db.commit()
    return ref


def seed_quizzes_and_assessments(db, ref):
    """One quiz per topic (its 5 authored questions) + one mixed, timed assessment per subject."""
    for key, sinfo in ref.items():
        for t_name, tinfo in sinfo["topics"].items():
            qids = [r["id"] for r in db.execute(
                "SELECT id FROM questions WHERE topic_id = ? ORDER BY id", (tinfo["id"],)).fetchall()]
            cur = db.execute(
                """INSERT INTO quizzes (title, description, subject_id, topic_id, difficulty, is_published, created_at)
                   VALUES (?,?,?,?,?,1,?)""",
                (f"{t_name} Quiz", f"A short quiz covering {t_name}.", sinfo["id"], tinfo["id"], "Mixed", now()))
            quiz_id = cur.lastrowid
            for pos, qid in enumerate(qids):
                db.execute("INSERT INTO quiz_questions (quiz_id, question_id, position) VALUES (?,?,?)",
                          (quiz_id, qid, pos))

        # subject-wide timed assessment: round-robin 2 questions per topic, capped at 24
        all_topic_qids = []
        for t_name, tinfo in sinfo["topics"].items():
            qids = [r["id"] for r in db.execute(
                "SELECT id FROM questions WHERE topic_id = ? ORDER BY id LIMIT 2", (tinfo["id"],)).fetchall()]
            all_topic_qids.append(qids)
        mixed = []
        i = 0
        while len(mixed) < 24 and any(all_topic_qids):
            for group in all_topic_qids:
                if i < len(group):
                    mixed.append(group[i])
            i += 1
            if i > 2:
                break
        time_limit = max(20, round(len(mixed) * 1.5))
        cur = db.execute(
            """INSERT INTO assessments (title, subject_id, instructions, difficulty, time_limit_min,
                                        is_published, created_at) VALUES (?,?,?,?,?,1,?)""",
            (f"{sinfo['name']} — Comprehensive Assessment",
             sinfo["id"], "Timed, mixed-difficulty assessment covering every topic in this subject. "
                         "Each question carries marks based on its difficulty. You cannot pause the timer.",
             "Mixed", time_limit, now()))
        assess_id = cur.lastrowid
        for pos, qid in enumerate(mixed):
            db.execute("INSERT INTO assessment_questions (assessment_id, question_id, position) VALUES (?,?,?)",
                      (assess_id, qid, pos))
    db.commit()


# ---------------------------------------------------------------- users ----

def make_admin(db):
    return db.execute(
        """INSERT INTO users (name, email, password_hash, role, branch, semester, created_at, last_login)
           VALUES ('Admin User','admin@eduanalytics.com',?, 'admin', 'Administration', NULL, ?, ?)""",
        (generate_password_hash("admin123"), ts_days_ago(200), ts_days_ago(0))).lastrowid


def seed_notifications(db, students):
    for uid, _ in students:
        db.execute(
            """INSERT INTO notifications (user_id, kind, title, message, url, is_read, created_at)
               VALUES (?, 'welcome', 'Welcome to EduAnalytics',
                       'Your personalized dashboard is ready — explore your subjects to get started.',
                       '/student/dashboard', 1, ?)""", (uid, ts_days_ago(RNG.randint(60, 150))))
        top_gap = db.execute(
            """SELECT lg.severity, t.name FROM learning_gaps lg JOIN topics t ON t.id = lg.topic_id
               WHERE lg.user_id=? AND lg.status='active'
               ORDER BY CASE lg.severity WHEN 'High' THEN 0 WHEN 'Medium' THEN 1 ELSE 2 END LIMIT 1""",
            (uid,)).fetchone()
        if top_gap:
            db.execute(
                """INSERT INTO notifications (user_id, kind, title, message, url, is_read, created_at)
                   VALUES (?, 'learning_gap', ?, ?, '/student/learning-gaps', 0, ?)""",
                (uid, f"{top_gap['severity']} gap detected in {top_gap['name']}",
                 f"Your recent attempts show a {top_gap['severity'].lower()}-severity gap in {top_gap['name']}. "
                 "Check your recommendations for what to study next.", ts_days_ago(RNG.randint(1, 10))))
        last_attempt = db.execute(
            """SELECT title, percentage FROM attempts WHERE user_id=? AND status='submitted'
               ORDER BY submitted_at DESC LIMIT 1""", (uid,)).fetchone()
        if last_attempt:
            db.execute(
                """INSERT INTO notifications (user_id, kind, title, message, url, is_read, created_at)
                   VALUES (?, 'result', 'Result available', ?, '/student/history', 1, ?)""",
                (uid, f"You scored {last_attempt['percentage']}% on {last_attempt['title']}.",
                 ts_days_ago(RNG.randint(0, 5))))
    db.commit()


def build(n_students, db_path, model_path=None, metrics_path=None, data_path=None, creds_path=None, log=print):
    """Seed a complete database (curriculum, students, activity, gaps, trained model, recommendations)."""
    from ml.train_model import train
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db = connect(db_path)
    db.execute("PRAGMA synchronous = OFF")           # seeding-only speed-up; DB is rebuilt from scratch anyway
    db.execute("PRAGMA journal_mode = MEMORY")
    init_schema(db)
    if db.execute("SELECT COUNT(*) FROM subjects").fetchone()[0]:
        raise SystemExit("Database already seeded. Run `python -m scripts.init_db --reset` first for a clean rebuild.")

    log("seeding curriculum (9 subjects, 79 topics, 395 questions)...")
    ref = seed_curriculum(db)
    log("seeding quizzes + assessments...")
    seed_quizzes_and_assessments(db, ref)
    log("seeding admin...")
    make_admin(db)
    rng = random.Random(seed_students.SEED)
    profiles = seed_students.build_profiles(n_students)
    log(f"creating {n_students + 1} student accounts...")
    students, cred_rows = seed_students.insert_students(db, profiles, rng)
    if creds_path:
        seed_students.write_credentials(cred_rows, creds_path)
    log("simulating enrollments, quiz / practice / assessment attempts, topic + material progress...")
    seed_students.simulate_activity(db, students, rng,
                                    progress_cb=lambda n, total: log(f"  {n}/{total} students simulated"))
    log("detecting learning gaps (threshold-based, per topic)...")
    for uid, _ in students:
        gaps_svc.detect_all_for_user(db, uid)
    db.commit()
    db.close()

    log("training + comparing ML models on the seeded database...")
    metrics = train(db_path=db_path, model_path=model_path, metrics_path=metrics_path, data_path=data_path,
                    verbose=True)

    db = connect(db_path)
    db.execute("PRAGMA synchronous = OFF")
    log("generating recommendations (learning gaps + ML performance level)...")
    for uid, _ in students:
        recommend_svc.generate_for_user(db, uid)
    log("seeding notifications...")
    seed_notifications(db, students)
    counts = {t: db.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"] for t in
              ("subjects", "topics", "questions", "attempts", "attempt_answers", "student_progress",
               "learning_gaps", "recommendations")}
    n_stu = db.execute("SELECT COUNT(*) c FROM users WHERE role='student'").fetchone()["c"]
    db.close()
    log(f"done: {n_stu} students, " + ", ".join(f"{v} {k}" for k, v in counts.items()))
    return metrics


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", type=int, default=seed_students.N_STUDENTS,
                    help="number of generated students (plus the demo student)")
    args = ap.parse_args()
    build(args.students, Config.DATABASE, Config.MODEL_PATH, Config.METRICS_PATH, Config.DATASET_PATH,
          os.path.join(ROOT, "data", "seeded_students.csv"))


if __name__ == "__main__":
    main()
