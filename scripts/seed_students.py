"""Programmatic generation of 1000+ synthetic students with realistic, *different* academic data.

Approach (adapted from the reference project's generator): every student gets latent traits
(ability, effort, engagement). Profile numbers (attendance, study hours, GPA) are driven by those
traits plus noise. Quiz / practice / assessment answers are then simulated question-by-question
from the student's per-topic skill, and everything else (topic scores, progress, gaps) is derived
from those answers by the real application services. A separate end-of-term outcome
(student_outcomes) is the ML target; it is NOT a formula of the model features.
"""
import csv
import math
import os
import random
from datetime import datetime, timedelta

import numpy as np
from werkzeug.security import generate_password_hash

from db import now
from services import progress as progress_svc

N_STUDENTS = 1000
DEFAULT_PASSWORD = "student123"
SEED = 2024

FIRST = ["Aarav", "Ishita", "Rohan", "Sneha", "Vikram", "Ananya", "Karthik", "Meera", "Arjun", "Divya",
         "Rahul", "Priya", "Siddharth", "Neha", "Aditya", "Pooja", "Kunal", "Riya", "Nikhil", "Kavya",
         "Manish", "Shreya", "Varun", "Tanvi", "Harsh", "Aishwarya", "Rajat", "Simran", "Pranav", "Nandini",
         "Abhishek", "Lakshmi", "Suresh", "Deepa", "Gaurav", "Swati", "Mohit", "Anjali", "Yash", "Bhavna",
         "Tarun", "Sanjana", "Dhruv", "Mansi", "Vivek", "Radhika", "Ayush", "Komal", "Naveen", "Trisha"]
LAST = ["Sharma", "Verma", "Iyer", "Reddy", "Nair", "Gupta", "Menon", "Rao", "Kapoor", "Das",
        "Singh", "Pillai", "Joshi", "Bhat", "Mehta", "Chatterjee", "Naidu", "Kulkarni", "Patel", "Shetty",
        "Agarwal", "Banerjee", "Chopra", "Desai", "Fernandes", "Ghosh", "Hegde", "Jain", "Khan", "Lal",
        "Malhotra", "Nambiar", "Oberoi", "Prasad", "Rastogi", "Saxena", "Trivedi", "Upadhyay", "Varma", "Yadav"]
DIFF_BIAS = {"Easy": 1.0, "Medium": 0.1, "Hard": -0.9}
TOPIC_LEVEL_PENALTY = {"Beginner": 0.0, "Intermediate": -0.15, "Advanced": -0.35}
LOW_MAX, HIGH_MIN = 50.0, 72.0        # end-of-term score -> Low / Medium / High


def _clip(x, lo, hi):
    return float(min(hi, max(lo, x)))


def _ts(days_ago, hour=None, rng=None):
    # always draw the same number of random values so the seed stream never depends on the wall clock
    h, m, sec, fallback = rng.randint(8, 23), rng.randint(0, 59), rng.randint(0, 59), rng.randint(5, 600)
    now_ = datetime.now().replace(microsecond=0)
    dt = (now_ - timedelta(days=days_ago)).replace(hour=hour if hour is not None else h, minute=m, second=sec)
    if dt > now_:
        dt = now_ - timedelta(minutes=fallback)
    return dt


def _fmt(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def p_correct(skill, difficulty):
    z = 1.5 * skill + DIFF_BIAS[difficulty]
    return 0.25 + 0.75 / (1.0 + math.exp(-z))


def build_profiles(n, seed=SEED):
    """Latent traits + observable profile for n students (no DB access)."""
    g = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ability, effort, engagement = g.normal(), g.normal(), g.normal()
        attendance = _clip(78 + 6 * effort + 4 * ability + g.normal(0, 7), 40, 100)
        hours = _clip(10 + 3.5 * effort + 1.5 * ability + g.normal(0, 2.5), 1, 35)
        gpa = _clip(6.9 + 0.9 * ability + 0.25 * effort + g.normal(0, 0.5), 4.0, 10.0)
        final = _clip(60 + 14 * ability + 6 * effort + g.normal(0, 6.5), 5, 99)
        out.append({"ability": float(ability), "effort": float(effort), "engagement": float(engagement),
                    "attendance": round(attendance, 1), "hours": round(hours, 1), "gpa": round(gpa, 2),
                    "final_score": round(final, 1)})
    return out


def level_for(score):
    return "Low" if score < LOW_MAX else ("High" if score >= HIGH_MIN else "Medium")


def insert_students(db, profiles, rng):
    """Creates the demo student + len(profiles) generated students. Returns [(user_id, profile)]."""
    pw_hash = generate_password_hash(DEFAULT_PASSWORD)          # one hash reused for the demo cohort
    used, rows, created = set(), [], []
    demo = {"ability": 0.15, "effort": 0.1, "engagement": 1.2, "attendance": 84.0, "hours": 13.0,
            "gpa": 7.6, "final_score": 66.0}
    people = [("Demo Student", "student@eduanalytics.com", "STU2024001", demo, 120)]
    for i, p in enumerate(profiles, start=1):
        first, last = rng.choice(FIRST), rng.choice(LAST)
        email = f"{first.lower()}.{last.lower()}{i}@eduanalytics.com"
        people.append((f"{first} {last}", email, f"STU2024{i:04d}", p, rng.randint(30, 180)))
    for name, email, code, p, age in people:
        assert email not in used and code not in used
        used.update((email, code))
        cur = db.execute(
            """INSERT INTO users (student_code, name, email, password_hash, role, branch, semester, phone,
                                  attendance_percentage, study_hours_per_week, previous_gpa, created_at, last_login)
               VALUES (?,?,?,?, 'student', 'Computer Science & Engineering', ?, ?, ?, ?, ?, ?, ?)""",
            (code, name, email, pw_hash, rng.choice([3, 4, 5, 6, 7]), f"9{rng.randint(100000000, 999999999)}",
             p["attendance"], p["hours"], p["gpa"], _fmt(_ts(age, rng=rng)), _fmt(_ts(rng.randint(0, 6), rng=rng))))
        created.append((cur.lastrowid, p))
        rows.append((code, name, email, DEFAULT_PASSWORD))
    db.commit()
    return created, rows


def write_credentials(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["student_code", "name", "email", "password"])
        w.writerows(rows)


def _load_catalog(db):
    cat = {"topics": {}, "subjects": {}, "quiz_for_topic": {}, "assess_for_subject": {}}
    for t in db.execute("SELECT id, subject_id, difficulty, name FROM topics ORDER BY position, id"):
        cat["topics"][t["id"]] = {"subject_id": t["subject_id"], "difficulty": t["difficulty"], "name": t["name"],
                                  "questions": [], "materials": []}
        cat["subjects"].setdefault(t["subject_id"], []).append(t["id"])
    for q in db.execute("SELECT id, topic_id, correct_option, difficulty, marks FROM questions ORDER BY id"):
        cat["topics"][q["topic_id"]]["questions"].append(dict(q))
    for m in db.execute("SELECT id, topic_id, type FROM materials"):
        cat["topics"][m["topic_id"]]["materials"].append(dict(m))
    for q in db.execute("SELECT id, topic_id FROM quizzes WHERE topic_id IS NOT NULL"):
        cat["quiz_for_topic"][q["topic_id"]] = q["id"]
    for a in db.execute("SELECT id, subject_id, time_limit_min, title FROM assessments"):
        qids = [r[0] for r in db.execute(
            "SELECT question_id FROM assessment_questions WHERE assessment_id=? ORDER BY position", (a["id"],))]
        qmap = {q["id"]: q for t in cat["topics"].values() for q in t["questions"]}
        cat["assess_for_subject"][a["subject_id"]] = {
            "id": a["id"], "limit": a["time_limit_min"], "title": a["title"],
            "questions": [dict(qmap[i]) for i in qids]}
    return cat


def _record_attempt(db, uid, kind, questions, skill_of, started, rng, *, quiz_id=None, assessment_id=None,
                    subject_id=None, topic_id=None, title=None, time_limit_sec=None, blank_prob=0.0):
    total = obtained = correct = answered = 0
    answers = []
    for pos, q in enumerate(questions):
        total += q["marks"]
        if blank_prob and rng.random() < blank_prob:
            answers.append((pos, q, None, None, 0))
            continue
        answered += 1
        if rng.random() < p_correct(skill_of(q), q["difficulty"]):
            sel, ok = q["correct_option"], 1
            obtained += q["marks"]
            correct += 1
        else:
            sel, ok = rng.choice([o for o in "ABCD" if o != q["correct_option"]]), 0
        answers.append((pos, q, sel, ok, q["marks"] if ok else 0))
    pct = round(obtained / total * 100, 2) if total else 0.0
    acc = round(correct / answered * 100, 2) if answered else 0.0
    dur = sum(rng.randint(25, 80) for _ in questions)
    submitted = min(started + timedelta(seconds=dur), datetime.now().replace(microsecond=0))
    cur = db.execute(
        """INSERT INTO attempts (user_id, kind, quiz_id, assessment_id, subject_id, topic_id, title, difficulty,
                                 started_at, submitted_at, status, total_marks, obtained_marks, percentage,
                                 accuracy, duration_sec, time_limit_sec)
           VALUES (?,?,?,?,?,?,?, 'Mixed', ?,?, 'submitted', ?,?,?,?,?,?)""",
        (uid, kind, quiz_id, assessment_id, subject_id, topic_id, title, _fmt(started), _fmt(submitted),
         total, obtained, pct, acc, dur, time_limit_sec))
    aid = cur.lastrowid
    db.executemany(
        """INSERT INTO attempt_answers (attempt_id, question_id, position, selected_option, correct_option,
                                        is_correct, marks_awarded, answered_at) VALUES (?,?,?,?,?,?,?,?)""",
        [(aid, q["id"], pos, sel, q["correct_option"], ok, aw, _fmt(started) if sel else None)
         for pos, q, sel, ok, aw in answers])
    return _fmt(submitted)


def simulate_activity(db, students, rng, progress_cb=None):
    """students: [(user_id, profile)]. Inserts enrollments, attempts, answers, material progress,
    topic progress and the end-of-term outcome for every student."""
    cat = _load_catalog(db)
    subject_ids = list(cat["subjects"])
    last_touch = {}
    for n, (uid, p) in enumerate(students, start=1):
        window = rng.randint(35, 110)
        n_sub = len(subject_ids) if uid == students[0][0] else rng.randint(5, len(subject_ids))
        chosen = subject_ids if n_sub == len(subject_ids) else rng.sample(subject_ids, n_sub)
        db.executemany("INSERT INTO enrollments (user_id, subject_id, enrolled_at) VALUES (?,?,?)",
                       [(uid, sid, _fmt(_ts(window + rng.randint(1, 30), rng=rng))) for sid in chosen])
        affinity = {sid: rng.gauss(0, 0.4) for sid in chosen}
        topic_off, touched = {}, set()
        n_quiz = n_assess = 0
        p_material = _clip(0.55 + 0.18 * p["effort"], 0.15, 0.97)
        for sid in chosen:
            tids = list(cat["subjects"][sid])
            rng.shuffle(tids)
            k = int(_clip(round(rng.gauss(3.6 + 1.4 * p["engagement"], 1.0)), 2, len(tids)))
            for tid in tids[:k]:
                topic = cat["topics"][tid]
                topic_off[tid] = rng.gauss(0, 0.55)
                base = rng.uniform(1, window)
                t_frac = 1 - base / window
                skill = (p["ability"] + affinity[sid] + topic_off[tid] + TOPIC_LEVEL_PENALTY[topic["difficulty"]]
                         + 0.35 * t_frac)
                touched.add(tid)
                qs = topic["questions"]
                sk = lambda q, s=skill: s
                # practice (+ occasional retake, after which the student has learned a little)
                reps = 1 + (rng.random() < 0.25 + 0.1 * max(p["engagement"], 0))
                for r in range(reps):
                    off = max(0.05, base - r * rng.uniform(1, 4))
                    ts = _record_attempt(db, uid, "practice", qs, lambda q, s=skill + 0.2 * r: s,
                                         _ts(off, rng=rng), rng, subject_id=sid, topic_id=tid,
                                         title=f"{topic['name']} Practice")
                    last_touch[(uid, tid)] = max(last_touch.get((uid, tid), ""), ts)
                if rng.random() < 0.55 + 0.2 * max(min(p["engagement"], 1), -1):
                    ts = _record_attempt(db, uid, "quiz", qs, sk, _ts(max(0.05, base - rng.uniform(0.3, 3)), rng=rng),
                                         rng, quiz_id=cat["quiz_for_topic"].get(tid), subject_id=sid, topic_id=tid,
                                         title=f"{topic['name']} Quiz")
                    last_touch[(uid, tid)] = max(last_touch[(uid, tid)], ts)
                    n_quiz += 1
                for m in topic["materials"]:
                    if m["type"] in ("notes", "article"):
                        done = rng.random() < p_material
                    else:
                        done = rng.random() < p_material * 0.9
                    started = _fmt(_ts(base, rng=rng))
                    db.execute(
                        """INSERT OR IGNORE INTO material_progress (user_id, material_id, completed, first_accessed,
                                                                    last_accessed, completed_at) VALUES (?,?,?,?,?,?)""",
                        (uid, m["id"], int(done), started, started, started if done else None))
            a = cat["assess_for_subject"].get(sid)
            if a and (rng.random() < 0.55):
                _assess(db, uid, a, sid, p, affinity, topic_off, window, rng, cat)
                n_assess += 1
        if n_quiz == 0:                       # guarantee minimum real activity for every seeded student
            tid = next(iter(touched))
            t = cat["topics"][tid]
            skill = p["ability"] + topic_off[tid]
            _record_attempt(db, uid, "quiz", t["questions"], lambda q, s=skill: s, _ts(2, rng=rng), rng,
                            quiz_id=cat["quiz_for_topic"].get(tid), subject_id=t["subject_id"], topic_id=tid,
                            title=f"{t['name']} Quiz")
        if n_assess == 0:
            sid = chosen[0]
            _assess(db, uid, cat["assess_for_subject"][sid], sid, p, affinity, topic_off, window, rng, cat)
        for tid in touched:
            progress_svc.recompute_topic_progress(db, uid, tid)
        db.execute("INSERT INTO student_outcomes (user_id, final_score, performance_level, recorded_at) VALUES (?,?,?,?)",
                   (uid, p["final_score"], level_for(p["final_score"]), now()))
        db.commit()
        if progress_cb and n % 100 == 0:
            progress_cb(n, len(students))
    db.executemany("UPDATE student_progress SET last_activity=? WHERE user_id=? AND topic_id=?",
                   [(ts, u, t) for (u, t), ts in last_touch.items()])
    db.commit()


def _assess(db, uid, a, sid, p, affinity, topic_off, window, rng, cat):
    base = rng.uniform(1, window)
    t_frac = 1 - base / window

    def skill_of(q):
        tid = q["topic_id"]
        off = topic_off.get(tid)
        if off is None:
            off = topic_off[tid] = rng.gauss(0, 0.55)
        return (p["ability"] + affinity.get(sid, 0) + off + 0.35 * t_frac
                + TOPIC_LEVEL_PENALTY[cat["topics"][tid]["difficulty"]])
    _record_attempt(db, uid, "assessment", a["questions"], skill_of, _ts(base, rng=rng), rng,
                    assessment_id=a["id"], subject_id=sid, title="Comprehensive Assessment",
                    time_limit_sec=a["limit"] * 60, blank_prob=0.03)
