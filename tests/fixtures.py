"""Builds an isolated Flask app + temp SQLite DB with small, deterministic fixture data,
so the test suite never touches the real development database."""
import json
import os
import tempfile

from werkzeug.security import generate_password_hash

from app import create_app
from config import Config
from db import connect, init_schema, now


class TestConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = False


def make_app():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    TestConfig.DATABASE = path

    conn = connect(path)
    init_schema(conn)

    admin_id = conn.execute(
        "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?,?,?,'admin',?)",
        ("Test Admin", "admin@test.local", generate_password_hash("adminpass"), now())).lastrowid
    student_id = conn.execute(
        """INSERT INTO users (student_code, name, email, password_hash, role, attendance_percentage,
                              study_hours_per_week, previous_gpa, created_at)
           VALUES ('STU001','Test Student','student@test.local',?, 'student', 80, 10, 7.5, ?)""",
        (generate_password_hash("studentpass"), now())).lastrowid

    subject_id = conn.execute(
        "INSERT INTO subjects (name, category, description, image_path, created_at) VALUES (?,?,?,?,?)",
        ("Test Subject", "Testing", "A subject used only by the automated test suite.",
         "images/courses/default.jpg", now())).lastrowid
    topic_id = conn.execute(
        """INSERT INTO topics (subject_id, name, description, difficulty, position, concepts)
           VALUES (?,?,?,?,?,?)""",
        (subject_id, "Test Topic", "A topic used only by the automated test suite.", "Beginner", 0,
         json.dumps([["Concept One", "Explanation one."], ["Concept Two", "Explanation two."]]))).lastrowid

    conn.execute(
        """INSERT INTO materials (topic_id, title, description, type, url, content, est_minutes,
                                  is_recommended, position, created_at)
           VALUES (?,'Test Notes','Notes for testing','notes',NULL,'Some **bold** notes content.',5,1,0,?)""",
        (topic_id, now()))
    conn.execute(
        """INSERT INTO materials (topic_id, title, description, type, url, content, est_minutes,
                                  is_recommended, position, created_at)
           VALUES (?,'Test Practice','Practice for testing','practice',NULL,NULL,5,0,1,?)""",
        (topic_id, now()))

    # 5 questions: 2 correct-friendly (Easy), 3 designed so a fixed wrong-answer pattern fails them
    question_ids = []
    specs = [
        ("Easy", "1 + 1 = ?", "2", ["3", "4", "5"]),
        ("Easy", "2 + 2 = ?", "4", ["3", "5", "6"]),
        ("Medium", "Capital of France?", "Paris", ["Rome", "Berlin", "Madrid"]),
        ("Medium", "Largest planet?", "Jupiter", ["Mars", "Venus", "Saturn"]),
        ("Hard", "sqrt(144) = ?", "12", ["10", "11", "14"]),
    ]
    for diff, text, correct, wrongs in specs:
        cur = conn.execute(
            """INSERT INTO questions (topic_id, text, option_a, option_b, option_c, option_d, correct_option,
                                      explanation, difficulty, marks, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (topic_id, text, correct, wrongs[0], wrongs[1], wrongs[2], "A", f"The answer is {correct}.",
             diff, {"Easy": 1, "Medium": 2, "Hard": 3}[diff], now()))
        question_ids.append(cur.lastrowid)

    quiz_id = conn.execute(
        "INSERT INTO quizzes (title, description, subject_id, topic_id, difficulty, is_published, created_at) "
        "VALUES ('Test Quiz','d',?,?,'Mixed',1,?)", (subject_id, topic_id, now())).lastrowid
    for pos, qid in enumerate(question_ids):
        conn.execute("INSERT INTO quiz_questions (quiz_id, question_id, position) VALUES (?,?,?)",
                    (quiz_id, qid, pos))

    assessment_id = conn.execute(
        "INSERT INTO assessments (title, subject_id, instructions, difficulty, time_limit_min, is_published, created_at) "
        "VALUES ('Test Assessment',?,'Timed test','Mixed',10,1,?)", (subject_id, now())).lastrowid
    for pos, qid in enumerate(question_ids):
        conn.execute("INSERT INTO assessment_questions (assessment_id, question_id, position) VALUES (?,?,?)",
                    (assessment_id, qid, pos))

    conn.execute("INSERT INTO enrollments (user_id, subject_id, enrolled_at) VALUES (?,?,?)",
                (student_id, subject_id, now()))
    conn.commit()
    conn.close()

    app = create_app(TestConfig)
    ids = {"admin_id": admin_id, "student_id": student_id, "subject_id": subject_id, "topic_id": topic_id,
          "question_ids": question_ids, "quiz_id": quiz_id, "assessment_id": assessment_id}
    return app, path, ids


def cleanup(path):
    try:
        os.remove(path)
    except OSError:
        pass


_SEEDED_CACHE = {}


def make_seeded_app(n_students=150):
    """Full seeded database (real curriculum, generated students, trained model) in a temp dir.
    Built once per test run and shared read-mostly; tests that mutate should use fresh_seeded_copy()."""
    if n_students in _SEEDED_CACHE:
        return _SEEDED_CACHE[n_students]
    import shutil
    from scripts import seed_data
    tmp = tempfile.mkdtemp(prefix="edu_seeded_")
    paths = dict(db=os.path.join(tmp, "eduanalytics.db"), model=os.path.join(tmp, "models", "performance_model.pkl"),
                 metrics=os.path.join(tmp, "models", "model_metrics.json"),
                 data=os.path.join(tmp, "data", "student_performance.csv"))
    seed_data.build(n_students, paths["db"], paths["model"], paths["metrics"], paths["data"],
                    os.path.join(tmp, "data", "seeded_students.csv"), log=lambda *_: None)
    _SEEDED_CACHE[n_students] = (tmp, paths)
    return _SEEDED_CACHE[n_students]


def fresh_seeded_copy(n_students=150):
    """Independent copy of the seeded environment so a test may mutate DB / model files freely."""
    import shutil
    tmp0, _ = make_seeded_app(n_students)
    tmp = tempfile.mkdtemp(prefix="edu_seeded_copy_")
    shutil.copytree(tmp0, tmp, dirs_exist_ok=True)
    paths = dict(db=os.path.join(tmp, "eduanalytics.db"), model=os.path.join(tmp, "models", "performance_model.pkl"),
                 metrics=os.path.join(tmp, "models", "model_metrics.json"),
                 data=os.path.join(tmp, "data", "student_performance.csv"))

    class SeededConfig(TestConfig):
        DATABASE = paths["db"]
        MODEL_PATH = paths["model"]
        METRICS_PATH = paths["metrics"]
        DATASET_PATH = paths["data"]
    return create_app(SeededConfig), tmp, paths
