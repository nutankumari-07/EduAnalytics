"""SQLite access layer (plain sqlite3, foreign keys enforced)."""
import json
import sqlite3
from datetime import datetime

from flask import current_app, g

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_code TEXT UNIQUE,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('student','admin')),
    is_active INTEGER NOT NULL DEFAULT 1,
    branch TEXT DEFAULT 'Computer Science & Engineering',
    semester INTEGER DEFAULT 5,
    phone TEXT,
    attendance_percentage REAL DEFAULT 80,
    study_hours_per_week REAL DEFAULT 10,
    previous_gpa REAL DEFAULT 7.0,
    created_at TEXT NOT NULL,
    last_login TEXT
);

CREATE TABLE IF NOT EXISTS subjects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    category TEXT,
    description TEXT,
    image_path TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    difficulty TEXT NOT NULL DEFAULT 'Intermediate'
        CHECK (difficulty IN ('Beginner','Intermediate','Advanced')),
    position INTEGER NOT NULL DEFAULT 0,
    concepts TEXT NOT NULL DEFAULT '[]',
    UNIQUE (subject_id, name)
);

CREATE TABLE IF NOT EXISTS materials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT,
    type TEXT NOT NULL CHECK (type IN ('notes','slides','video','pdf','article','practice')),
    url TEXT,
    content TEXT,
    est_minutes INTEGER NOT NULL DEFAULT 10,
    is_recommended INTEGER NOT NULL DEFAULT 0,
    position INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    option_a TEXT NOT NULL,
    option_b TEXT NOT NULL,
    option_c TEXT NOT NULL,
    option_d TEXT NOT NULL,
    correct_option TEXT NOT NULL CHECK (correct_option IN ('A','B','C','D')),
    explanation TEXT NOT NULL,
    difficulty TEXT NOT NULL CHECK (difficulty IN ('Easy','Medium','Hard')),
    marks INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quizzes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    topic_id INTEGER REFERENCES topics(id) ON DELETE CASCADE,
    difficulty TEXT NOT NULL DEFAULT 'Mixed',
    is_published INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS quiz_questions (
    quiz_id INTEGER NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
    question_id INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    position INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (quiz_id, question_id)
);

CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    instructions TEXT,
    difficulty TEXT NOT NULL DEFAULT 'Mixed',
    time_limit_min INTEGER NOT NULL DEFAULT 30,
    is_published INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assessment_questions (
    assessment_id INTEGER NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
    question_id INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    position INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (assessment_id, question_id)
);

CREATE TABLE IF NOT EXISTS enrollments (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    enrolled_at TEXT NOT NULL,
    PRIMARY KEY (user_id, subject_id)
);

CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('practice','quiz','assessment')),
    quiz_id INTEGER REFERENCES quizzes(id) ON DELETE SET NULL,
    assessment_id INTEGER REFERENCES assessments(id) ON DELETE SET NULL,
    subject_id INTEGER REFERENCES subjects(id) ON DELETE SET NULL,
    topic_id INTEGER REFERENCES topics(id) ON DELETE SET NULL,
    title TEXT,
    difficulty TEXT,
    started_at TEXT NOT NULL,
    submitted_at TEXT,
    status TEXT NOT NULL DEFAULT 'in_progress' CHECK (status IN ('in_progress','submitted')),
    total_marks REAL NOT NULL DEFAULT 0,
    obtained_marks REAL NOT NULL DEFAULT 0,
    percentage REAL NOT NULL DEFAULT 0,
    accuracy REAL NOT NULL DEFAULT 0,
    duration_sec INTEGER NOT NULL DEFAULT 0,
    time_limit_sec INTEGER
);
CREATE INDEX IF NOT EXISTS idx_attempts_user ON attempts(user_id, kind, status);

CREATE TABLE IF NOT EXISTS attempt_answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id INTEGER NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
    question_id INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    position INTEGER NOT NULL DEFAULT 0,
    selected_option TEXT CHECK (selected_option IN ('A','B','C','D')),
    correct_option TEXT NOT NULL,
    is_correct INTEGER,
    marks_awarded REAL NOT NULL DEFAULT 0,
    answered_at TEXT,
    UNIQUE (attempt_id, question_id)
);
CREATE INDEX IF NOT EXISTS idx_answers_attempt ON attempt_answers(attempt_id);
CREATE INDEX IF NOT EXISTS idx_answers_question ON attempt_answers(question_id);

CREATE TABLE IF NOT EXISTS material_progress (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    material_id INTEGER NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
    completed INTEGER NOT NULL DEFAULT 0,
    first_accessed TEXT,
    last_accessed TEXT,
    completed_at TEXT,
    PRIMARY KEY (user_id, material_id)
);

CREATE TABLE IF NOT EXISTS student_progress (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    materials_done INTEGER NOT NULL DEFAULT 0,
    materials_total INTEGER NOT NULL DEFAULT 0,
    questions_attempted INTEGER NOT NULL DEFAULT 0,
    questions_total INTEGER NOT NULL DEFAULT 0,
    answers_graded INTEGER NOT NULL DEFAULT 0,
    answers_correct INTEGER NOT NULL DEFAULT 0,
    score REAL,
    progress_pct REAL NOT NULL DEFAULT 0,
    completed INTEGER NOT NULL DEFAULT 0,
    last_activity TEXT,
    PRIMARY KEY (user_id, topic_id)
);

CREATE TABLE IF NOT EXISTS learning_gaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    severity TEXT NOT NULL CHECK (severity IN ('High','Medium','Low')),
    score REAL NOT NULL,
    attempts_count INTEGER NOT NULL DEFAULT 0,
    observed_issue TEXT,
    suggested_actions TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','resolved')),
    detected_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (user_id, topic_id)
);

CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    topic_id INTEGER REFERENCES topics(id) ON DELETE CASCADE,
    material_id INTEGER REFERENCES materials(id) ON DELETE SET NULL,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    reason TEXT NOT NULL,
    action_label TEXT NOT NULL,
    action_url TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 50,
    source TEXT NOT NULL DEFAULT 'system' CHECK (source IN ('system','admin')),
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_recs_user ON recommendations(user_id, is_active);

CREATE TABLE IF NOT EXISTS student_outcomes (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    final_score REAL NOT NULL,
    performance_level TEXT NOT NULL CHECK (performance_level IN ('Low','Medium','High')),
    recorded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    url TEXT,
    is_read INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notif_user ON notifications(user_id, is_read);
"""


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def connect(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def get_db():
    if "db" not in g:
        g.db = connect(current_app.config["DATABASE"])
    return g.db


def close_db(exc=None):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def init_app_teardown(app):
    app.teardown_appcontext(close_db)


def init_schema(conn):
    conn.executescript(SCHEMA)
    conn.commit()


def one(db, sql, args=()):
    return db.execute(sql, args).fetchone()


def all_(db, sql, args=()):
    return db.execute(sql, args).fetchall()


def scalar(db, sql, args=(), default=0):
    row = db.execute(sql, args).fetchone()
    return default if row is None or row[0] is None else row[0]


def jload(text, default=None):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return [] if default is None else default
