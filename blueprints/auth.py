import re
import sqlite3

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from db import get_db, now

auth_bp = Blueprint("auth", __name__)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _next_student_code(db):
    """STU<year><4-digit sequence>, guaranteed unused (the column is also UNIQUE in the schema)."""
    n = db.execute("SELECT COUNT(*) c FROM users WHERE role='student'").fetchone()["c"] + 1
    while True:
        code = f"STU{now()[:4]}{n:04d}"
        if not db.execute("SELECT 1 FROM users WHERE student_code = ?", (code,)).fetchone():
            return code
        n += 1


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("main.index"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if user and user["is_active"] and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["role"] = user["role"]
            session["name"] = user["name"]
            session.permanent = True
            db.execute("UPDATE users SET last_login = ? WHERE id = ?", (now(), user["id"]))
            db.commit()
            flash(f"Welcome back, {user['name'].split()[0]}!", "success")
            return redirect(url_for("admin.dashboard") if user["role"] == "admin" else url_for("student.dashboard"))
        flash("Incorrect email or password.", "error")
    return render_template("auth/login.html")


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("main.index"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        errors = []
        if len(name) < 2:
            errors.append("Please enter your full name.")
        if not EMAIL_RE.match(email):
            errors.append("Please enter a valid email address.")
        if len(password) < 6:
            errors.append("Password must be at least 6 characters.")
        if password != confirm:
            errors.append("Passwords do not match.")

        db = get_db()
        if not errors and db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone():
            errors.append("An account with that email already exists.")

        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("auth/register.html", name=name, email=email)

        # Real, persistent record: unique student code + hashed password. Attendance / study hours / GPA are
        # left empty (NULL) - they are not invented; the student fills them in on the profile page.
        pw_hash = generate_password_hash(password)
        cur = None
        for _ in range(5):
            try:
                cur = db.execute(
                    """INSERT INTO users (student_code, name, email, password_hash, role, attendance_percentage,
                                          study_hours_per_week, previous_gpa, created_at)
                       VALUES (?,?,?,?, 'student', NULL, NULL, NULL, ?)""",
                    (_next_student_code(db), name, email, pw_hash, now()))
                break
            except sqlite3.IntegrityError:
                db.rollback()
                if db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone():
                    break                      # someone registered this email in the meantime
        if cur is None:
            flash("An account with that email already exists.", "error")
            return render_template("auth/register.html", name=name, email=email)
        uid = cur.lastrowid
        for row in db.execute("SELECT id FROM subjects").fetchall():
            db.execute("INSERT OR IGNORE INTO enrollments (user_id, subject_id, enrolled_at) VALUES (?,?,?)",
                      (uid, row["id"], now()))
        db.execute(
            """INSERT INTO notifications (user_id, kind, title, message, url, is_read, created_at)
               VALUES (?, 'welcome', 'Welcome to EduAnalytics!',
                       'Your account is ready. Start with a subject that interests you, or jump into practice.',
                       '/student/dashboard', 0, ?)""", (uid, now()))
        db.commit()
        session.clear()
        session["user_id"] = uid
        session["role"] = "student"
        session["name"] = name
        session.permanent = True
        flash("Account created — welcome to EduAnalytics!", "success")
        return redirect(url_for("student.dashboard"))
    return render_template("auth/register.html")


@auth_bp.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))
