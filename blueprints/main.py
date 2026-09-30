from flask import Blueprint, redirect, session, url_for

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    if session.get("user_id"):
        if session.get("role") == "admin":
            return redirect(url_for("admin.dashboard"))
        return redirect(url_for("student.dashboard"))
    return redirect(url_for("auth.login"))
