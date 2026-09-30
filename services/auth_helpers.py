"""Session-based auth helpers: login/role decorators + current-user loader.

Every student-only route is isolated to session['user_id']; every cross-role access
(a student hitting /admin/*, or vice versa) returns a real 403, not a silent redirect.
"""
from functools import wraps

from flask import abort, current_app, flash, g, redirect, session, url_for

from db import get_db


def current_user():
    if "_user" not in g:
        uid = session.get("user_id")
        g._user = None
        if uid:
            db = get_db()
            g._user = db.execute("SELECT * FROM users WHERE id = ? AND is_active = 1", (uid,)).fetchone()
    return g._user


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to continue.", "warning")
            return redirect(url_for("auth.login"))
        if current_user() is None:
            session.clear()
            flash("Your session has expired. Please log in again.", "warning")
            return redirect(url_for("auth.login"))
        return view(*args, **kwargs)
    return wrapped


def role_required(role):
    def decorator(view):
        @wraps(view)
        @login_required
        def wrapped(*args, **kwargs):
            if session.get("role") != role:
                abort(403)
            return view(*args, **kwargs)
        return wrapped
    return decorator
