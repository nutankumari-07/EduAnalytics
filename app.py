"""EduAnalytics Flask application factory."""
import os
import re

from flask import Flask, render_template, session
from markupsafe import Markup, escape

from config import Config
import db as db_module


def notes_format(text):
    """Minimal, safe formatter for the **bold** study-note paragraphs stored in materials.content."""
    if not text:
        return ""
    out = []
    for para in text.split("\n\n"):
        safe = str(escape(para))
        safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
        out.append(f"<p>{safe}</p>")
    return Markup("".join(out))


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db_module.init_app_teardown(app)
    app.jinja_env.filters["notes_format"] = notes_format

    from blueprints.auth import auth_bp
    from blueprints.student import student_bp
    from blueprints.admin import admin_bp
    from blueprints.main import main_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(student_bp, url_prefix="/student")
    app.register_blueprint(admin_bp, url_prefix="/admin")

    @app.context_processor
    def inject_globals():
        return {"current_role": session.get("role"), "current_name": session.get("name")}

    @app.errorhandler(404)
    def not_found(e):
        return render_template("errors/404.html"), 404

    @app.errorhandler(403)
    def forbidden(e):
        return render_template("errors/403.html"), 403

    @app.errorhandler(500)
    def server_error(e):
        return render_template("errors/500.html"), 500

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
