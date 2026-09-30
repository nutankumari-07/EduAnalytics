"""Integration tests against a fully seeded database (real curriculum, generated students, trained model).

Run:  python -m unittest discover -s tests -v
"""
import csv
import json
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from werkzeug.security import check_password_hash

from db import connect
from ml.train_model import TIE_TOLERANCE, select_best
from services import features as features_svc
from tests.fixtures import fresh_seeded_copy, make_seeded_app

N = 150


def _load(path):
    with open(path) as fh:
        return json.load(fh)


MANDATORY = {"Logistic Regression", "Decision Tree", "Random Forest", "KNN", "SVM", "Gradient Boosting", "Naive Bayes"}


class SeededBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app, cls.tmp, cls.paths = fresh_seeded_copy(N)
        with open(os.path.join(cls.tmp, "data", "seeded_students.csv")) as fh:
            cls.creds = list(csv.DictReader(fh))
        cls.metrics = _load(cls.paths["metrics"])

    def setUp(self):
        self.client = self.app.test_client()

    def db(self):
        return connect(self.paths["db"])

    def login(self, email, password, client=None):
        return (client or self.client).post("/login", data={"email": email, "password": password})

    def admin(self):
        self.assertEqual(self.login("admin@eduanalytics.com", "admin123").status_code, 302)


class SeedDataTests(SeededBase):
    def test_student_count_and_unique_credentials(self):
        db = self.db()
        n = db.execute("SELECT COUNT(*) c FROM users WHERE role='student'").fetchone()["c"]
        self.assertEqual(n, N + 1)
        self.assertEqual(db.execute("SELECT COUNT(DISTINCT student_code) c FROM users WHERE role='student'").fetchone()["c"], n)
        self.assertEqual(db.execute("SELECT COUNT(DISTINCT email) c FROM users WHERE role='student'").fetchone()["c"], n)
        self.assertEqual(len(self.creds), n)
        self.assertEqual(db.execute("SELECT COUNT(*) c FROM users WHERE password_hash IS NULL OR password_hash='' "
                                    "OR password_hash = ?", ("student123",)).fetchone()["c"], 0)

    def test_every_student_has_varied_activity_and_outcome(self):
        db = self.db()
        for kind in ("quiz", "practice", "assessment"):
            missing = db.execute("SELECT COUNT(*) c FROM users u WHERE role='student' AND NOT EXISTS "
                                 "(SELECT 1 FROM attempts a WHERE a.user_id=u.id AND a.kind=? AND a.status='submitted')",
                                 (kind,)).fetchone()["c"]
            self.assertEqual(missing, 0, kind)
        self.assertEqual(db.execute("SELECT COUNT(*) c FROM student_outcomes").fetchone()["c"], N + 1)
        for col in ("attendance_percentage", "study_hours_per_week", "previous_gpa"):
            self.assertGreater(db.execute(f"SELECT COUNT(DISTINCT {col}) c FROM users WHERE role='student'").fetchone()["c"], 50, col)
        self.assertGreater(db.execute("SELECT COUNT(DISTINCT ROUND(a,0)) c FROM (SELECT AVG(percentage) a FROM attempts "
                                      "WHERE kind='quiz' GROUP BY user_id)").fetchone()["c"], 30)

    def test_seeded_attempt_scores_match_real_grader(self):
        """Bulk-seeded attempts must be consistent with services.scoring.grade_attempt."""
        from services.scoring import grade_attempt
        db = self.db()
        for row in db.execute("SELECT id, percentage, accuracy, obtained_marks FROM attempts ORDER BY RANDOM() LIMIT 40").fetchall():
            r = grade_attempt(db, row["id"])
            self.assertAlmostEqual(r["percentage"], row["percentage"], places=2)
            self.assertAlmostEqual(r["accuracy"], row["accuracy"], places=2)
            self.assertAlmostEqual(r["obtained_marks"], row["obtained_marks"], places=2)

    def test_topic_performance_and_gaps_populated(self):
        db = self.db()
        self.assertGreater(db.execute("SELECT COUNT(*) c FROM student_progress WHERE score IS NOT NULL").fetchone()["c"], N * 10)
        self.assertGreater(db.execute("SELECT COUNT(*) c FROM learning_gaps WHERE status='active'").fetchone()["c"], N)
        self.assertGreater(db.execute("SELECT COUNT(*) c FROM recommendations WHERE is_active=1").fetchone()["c"], N)


class LoginAndIsolationTests(SeededBase):
    def test_admin_login(self):
        self.admin()
        self.assertEqual(self.client.get("/admin/dashboard").status_code, 200)

    def test_multiple_seeded_students_can_log_in_and_see_their_own_name(self):
        for row in self.creds[:: max(1, len(self.creds) // 8)][:8] + [self.creds[-1]]:
            c = self.app.test_client()
            self.assertEqual(self.login(row["email"], row["password"], c).status_code, 302, row["email"])
            html = c.get("/student/dashboard", follow_redirects=True).get_data(as_text=True)
            self.assertIn(row["name"].split()[0], html)
            self.assertEqual(c.get("/admin/dashboard").status_code, 403)

    def test_wrong_password_rejected(self):
        self.assertEqual(self.login(self.creds[3]["email"], "nope").status_code, 200)

    def test_student_cannot_read_another_students_attempt_or_history(self):
        db = self.db()
        a, b = self.creds[10], self.creds[11]
        ida = db.execute("SELECT id FROM users WHERE email=?", (a["email"],)).fetchone()["id"]
        idb = db.execute("SELECT id FROM users WHERE email=?", (b["email"],)).fetchone()["id"]
        b_attempt = db.execute("SELECT id FROM attempts WHERE user_id=? LIMIT 1", (idb,)).fetchone()["id"]
        a_attempt_ids = {r["id"] for r in db.execute("SELECT id FROM attempts WHERE user_id=?", (ida,)).fetchall()}
        self.login(a["email"], a["password"])
        self.assertEqual(self.client.get(f"/student/results/{b_attempt}").status_code, 404)
        self.assertEqual(self.client.get(f"/student/attempt/{b_attempt}").status_code, 404)
        hist = self.client.get("/student/history").get_data(as_text=True)
        for link_id in set(int(x) for x in re.findall(r"/student/results/(\d+)", hist)):
            self.assertIn(link_id, a_attempt_ids)
        self.assertNotIn(b["email"], hist)
        self.assertNotIn(b["name"], self.client.get("/student/dashboard").get_data(as_text=True))

    def test_student_pages_show_no_technical_ml_details(self):
        row = self.creds[5]
        self.login(row["email"], row["password"])
        name = self.metrics["model_name"]
        for url in ("/student/dashboard", "/student/performance", "/student/recommendations", "/student/progress"):
            html = self.client.get(url).get_data(as_text=True)
            self.assertNotIn(name, html, url)
            for word in ("Confusion", "Weighted F1", "F1 score", "cross-validation"):
                self.assertNotIn(word, html, url)
        self.assertEqual(self.client.get("/admin/ml-analytics").status_code, 403)
        self.assertEqual(self.client.post("/admin/ml-analytics/retrain").status_code, 403)


class StudentFeatureTests(SeededBase):
    def test_pages_load_with_seeded_data(self):
        row = self.creds[20]
        self.login(row["email"], row["password"])
        db = self.db()
        uid = db.execute("SELECT id FROM users WHERE email=?", (row["email"],)).fetchone()["id"]
        for url in ("/student/dashboard", "/student/subjects", "/student/learning-gaps", "/student/recommendations",
                    "/student/progress", "/student/history", "/student/performance", "/student/profile"):
            self.assertEqual(self.client.get(url).status_code, 200, url)
        sid = db.execute("SELECT subject_id FROM enrollments WHERE user_id=? LIMIT 1", (uid,)).fetchone()["subject_id"]
        self.assertEqual(self.client.get(f"/student/subjects/{sid}").status_code, 200)
        tid = db.execute("SELECT id FROM topics WHERE subject_id=? LIMIT 1", (sid,)).fetchone()["id"]
        self.assertEqual(self.client.get(f"/student/topics/{tid}").status_code, 200)
        mid = db.execute("SELECT id FROM materials WHERE topic_id=? AND type='notes'", (tid,)).fetchone()["id"]
        self.assertEqual(self.client.get(f"/student/materials/{mid}/view").status_code, 200)
        html = self.client.get("/student/dashboard").get_data(as_text=True)
        self.assertRegex(html, r"Low|Medium|High")           # seeded students have a prediction

    def test_practice_quiz_assessment_flow_only_affects_own_data(self):
        row = self.creds[30]
        other = self.creds[31]
        self.login(row["email"], row["password"])
        db = self.db()
        uid = db.execute("SELECT id FROM users WHERE email=?", (row["email"],)).fetchone()["id"]
        oid = db.execute("SELECT id FROM users WHERE email=?", (other["email"],)).fetchone()["id"]
        before_other = db.execute("SELECT COUNT(*) c FROM attempts WHERE user_id=?", (oid,)).fetchone()["c"]
        before = db.execute("SELECT COUNT(*) c FROM attempts WHERE user_id=?", (uid,)).fetchone()["c"]
        topic = db.execute("SELECT id, subject_id FROM topics WHERE subject_id IN (SELECT subject_id FROM enrollments WHERE user_id=?) LIMIT 1", (uid,)).fetchone()
        quiz = db.execute("SELECT id FROM quizzes WHERE topic_id=?", (topic["id"],)).fetchone()["id"]
        asm = db.execute("SELECT id FROM assessments WHERE subject_id=?", (topic["subject_id"],)).fetchone()["id"]
        for url in (f"/student/practice/{topic['id']}/start", f"/student/quiz/{quiz}/start", f"/student/assessment/{asm}/start"):
            r = self.client.post(url)
            self.assertEqual(r.status_code, 302, url)
            page = self.client.get(r.headers["Location"]).get_data(as_text=True)
            qids = sorted(set(int(x) for x in re.findall(r'name="q_(\d+)"', page)))
            self.assertTrue(qids)
            form = {f"q_{q}": "A" for q in qids}
            r2 = self.client.post(r.headers["Location"], data=form)
            self.assertEqual(r2.status_code, 302)
            self.assertEqual(self.client.get(r2.headers["Location"]).status_code, 200)
        db = self.db()
        self.assertEqual(db.execute("SELECT COUNT(*) c FROM attempts WHERE user_id=?", (uid,)).fetchone()["c"], before + 3)
        self.assertEqual(db.execute("SELECT COUNT(*) c FROM attempts WHERE user_id=?", (oid,)).fetchone()["c"], before_other)

    def test_gaps_are_threshold_based_and_recommendations_use_gaps_and_level(self):
        db = self.db()
        gaps = db.execute("SELECT severity, score FROM learning_gaps WHERE status='active'").fetchall()
        for g in gaps:                       # every active gap must be below the highest gap threshold (82%)
            self.assertLess(g["score"], 82)
            if g["severity"] == "High":
                self.assertLess(g["score"], 60)
        kinds = {r["kind"] for r in db.execute("SELECT DISTINCT kind FROM recommendations WHERE is_active=1").fetchall()}
        self.assertIn("weak_topic", kinds)
        self.assertTrue(kinds & {"foundation", "challenge"}, kinds)
        rec = db.execute("SELECT reason FROM recommendations WHERE kind='weak_topic' AND reason LIKE '%performance level%' LIMIT 1").fetchone()
        self.assertIsNotNone(rec)


class AdminAnalyticsTests(SeededBase):
    def test_admin_dashboard_and_management_pages(self):
        self.admin()
        db = self.db()
        sid = db.execute("SELECT id FROM subjects ORDER BY id LIMIT 1").fetchone()["id"]
        tid = db.execute("SELECT id FROM topics WHERE subject_id=? ORDER BY id LIMIT 1", (sid,)).fetchone()["id"]
        for url in ("/admin/dashboard", "/admin/students", "/admin/subjects", f"/admin/subjects/{sid}/topics",
                    f"/admin/topics/{tid}/materials", f"/admin/topics/{tid}/edit", "/admin/questions", "/admin/quizzes",
                    "/admin/assessments", "/admin/learning-gaps", "/admin/recommendations", "/admin/accounts",
                    "/admin/analytics", "/admin/ml-analytics"):
            self.assertEqual(self.client.get(url).status_code, 200, url)

    def test_admin_can_search_and_view_individual_student(self):
        self.admin()
        row = self.creds[42]
        html = self.client.get("/admin/students?q=" + row["student_code"]).get_data(as_text=True)
        self.assertIn(row["email"], html)
        uid = self.db().execute("SELECT id FROM users WHERE email=?", (row["email"],)).fetchone()["id"]
        detail = self.client.get(f"/admin/students/{uid}")
        self.assertEqual(detail.status_code, 200)
        self.assertIn(row["name"], detail.get_data(as_text=True))

    def test_analytics_page_is_database_driven_and_populated(self):
        self.admin()
        html = self.client.get("/admin/analytics").get_data(as_text=True)
        db = self.db()
        self.assertNotIn("<canvas", html)                       # no client-side chart dependency
        self.assertNotIn("No submitted attempts yet", html)
        self.assertNotIn("Not enough attempt history yet", html)
        self.assertIn("<polyline", html)                        # score trend drawn
        # subject averages come from the DB
        for r in db.execute("SELECT s.name, ROUND(AVG(a.percentage),1) v FROM attempts a JOIN subjects s ON s.id=a.subject_id "
                            "WHERE a.status='submitted' GROUP BY s.id").fetchall():
            self.assertIn(r["name"].replace("&", "&amp;"), html)
            self.assertIn(f"{r['v']}%", html)
        self.assertEqual(html.count("imp-bar-row"), 9 + db.execute("SELECT COUNT(DISTINCT kind) c FROM attempts").fetchone()["c"])
        # donut total equals number of students with a prediction
        frame = features_svc.feature_frame(db)
        m = re.search(r'<text[^>]*font-size="26"[^>]*>(\d+)</text>', html)
        self.assertEqual(int(m.group(1)), len(frame))
        self.assertGreater(len(frame), 0)

    def test_analytics_charts_react_to_database_changes(self):
        """Proves nothing is hard-coded: wipe attempts -> empty-state text appears."""
        app, tmp, paths = fresh_seeded_copy(N)
        c = app.test_client()
        c.post("/login", data={"email": "admin@eduanalytics.com", "password": "admin123"})
        conn = connect(paths["db"])
        conn.execute("DELETE FROM attempts")
        conn.commit()
        html = c.get("/admin/analytics").get_data(as_text=True)
        self.assertIn("No submitted attempts yet", html)
        self.assertNotIn("<polyline", html)


class MLTests(SeededBase):
    def test_all_required_models_compared_with_same_setup(self):
        names = {m["name"] for m in self.metrics["model_comparison"]}
        self.assertTrue(MANDATORY <= names, MANDATORY - names)
        self.assertIn("Extra Trees", names)
        skipped = {s["name"] for s in self.metrics["models_skipped"]}
        try:
            import xgboost  # noqa: F401
            self.assertIn("XGBoost", names)
        except ImportError:
            self.assertIn("XGBoost", skipped)
        self.assertEqual(self.metrics["features"], features_svc.FEATURES)
        self.assertEqual(self.metrics["dataset_size"], self.metrics["train_size"] + self.metrics["test_size"])

    def test_selection_follows_the_stated_rule_and_is_not_hardcoded(self):
        rows = [{"name": m["name"], "f1": m["f1"], "cv_f1_mean": m["cv_f1_mean"], "cv_f1_std": m["cv_f1_std"]}
                for m in self.metrics["model_comparison"]]
        best = max(r["f1"] for r in rows)
        tied = sorted([r for r in rows if best - r["f1"] <= TIE_TOLERANCE + 1e-3],
                      key=lambda r: (-(r["cv_f1_mean"] - r["cv_f1_std"]), -r["f1"]))
        self.assertIn(self.metrics["model_name"], [t["name"] for t in tied])
        self.assertGreaterEqual(self.metrics["f1"], best - TIE_TOLERANCE - 1e-3)
        # synthetic check of the selector itself: a clearly better model wins whatever its name
        fake = [dict(name=n, f1=0.7, cv_f1_mean=0.7, cv_f1_std=0.03, ) for n in ("A", "B", "C")]
        fake[2]["f1"] = 0.9
        self.assertEqual(select_best(fake)[0]["name"], "C")
        tie = [dict(name="X", f1=0.80, cv_f1_mean=0.78, cv_f1_std=0.06), dict(name="Y", f1=0.798, cv_f1_mean=0.79, cv_f1_std=0.02)]
        self.assertEqual(select_best(tie)[0]["name"], "Y")       # near-tie -> best conservative CV score wins
        # low variance alone must not win when the CV mean is much worse
        worse = [dict(name="Steady-but-worse", f1=0.770, cv_f1_mean=0.70, cv_f1_std=0.010),
                 dict(name="Better-cv", f1=0.769, cv_f1_mean=0.78, cv_f1_std=0.020)]
        self.assertEqual(select_best(worse)[0]["name"], "Better-cv")

    def test_saved_model_matches_metrics_and_dataset(self):
        import joblib
        bundle = joblib.load(self.paths["model"])
        self.assertEqual(bundle["model_name"], self.metrics["model_name"])
        self.assertEqual(bundle["features"], self.metrics["features"])
        with open(self.paths["data"]) as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), self.metrics["dataset_size"])
        self.assertEqual(self.metrics["trained_at"], bundle["trained_at"])

    def test_ml_analytics_page_shows_current_run_and_is_admin_only(self):
        self.assertEqual(self.client.get("/admin/ml-analytics").status_code, 302)   # anonymous -> login
        self.admin()
        html = self.client.get("/admin/ml-analytics").get_data(as_text=True)
        self.assertIn(f"Selected model: {self.metrics['model_name']}", html)
        self.assertIn(self.metrics["trained_at"], html)
        self.assertIn(self.metrics["selection_reason"].replace("'", "&#39;"), html)
        self.assertIn("Highest weighted F1-score", html)
        for m in self.metrics["model_comparison"]:
            self.assertIn(m["name"], html)
        for word in ("Confusion matrix", "Per-class metrics", "Feature importance", "Model comparison"):
            self.assertIn(word, html)

    def test_retraining_from_admin_uses_current_database_and_rewrites_metrics(self):
        app, tmp, paths = fresh_seeded_copy(N)
        c = app.test_client()
        c.post("/login", data={"email": "admin@eduanalytics.com", "password": "admin123"})
        old = _load(paths["metrics"])
        conn = connect(paths["db"])                    # add one more fully-active student outcome? simply drop a student
        victim = conn.execute("SELECT id FROM users WHERE role='student' ORDER BY id DESC LIMIT 1").fetchone()["id"]
        conn.execute("DELETE FROM users WHERE id=?", (victim,))
        conn.commit()
        import time
        time.sleep(1.1)
        r = c.post("/admin/ml-analytics/retrain", follow_redirects=True)
        self.assertEqual(r.status_code, 200)
        new = _load(paths["metrics"])
        self.assertEqual(new["dataset_size"], old["dataset_size"] - 1)
        self.assertNotEqual(new["trained_at"], old["trained_at"])
        self.assertIn(new["model_name"], r.get_data(as_text=True))

    def test_retrain_with_too_little_data_fails_gracefully(self):
        app, tmp, paths = fresh_seeded_copy(N)
        c = app.test_client()
        c.post("/login", data={"email": "admin@eduanalytics.com", "password": "admin123"})
        conn = connect(paths["db"])
        conn.execute("DELETE FROM student_outcomes")
        conn.commit()
        r = c.post("/admin/ml-analytics/retrain", follow_redirects=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("Retraining failed", r.get_data(as_text=True))


class RegistrationTests(SeededBase):
    def test_new_student_is_persisted_hashed_unique_and_gets_no_fake_prediction(self):
        app, tmp, paths = fresh_seeded_copy(N)
        c = app.test_client()
        codes = []
        for i in (1, 2):
            cc = app.test_client()
            r = cc.post("/register", data={"name": f"Fresh Student{i}", "email": f"fresh{i}@example.com",
                                           "password": "secret123", "confirm_password": "secret123"})
            self.assertEqual(r.status_code, 302)
        conn = connect(paths["db"])                      # brand-new connection: proves it is really persisted
        for i in (1, 2):
            u = conn.execute("SELECT * FROM users WHERE email=?", (f"fresh{i}@example.com",)).fetchone()
            self.assertIsNotNone(u)
            self.assertNotEqual(u["password_hash"], "secret123")
            self.assertTrue(check_password_hash(u["password_hash"], "secret123"))
            self.assertTrue(u["student_code"])
            self.assertIsNone(u["attendance_percentage"])
            codes.append(u["student_code"])
            feats, status = features_svc.features_for_user(conn, u["id"])
            self.assertIsNone(feats)
            self.assertFalse(status["sufficient"])
            self.assertEqual(conn.execute("SELECT COUNT(*) c FROM student_outcomes WHERE user_id=?", (u["id"],)).fetchone()["c"], 0)
        self.assertNotEqual(codes[0], codes[1])
        all_codes = [r[0] for r in conn.execute("SELECT student_code FROM users WHERE role='student'")]
        self.assertEqual(len(all_codes), len(set(all_codes)))
        # can log in again from a fresh client; dashboard shows no prediction and says what is missing
        c2 = app.test_client()
        self.assertEqual(c2.post("/login", data={"email": "fresh1@example.com", "password": "secret123"}).status_code, 302)
        html = c2.get("/student/dashboard").get_data(as_text=True)
        self.assertIn("Needs more activity", html)
        perf = c2.get("/student/performance").get_data(as_text=True)
        self.assertIn("No prediction yet", perf)
        self.assertNotIn("None", perf.split("<h1>")[1][:1500])
        # not part of the training data
        from ml.train_model import build_dataset
        ds = build_dataset(paths["db"])
        self.assertNotIn("fresh1@example.com", conn.execute("SELECT email FROM users WHERE id IN (%s)" % ",".join(str(int(x)) for x in ds["user_id"])).fetchall().__repr__())

    def test_prediction_appears_only_after_profile_and_enough_activity(self):
        app, tmp, paths = fresh_seeded_copy(N)
        c = app.test_client()
        c.post("/register", data={"name": "Active Newbie", "email": "newbie@example.com", "password": "secret123",
                                  "confirm_password": "secret123"})
        conn = connect(paths["db"])
        uid = conn.execute("SELECT id FROM users WHERE email='newbie@example.com'").fetchone()["id"]

        def status():
            return features_svc.activity_status(connect(paths["db"]), uid)

        def run(url):
            r = c.post(url)
            loc = r.headers["Location"]
            page = c.get(loc).get_data(as_text=True)
            qids = sorted(set(int(x) for x in re.findall(r'name="q_(\d+)"', page)))
            c.post(loc, data={f"q_{q}": "A" for q in qids})

        topics = conn.execute("SELECT t.id, t.subject_id FROM topics t ORDER BY t.id LIMIT 4").fetchall()
        for t in topics:                                    # 4 practice sets over 4 topics
            run(f"/student/practice/{t['id']}/start")
        self.assertFalse(status()["sufficient"])
        run(f"/student/quiz/{conn.execute('SELECT id FROM quizzes WHERE topic_id=?', (topics[0]['id'],)).fetchone()['id']}/start")
        run(f"/student/assessment/{conn.execute('SELECT id FROM assessments WHERE subject_id=?', (topics[0]['subject_id'],)).fetchone()['id']}/start")
        st = status()
        self.assertFalse(st["sufficient"])                  # activity is enough, profile is still empty
        self.assertFalse(st["profile_ok"])
        self.assertIsNone(features_svc.features_for_user(connect(paths["db"]), uid)[0])
        c.post("/student/profile", data={"phone": "", "study_hours_per_week": "12", "attendance_percentage": "82", "previous_gpa": "7.4"})
        st = status()
        self.assertTrue(st["sufficient"], st)
        html = c.get("/student/dashboard").get_data(as_text=True)
        self.assertNotIn("Needs more activity", html)
        perf = c.get("/student/performance").get_data(as_text=True)
        self.assertRegex(perf, r'class="num">(Low|Medium|High)<')


if __name__ == "__main__":
    unittest.main()
