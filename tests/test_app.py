"""EduAnalytics test suite (unittest — runnable now with the stdlib, and auto-discovered
by pytest too since pytest natively collects unittest.TestCase classes).

Run:  python -m unittest discover -s tests -v      (from the project root)
"""
import json
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.fixtures import cleanup, make_app
from db import connect
from services import gaps as gaps_svc
from services import recommend as recommend_svc
from services.scoring import submit_attempt


class EduAnalyticsTestCase(unittest.TestCase):
    def setUp(self):
        self.app, self.db_path, self.ids = make_app()
        self.client = self.app.test_client()

    def tearDown(self):
        cleanup(self.db_path)

    # -------------------------------------------------- helpers ----
    def login(self, email, password):
        return self.client.post("/login", data={"email": email, "password": password})

    def login_student(self):
        return self.login("student@test.local", "studentpass")

    def login_admin(self):
        return self.login("admin@test.local", "adminpass")

    def start_attempt(self, url):
        r = self.client.post(url)
        self.assertEqual(r.status_code, 302)
        attempt_id = int(r.headers["Location"].rstrip("/").split("/")[-1])
        return attempt_id

    def question_ids_on_page(self, html):
        return sorted(set(int(x) for x in re.findall(r'name="q_(\d+)"', html)), key=int)


# ==================================================================== auth ====

class AuthTests(EduAnalyticsTestCase):
    def test_login_success_redirects_to_role_dashboard(self):
        r = self.login_student()
        self.assertEqual(r.status_code, 302)
        self.assertIn("/student/dashboard", r.headers["Location"])

    def test_login_wrong_password_rejected(self):
        r = self.login("student@test.local", "wrongpassword")
        self.assertEqual(r.status_code, 200)  # re-renders login form
        self.assertIn(b"Incorrect email or password", r.data)

    def test_login_unknown_email_rejected(self):
        r = self.login("nobody@test.local", "whatever")
        self.assertIn(b"Incorrect email or password", r.data)

    def test_register_creates_active_student_enrolled_in_all_subjects(self):
        r = self.client.post("/register", data={
            "name": "New Student", "email": "newstudent@test.local",
            "password": "abcdef", "confirm_password": "abcdef"}, follow_redirects=True)
        self.assertEqual(r.status_code, 200)
        conn = connect(self.db_path)
        user = conn.execute("SELECT * FROM users WHERE email='newstudent@test.local'").fetchone()
        self.assertIsNotNone(user)
        self.assertEqual(user["role"], "student")
        enrollments = conn.execute("SELECT COUNT(*) c FROM enrollments WHERE user_id=?", (user["id"],)).fetchone()["c"]
        subjects = conn.execute("SELECT COUNT(*) c FROM subjects").fetchone()["c"]
        self.assertEqual(enrollments, subjects)
        conn.close()

    def test_register_rejects_mismatched_passwords(self):
        r = self.client.post("/register", data={
            "name": "Bad", "email": "bad@test.local", "password": "abcdef", "confirm_password": "different"})
        self.assertIn(b"Passwords do not match", r.data)

    def test_register_rejects_duplicate_email(self):
        r = self.client.post("/register", data={
            "name": "Dup", "email": "student@test.local", "password": "abcdef", "confirm_password": "abcdef"})
        self.assertIn(b"already exists", r.data)

    def test_logout_clears_session(self):
        self.login_student()
        self.client.get("/logout")
        r = self.client.get("/student/dashboard")
        self.assertEqual(r.status_code, 302)
        self.assertIn("/login", r.headers["Location"])


# ==================================================================== RBAC ====

class RBACTests(EduAnalyticsTestCase):
    def test_unauthenticated_redirected_to_login(self):
        r = self.client.get("/student/dashboard")
        self.assertEqual(r.status_code, 302)
        self.assertIn("/login", r.headers["Location"])

    def test_student_cannot_access_admin_dashboard(self):
        self.login_student()
        r = self.client.get("/admin/dashboard")
        self.assertEqual(r.status_code, 403)

    def test_admin_cannot_access_student_dashboard(self):
        self.login_admin()
        r = self.client.get("/student/dashboard")
        self.assertEqual(r.status_code, 403)

    def test_student_isolated_from_other_students_attempts(self):
        self.login_student()
        attempt_id = self.start_attempt(f"/student/practice/{self.ids['topic_id']}/start")
        self.client.get("/logout")
        # a second student should not be able to view or submit the first student's attempt
        self.client.post("/register", data={
            "name": "Other Student", "email": "other@test.local",
            "password": "abcdef", "confirm_password": "abcdef"})
        r = self.client.get(f"/student/attempt/{attempt_id}")
        self.assertEqual(r.status_code, 404)
        r = self.client.get(f"/student/results/{attempt_id}")
        self.assertEqual(r.status_code, 404)

    def test_admin_only_routes_require_admin(self):
        self.login_student()
        for url in ["/admin/students", "/admin/subjects", "/admin/questions", "/admin/analytics",
                   "/admin/ml-analytics", "/admin/accounts"]:
            r = self.client.get(url)
            self.assertEqual(r.status_code, 403, f"{url} should be admin-only")


# ==================================================================== content ====

class ContentTests(EduAnalyticsTestCase):
    def test_subjects_and_topics_load_with_real_data(self):
        self.login_student()
        r = self.client.get("/student/subjects")
        self.assertIn(b"Test Subject", r.data)
        r = self.client.get(f"/student/topics/{self.ids['topic_id']}")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Test Topic", r.data)
        self.assertIn(b"Test Notes", r.data)

    def test_no_placeholder_or_dummy_question_text(self):
        conn = connect(self.db_path)
        rows = conn.execute("SELECT text FROM questions").fetchall()
        conn.close()
        for r in rows:
            self.assertNotIn("Sample question", r["text"])
            self.assertNotIn("Lorem ipsum", r["text"])

    def test_material_urls_are_never_empty_for_link_types(self):
        conn = connect(self.db_path)
        rows = conn.execute("SELECT type, url FROM materials WHERE type IN ('article','video','pdf')").fetchall()
        conn.close()
        for r in rows:
            self.assertTrue(r["url"], "link-type material must always have a real URL")

    def test_notes_material_renders_without_url(self):
        self.login_student()
        mat = connect(self.db_path).execute("SELECT id FROM materials WHERE type='notes'").fetchone()
        r = self.client.get(f"/student/materials/{mat['id']}/view")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"bold", r.data)  # our **bold** formatter should render <strong>bold</strong>
        self.assertIn(b"<strong>bold</strong>", r.data)


# ==================================================================== scoring ====

class ScoringTests(EduAnalyticsTestCase):
    def test_practice_attempt_all_correct_scores_100(self):
        self.login_student()
        attempt_id = self.start_attempt(f"/student/practice/{self.ids['topic_id']}/start")
        r = self.client.get(f"/student/attempt/{attempt_id}")
        qids = self.question_ids_on_page(r.data.decode())
        form = {f"q_{qid}": "A" for qid in qids}  # option A is always correct in the fixture
        form["elapsed_sec"] = "30"
        r = self.client.post(f"/student/attempt/{attempt_id}", data=form)
        self.assertEqual(r.status_code, 302)
        conn = connect(self.db_path)
        attempt = conn.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone()
        conn.close()
        self.assertEqual(attempt["status"], "submitted")
        self.assertEqual(attempt["percentage"], 100.0)
        self.assertEqual(attempt["accuracy"], 100.0)

    def test_attempt_all_wrong_scores_zero(self):
        self.login_student()
        attempt_id = self.start_attempt(f"/student/practice/{self.ids['topic_id']}/start")
        r = self.client.get(f"/student/attempt/{attempt_id}")
        qids = self.question_ids_on_page(r.data.decode())
        form = {f"q_{qid}": "B" for qid in qids}  # B is always wrong in the fixture
        form["elapsed_sec"] = "30"
        self.client.post(f"/student/attempt/{attempt_id}", data=form)
        conn = connect(self.db_path)
        attempt = conn.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone()
        conn.close()
        self.assertEqual(attempt["percentage"], 0.0)

    def test_quiz_attempt_scoring_and_result_page(self):
        self.login_student()
        attempt_id = self.start_attempt(f"/student/quiz/{self.ids['quiz_id']}/start")
        r = self.client.get(f"/student/attempt/{attempt_id}")
        qids = self.question_ids_on_page(r.data.decode())
        form = {f"q_{qid}": "A" for qid in qids}
        form["elapsed_sec"] = "45"
        self.client.post(f"/student/attempt/{attempt_id}", data=form)
        r = self.client.get(f"/student/results/{attempt_id}")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"100.0%", r.data)

    def test_assessment_is_timed_and_scores_correctly(self):
        self.login_student()
        attempt_id = self.start_attempt(f"/student/assessment/{self.ids['assessment_id']}/start")
        conn = connect(self.db_path)
        attempt = conn.execute("SELECT time_limit_sec FROM attempts WHERE id=?", (attempt_id,)).fetchone()
        conn.close()
        self.assertEqual(attempt["time_limit_sec"], 600)  # 10 min * 60
        r = self.client.get(f"/student/attempt/{attempt_id}")
        qids = self.question_ids_on_page(r.data.decode())
        form = {f"q_{qid}": "A" for qid in qids}
        form["elapsed_sec"] = "300"
        self.client.post(f"/student/attempt/{attempt_id}", data=form)
        conn = connect(self.db_path)
        attempt = conn.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone()
        conn.close()
        self.assertEqual(attempt["percentage"], 100.0)
        self.assertEqual(attempt["kind"], "assessment")

    def test_unanswered_questions_marked_not_correct(self):
        self.login_student()
        attempt_id = self.start_attempt(f"/student/practice/{self.ids['topic_id']}/start")
        conn = connect(self.db_path)
        # simulate a partial submit directly via the scoring service
        qid = self.ids["question_ids"][0]
        submit_attempt(conn, attempt_id, {qid: "A"})
        row = conn.execute("SELECT * FROM attempt_answers WHERE attempt_id=? AND question_id!=?",
                          (attempt_id, qid)).fetchone()
        self.assertIsNone(row["is_correct"])
        conn.close()


# ==================================================================== gaps & recommendations ====

class GapsAndRecommendationsTests(EduAnalyticsTestCase):
    def _fail_everything(self):
        conn = connect(self.db_path)
        for _ in range(3):
            attempt_id = self.start_attempt_direct(conn)
            submit_attempt(conn, attempt_id, {qid: "B" for qid in self.ids["question_ids"]})
        conn.close()

    def start_attempt_direct(self, conn):
        from db import now
        cur = conn.execute(
            "INSERT INTO attempts (user_id, kind, subject_id, topic_id, title, difficulty, started_at, status) "
            "VALUES (?, 'practice', ?, ?, 'Practice', 'Mixed', ?, 'in_progress')",
            (self.ids["student_id"], self.ids["subject_id"], self.ids["topic_id"], now()))
        attempt_id = cur.lastrowid
        for pos, qid in enumerate(self.ids["question_ids"]):
            conn.execute("INSERT INTO attempt_answers (attempt_id, question_id, position, correct_option) "
                        "SELECT ?, id, ?, correct_option FROM questions WHERE id=?", (attempt_id, pos, qid))
        conn.commit()
        return attempt_id

    def test_gap_detected_after_consistent_failure(self):
        self._fail_everything()
        conn = connect(self.db_path)
        found = gaps_svc.detect_all_for_user(conn, self.ids["student_id"])
        self.assertTrue(len(found) >= 1)
        gap = conn.execute("SELECT * FROM learning_gaps WHERE user_id=? AND topic_id=?",
                          (self.ids["student_id"], self.ids["topic_id"])).fetchone()
        self.assertIsNotNone(gap)
        self.assertEqual(gap["severity"], "High")
        self.assertEqual(gap["status"], "active")
        self.assertIn("Test Topic", gap["observed_issue"])
        conn.close()

    def test_no_gap_when_performing_well(self):
        conn = connect(self.db_path)
        attempt_id = self.start_attempt_direct(conn)
        submit_attempt(conn, attempt_id, {qid: "A" for qid in self.ids["question_ids"]})
        found = gaps_svc.detect_all_for_user(conn, self.ids["student_id"])
        self.assertEqual(len(found), 0)
        conn.close()

    def test_recommendation_generated_from_active_gap(self):
        self._fail_everything()
        conn = connect(self.db_path)
        gaps_svc.detect_all_for_user(conn, self.ids["student_id"])
        recommend_svc.generate_for_user(conn, self.ids["student_id"])
        recs = conn.execute("SELECT * FROM recommendations WHERE user_id=? AND is_active=1",
                           (self.ids["student_id"],)).fetchall()
        self.assertTrue(len(recs) >= 1)
        self.assertTrue(any(r["topic_id"] == self.ids["topic_id"] for r in recs))
        for r in recs:
            self.assertTrue(r["action_url"])  # every recommendation must link somewhere real
        conn.close()

    def test_resolved_gap_marked_inactive_after_improvement(self):
        conn = connect(self.db_path)
        # one poor attempt establishes the gap...
        attempt_id = self.start_attempt_direct(conn)
        submit_attempt(conn, attempt_id, {qid: "B" for qid in self.ids["question_ids"]})
        gaps_svc.detect_all_for_user(conn, self.ids["student_id"])
        gap = conn.execute("SELECT * FROM learning_gaps WHERE user_id=? AND topic_id=?",
                          (self.ids["student_id"], self.ids["topic_id"])).fetchone()
        self.assertEqual(gap["status"], "active")
        # ...then sustained, strong performance should outweigh it and resolve the gap
        for _ in range(6):
            attempt_id = self.start_attempt_direct(conn)
            submit_attempt(conn, attempt_id, {qid: "A" for qid in self.ids["question_ids"]})
        gaps_svc.detect_all_for_user(conn, self.ids["student_id"])
        gap = conn.execute("SELECT * FROM learning_gaps WHERE user_id=? AND topic_id=?",
                          (self.ids["student_id"], self.ids["topic_id"])).fetchone()
        self.assertEqual(gap["status"], "resolved")
        conn.close()


# ==================================================================== progress ====

class ProgressTests(EduAnalyticsTestCase):
    def test_progress_recomputed_after_attempt_and_material(self):
        self.login_student()
        mat = connect(self.db_path).execute("SELECT id FROM materials WHERE type='notes'").fetchone()
        self.client.post(f"/student/materials/{mat['id']}/complete")
        attempt_id = self.start_attempt(f"/student/practice/{self.ids['topic_id']}/start")
        r = self.client.get(f"/student/attempt/{attempt_id}")
        qids = self.question_ids_on_page(r.data.decode())
        form = {f"q_{qid}": "A" for qid in qids}
        form["elapsed_sec"] = "20"
        self.client.post(f"/student/attempt/{attempt_id}", data=form)

        conn = connect(self.db_path)
        prog = conn.execute("SELECT * FROM student_progress WHERE user_id=? AND topic_id=?",
                           (self.ids["student_id"], self.ids["topic_id"])).fetchone()
        conn.close()
        self.assertIsNotNone(prog)
        self.assertEqual(prog["materials_done"], 1)
        self.assertEqual(prog["answers_correct"], len(qids))
        self.assertGreater(prog["progress_pct"], 0)

        r = self.client.get("/student/progress")
        self.assertIn(b"Test Topic", r.data)

    def test_progress_never_exceeds_100_percent_on_repeated_practice(self):
        self.login_student()
        for _ in range(4):
            attempt_id = self.start_attempt(f"/student/practice/{self.ids['topic_id']}/start")
            r = self.client.get(f"/student/attempt/{attempt_id}")
            qids = self.question_ids_on_page(r.data.decode())
            form = {f"q_{qid}": "A" for qid in qids}
            form["elapsed_sec"] = "20"
            self.client.post(f"/student/attempt/{attempt_id}", data=form)
        conn = connect(self.db_path)
        prog = conn.execute("SELECT * FROM student_progress WHERE user_id=? AND topic_id=?",
                           (self.ids["student_id"], self.ids["topic_id"])).fetchone()
        conn.close()
        self.assertLessEqual(prog["progress_pct"], 100.0)


# ==================================================================== admin CRUD ====

class AdminCRUDTests(EduAnalyticsTestCase):
    def test_admin_can_create_edit_delete_subject(self):
        self.login_admin()
        r = self.client.post("/admin/subjects/new", data={
            "name": "New Subject", "category": "Cat", "description": "Desc"}, follow_redirects=True)
        self.assertIn(b"New Subject", r.data)
        conn = connect(self.db_path)
        sid = conn.execute("SELECT id FROM subjects WHERE name='New Subject'").fetchone()["id"]
        conn.close()
        r = self.client.post(f"/admin/subjects/{sid}/edit", data={
            "name": "Renamed Subject", "category": "Cat", "description": "Desc"}, follow_redirects=True)
        self.assertIn(b"Renamed Subject", r.data)
        r = self.client.post(f"/admin/subjects/{sid}/delete", follow_redirects=True)
        self.assertNotIn(b"Renamed Subject", r.data)

    def test_admin_question_form_rejects_duplicate_options(self):
        self.login_admin()
        r = self.client.post(f"/admin/topics/{self.ids['topic_id']}/questions/new", data={
            "text": "Bad question?", "option_a": "X", "option_b": "X", "option_c": "Y", "option_d": "Z",
            "correct_option": "A", "explanation": "e", "difficulty": "Easy"})
        self.assertIn(b"must be distinct", r.data)

    def test_admin_material_requires_real_url_for_link_types(self):
        self.login_admin()
        r = self.client.post(f"/admin/topics/{self.ids['topic_id']}/materials/new", data={
            "title": "Broken video", "type": "video", "description": "x"})
        self.assertIn(b"required", r.data.lower())
        conn = connect(self.db_path)
        exists = conn.execute("SELECT COUNT(*) c FROM materials WHERE title='Broken video'").fetchone()["c"]
        conn.close()
        self.assertEqual(exists, 0)

    def test_admin_can_deactivate_student(self):
        self.login_admin()
        r = self.client.post(f"/admin/students/{self.ids['student_id']}/toggle-active", follow_redirects=True)
        self.assertEqual(r.status_code, 200)
        conn = connect(self.db_path)
        active = conn.execute("SELECT is_active FROM users WHERE id=?", (self.ids["student_id"],)).fetchone()["is_active"]
        conn.close()
        self.assertEqual(active, 0)

    def test_deactivated_student_cannot_log_in(self):
        self.login_admin()
        self.client.post(f"/admin/students/{self.ids['student_id']}/toggle-active")
        self.client.get("/logout")
        r = self.login_student()
        self.assertEqual(r.status_code, 200)  # re-renders login, not redirected


if __name__ == "__main__":
    unittest.main()
