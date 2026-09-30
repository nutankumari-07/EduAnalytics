# EduAnalytics

**An AI-Powered Web Platform for Personalized Student Learning Pathways**
B.Tech Final-Year Project (PRJ_538)

EduAnalytics is a full-stack EdTech platform where students practice, take quizzes and
timed assessments across a real 395-question engineering-curriculum question bank, get a
genuine per-topic learning-gap analysis (not a static label), receive recommendations that
link to real materials, and see a live machine-learning prediction of their performance
level. Admins manage every piece of content, review platform-wide analytics, and inspect
the trained model's metrics.

Everything in this repo is real and wired end-to-end: there are no placeholder questions,
no locked content, no dead links, and no hard-coded dashboard numbers.

---

## 1. Problem statement

Traditional LMS platforms show students a single overall score and leave them to figure
out what to study next. EduAnalytics instead:

- Breaks performance down **per topic and per difficulty band**, so a student knows
  *exactly* which concept is weak, not just that their average is low.
- Uses a trained ML model to **predict a performance level** (Low / Medium / High) from a
  student's attendance, study habits, and live learning analytics — and explains *why*
  via per-student feature contribution, not a single cohort-wide importance chart.
- Turns every detected weakness into a **clickable recommendation** pointing at a real
  topic, material, or practice set — never a generic "study more" message.
- Gives admins the same visibility platform-wide: which topics are commonly weak, how the
  model is performing, and what every student's gaps look like.

## 2. Objectives

1. Build a database-driven question bank across 9 engineering subjects, large enough for
   real practice (395 questions, not a demo set of 5–10).
2. Implement practice, quiz and timed-assessment modes with automatic scoring.
3. Detect learning gaps from actual attempt history (accuracy by topic and by difficulty,
   repeated mistakes, recency) rather than a static rule.
4. Train and serve a real scikit-learn model, with a full metrics page (accuracy,
   precision, recall, F1, confusion matrix, feature importance, model comparison).
5. Give admins full CRUD over subjects, topics, materials, and questions, with validation
   that makes a dummy/broken link impossible to save.
6. Ship a genuinely populated app: seeded students, attempts, gaps and recommendations, so
   nothing is empty on first run.

## 3. Features

### Student
- Dashboard with predicted performance level, subject progress, recent activity,
  personalized recommendations and notifications.
- Browse 9 subjects / 79 topics, each with study notes, a reference reading link, and a
  practice set.
- Practice (untimed), topic quizzes, and subject-wide timed assessments, all auto-scored.
- Detailed result pages with per-question correctness and explanations.
- **Learning Gaps** page: real per-topic accuracy analysis, weak-difficulty callouts, and
  concrete suggested actions.
- **Recommendations** page: generated directly from active gaps and unexplored topics.
- **Performance** page: ML-predicted level, model confidence, and a per-student factor
  breakdown (which of *your* numbers are pulling the prediction up or down).
- Progress tracker, full history (filterable by practice/quiz/assessment), profile, search.

### Admin
- Dashboard with live counts, a real performance-level distribution chart, and a "most
  common weak topics" panel driven by actual learning-gap data.
- Full CRUD: subjects (with cover-image upload), topics, learning materials, and the
  question bank (with subject/topic/difficulty/text filters).
- Material form **rejects** article/video/PDF entries without a real URL — the exact
  anti-pattern (dummy 404 links) this project was built to avoid.
- Quiz and assessment publish/unpublish controls.
- Results & Analytics: score by subject, by attempt type, 14-day trend, weak-topic list.
- Learning Gaps and Recommendations, platform-wide, filterable by severity.
- ML Analytics: full metrics for the trained model, plus a one-click retrain.
- Account management for admin users.

### Machine learning (overall performance level)
The model classifies a student's **overall performance level** (Low / Medium / High). It does
**not** detect individual topic gaps — that is a separate, threshold-based component (below).

**One dataset, one feature definition.** `services/features.py` defines the 8 features
(attendance %, study hours/week, previous GPA, quiz average, practice accuracy, assessment
average, average topic score, material completion %) and computes them from the live database.
`ml/train_model.py` builds the training set from the database (features) joined with
`student_outcomes` (the synthetic end-of-term result used as the target). Nothing is imported
from the old project and no stale metrics are reused: every run rewrites
`data/student_performance.csv`, `models/performance_model.pkl`, `models/model_metrics.json` and
`models/model_comparison.csv`.

**Automatic model selection.** All candidates use the same data, features, `StandardScaler`
preprocessing and one stratified 80/20 split (`random_state=42`); each also gets 5-fold stratified
CV on the training part. Candidates: Logistic Regression, Decision Tree, Random Forest, KNN, SVM,
Gradient Boosting, Naive Bayes (mandatory) plus Extra Trees, and XGBoost *only if the `xgboost`
package is installed* (it is optional; it was not installed in the build environment, so that
code path is untested here and the ML page lists it under "Not evaluated"). The winner is the
highest **weighted F1** on the held-out test set. Models within 0.5 pt of the best are treated as a
tie and broken by **cross-validation stability**, scored as CV weighted-F1 mean minus its std (so a
model that is merely consistently worse cannot win on low variance alone). The chosen model name and
the reason text come from the training run, not from code.

**Result of the shipped build** (1001 students, train 800 / test 201):
selected **Logistic Regression** — weighted F1 76.0%, accuracy 76.1%,
CV weighted F1 78.5% ± 2.1.

| Model | Accuracy | Weighted F1 | CV weighted F1 |
|---|---|---|---|
| Extra Trees | 76.6% | 76.3% | 76.8% ± 2.7 |
| Logistic Regression **(selected)** | 76.1% | 76.0% | 78.5% ± 2.1 |
| Gradient Boosting | 75.1% | 75.1% | 75.7% ± 1.9 |
| SVM | 75.1% | 74.9% | 77.9% ± 3.4 |
| Naive Bayes | 73.1% | 73.1% | 73.5% ± 1.1 |
| Random Forest | 73.1% | 73.0% | 76.6% ± 1.9 |
| KNN | 73.1% | 73.0% | 75.1% ± 2.4 |
| Decision Tree | 67.7% | 67.7% | 74.5% ± 1.8 |

Honest reading: the top models are within about a point of each other, so which one "wins" is
sensitive to the random draw of the synthetic data; the tie-break rule exists for exactly that.
The data is synthetic (a latent ability/effort model), so these numbers say how well the pipeline
works, not how well it would predict real students.

**No fabricated predictions.** A student gets a prediction only after completing their profile
(attendance %, study hours, previous GPA) and having at least 1 quiz, 1 practice, 1 assessment,
3 scored topics and 6 attempts. Newly registered accounts start with empty (NULL) profile fields and
no outcome, and are excluded from training until they qualify. Students never see model names or
metrics; the ML Analytics page is admin-only.

### Learning-gap detection (the real analysis)
For every topic a student has answered questions in, `services/gaps.py` computes:
overall accuracy, accuracy broken down by Easy/Medium/Hard, how many attempts touched the
topic, and how many questions have been missed more than once. Severity (High / Medium /
Low / none) is derived from these, and the observed-issue text is generated per student
(e.g. *"Scoring 42% on Arrays across 3 attempts... particularly weak on Hard-difficulty
questions (20%)... 2 question(s) answered incorrectly more than once"*) — not a canned
sentence.

### Recommendation engine
`services/recommend.py` combines two separate signals: (1) threshold-based **learning gaps**
(topic level) and (2) the ML **performance level** when the student qualifies for one. Each active
gap becomes a recommendation linking to the best material for that topic; a Low level raises gap
priorities and adds a "build your foundations" pick on a Beginner topic, a High level prefers
practice material and adds a "challenge yourself" pick on an Advanced topic; remaining slots are
"explore this next" suggestions for untouched topics. The ML level never creates or removes a gap.

---

## 4. Tech stack

- **Backend:** Python 3, Flask (blueprints: `auth`, `student`, `admin`, `main`)
- **Database:** SQLite via the stdlib `sqlite3` module (no ORM) — see `db.py` for schema
  and connection handling
- **ML:** scikit-learn, pandas, numpy, joblib
- **Frontend:** Jinja2 templates, hand-written CSS (dark EdTech theme). Charts are rendered
  server-side as inline SVG/CSS (`templates/_charts.html`) — no JavaScript chart library or CDN
- **Images:** Course cover art generated procedurally with Pillow (no external assets)
- **Auth:** Flask sessions, Werkzeug password hashing, role-based decorators with real 403s

## 5. Architecture

```
app.py                 Flask app factory, error handlers, Jinja filters
config.py               Config (paths, upload limits, session settings)
db.py                    Schema (SCHEMA string) + connection helpers
blueprints/
  auth.py                login / register / logout
  student.py             all student-facing routes
  admin.py                all admin-facing routes
  main.py                 role-aware root redirect
services/
  auth_helpers.py         login_required / role_required decorators, current_user()
  scoring.py               attempt grading
  progress.py              per-topic progress aggregation
  gaps.py                  learning-gap detection (the real analysis)
  recommend.py             recommendation generation (gaps + ML level)
  features.py              the single ML feature definition + "enough activity" gate
ml/
  train_model.py           DB-derived dataset, multi-model comparison, automatic selection, metrics
  predict.py               model loading, prediction (single + batch), per-student explanation
scripts/
  init_db.py               create schema (--reset to rebuild from scratch)
  seed_data.py             build(): curriculum + questions + students + activity + gaps + model + recs
  seed_students.py         generates 1000+ students and simulates their quiz/practice/assessment history
  generate_images.py       procedural course cover art
  data/                    curriculum.py + q_<subject>.py question banks
templates/               auth/ student/ admin/ errors/ + base.html
static/                  css/ (style.css) images/courses/ uploads/
tests/                   unittest suite + isolated-DB fixtures
data/                    student_performance.csv (exported training set), seeded_students.csv (demo logins)
models/                  performance_model.pkl + model_metrics.json + model_comparison.csv
```

### Database schema (highlights)

`users`, `subjects`, `topics`, `materials`, `questions`, `quizzes` / `quiz_questions`,
`assessments` / `assessment_questions`, `enrollments`, `attempts`, `attempt_answers`,
`material_progress`, `student_progress`, `learning_gaps`, `recommendations`,
`notifications`, `student_outcomes` (synthetic end-of-term result = ML target) — all with foreign keys (`ON DELETE CASCADE`) and indexes on the hot
query paths. See `db.py` for the full `CREATE TABLE` statements.

## 6. Install & run

```bash
cd EduAnalytics
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py                      # ready to run: the ZIP ships a seeded database + trained model
```

Open **http://127.0.0.1:5000**.

### Rebuilding from scratch (optional, ~45 s)
```bash
python -m scripts.init_db --reset  # empty schema
python -m scripts.seed_data        # curriculum, 1,001 students, activity, gaps, trains + selects the model, recommendations
# python -m scripts.seed_data --students 300   # smaller cohort if you prefer
```
The seed is deterministic (same numbers every time). Retraining alone: `python -m ml.train_model`
or the **Retrain & compare models** button on the admin ML Analytics page (both train from the
current database, so students added or changed since seeding are included once they qualify).

### Demo credentials
| Role    | Email                      | Password    |
|---------|-----------------------------|-------------|
| Student | student@eduanalytics.com   | student123  |
| Admin   | admin@eduanalytics.com     | admin123    |

The seed also creates **1,000 generated students** (student codes `STU20240001`–`STU20241000`).
Every one has a unique code and email, password `student123`, its own attendance / study hours / GPA,
and its own quiz, practice and assessment history. All logins are listed in `data/seeded_students.csv`
(demo data — plaintext passwords for convenience only; the database stores hashes). New students who
register through `/register` get a real database record, a unique student code and a hashed password.

### Fonts
The UI typeface (Google Fonts) is loaded from a CDN by `static/css/style.css` and falls back to
system fonts offline. Charts do not depend on any CDN.

## 7. Running the tests

```bash
python -m unittest discover -s tests -v
```

56 tests. The original 32 cover authentication, RBAC, content integrity, scoring, gap detection,
recommendations, progress and admin CRUD on a tiny fixture database. `tests/test_seeded.py` (24 tests)
builds a full seeded database in a temp directory (150 students, the real curriculum, a real training
run) and covers: unique student IDs and credentials, varied per-student data, seeded scores matching the
real grader, multiple student logins, cross-student data isolation, no ML details on student pages,
all student pages and the practice/quiz/assessment flows, admin pages and student search, analytics
charts populated from (and reacting to) the database, all required models compared on identical setup,
the selection rule, the saved model matching the metrics file, admin-only ML page, retraining from the
admin page, graceful failure when data is insufficient, and registration (persisted record, hashed
password, unique code, no prediction until profile + activity are complete). Tests never touch the
real `database/` or `models/`.

## 8. Retraining / regenerating content

- **Retrain the model:** `python -m ml.train_model`, or from the admin ML Analytics page.
- **Regenerate course cover art:** `python scripts/generate_images.py`.
- **Rebuild the database from scratch:**
  `python -m scripts.init_db --reset && python -m scripts.seed_data`.

## 9. Validation performed

- Full test suite: 56/56 passing.
- Live check against the shipped database over real HTTP: admin login; Results & Analytics shows
  the level distribution (1,001 students), all 9 subject averages and a 14-point trend; ML Analytics
  matches `models/model_metrics.json`; 25 randomly chosen students could log in, load every student
  page, saw only their own data, got 404 on another student's attempt and 403 on admin pages.
- Reproducibility: two independent seed builds produced identical metrics for all models.
- Empty analytics charts: on the original database the three queries did return rows (checked directly),
  and the chart data was present in the served HTML, so the blank areas were most likely caused by the
  client-side Chart.js script (loaded from a CDN) not running in the browser. This could not be reproduced
  or confirmed without a browser. Either way, the charts are now drawn server-side as SVG/CSS, need no
  JavaScript or network, and the database is seeded with enough attempts to populate them.
- Not verified: the optional XGBoost path (package unavailable in the build environment) and visual
  rendering in a real browser (no browser was available; pages were checked as served HTML).

## 10. Future scope

- Spaced-repetition scheduling for weak topics.
- Peer comparison / leaderboards (opt-in).
- Export of a student's performance report as PDF.
- Support for open-ended / code-execution questions alongside MCQs.
