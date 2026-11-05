# PhishLens

PhishLens is a shared-dashboard phishing and spam analysis project. It has no
login, registration, or per-user history. The Flask API serves analysis results
and MongoDB-backed shared statistics; the existing HTML, CSS, and JavaScript
frontend can run separately using Live Server.

## Backend setup

Use the existing virtual environment. Install dependencies only if they are
missing:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install MongoDB Community Server for Windows and start its `MongoDB` Windows
service (or start `mongod` with a local data directory). Confirm the server is
listening on `127.0.0.1:27017`. This project already has PyMongo installed; no
additional Python package is required.

Set the connection settings in the same PowerShell window used to start Flask.
`.env.example` is a reference only; the application does not load `.env` files.
The URI below contains no credentials and is the default local MongoDB address:

```powershell
$env:MONGODB_URI = "mongodb://127.0.0.1:27017/"
$env:MONGODB_DATABASE = "phishlens"
$env:MONGODB_COLLECTION = "analysis_history"
$env:CORS_ALLOWED_ORIGINS = "http://localhost:5500,http://127.0.0.1:5500"
Get-Service MongoDB
Start-Service MongoDB
.\.venv\Scripts\python.exe app.py
```

If `Get-Service MongoDB` reports that the service is not installed, install
MongoDB Community Server first. Starting a service may require an elevated
PowerShell window. Alternatively, start `mongod` directly with a data directory
and leave it running while Flask is in use. Check `GET /api/health`; its
`database.available` field should be `true` before relying on shared history.

For a hosted MongoDB deployment, provide its connection string through
`MONGODB_URI`; do not commit or share credentials. When MongoDB is not
configured or reachable, analysis responses explicitly report that records
were not saved, and shared history/statistics endpoints return HTTP 503 rather
than invented or user-local data. Analysis records store the subject or URL
and detection results, not email bodies or sender addresses. URL query strings
and fragments are removed from stored history; older URL targets are redacted
when the application connects to MongoDB. A 90-day TTL index expires shared
records automatically (MongoDB TTL deletion is asynchronous). History is
shared across the application with no accounts or per-user records, so anyone
who can access the app may see stored analysis targets and results. Do not
submit confidential or personal information.

## Email dataset preparation and training

The project's `emails.csv` is an existing tracked placeholder and is not
overwritten with message text. Dataset preparation writes the local-only
`.local_data/emails.csv` with `email_text,label` columns; `.local_data/`,
`classifier.joblib`, and `vectorizer.joblib` are ignored by Git.

The approved source for local academic training is the
[Apache SpamAssassin Public Mail Corpus](https://spamassassin.apache.org/old/publiccorpus/readme.html).
Its README reports 6,047 messages (about 31% spam): 500 + 1,397 spam and
2,500 + 250 + 1,400 legitimate messages. The category labels map directly to
`spam` and `legitimate`. The README describes the corpus as a resource for
spam-filter developers and says the messages came from public fora, senders
aware of publication, or public newsletters. It does **not** specify an open
dataset license and explicitly says copyright for message text remains with
original senders. Use is therefore restricted in this project to the user's
approved local academic training; do not redistribute the source messages,
normalized CSV, or model artifacts.

The preparation script downloads only the named official SpamAssassin
archives, parses email MIME messages without extracting archive paths, and
creates subject/body text. It writes archive SHA-256 hashes and measured
conversion/deduplication counts to `.local_data/dataset_report.json`.
The selected archives are `20030228_spam`, `20050311_spam_2`,
`20030228_easy_ham`, `20030228_hard_ham`, and `20030228_easy_ham_2`. The
corrected 2005 `spam_2` archive contains 1,396 numbered messages; the README's
headline category total still says 1,397 despite its revision note removing
one misclassified spam message. The selected archives therefore contain
6,046 message files, one fewer than the README's 6,047 headline total.
Non-message `cmds` helper files are excluded.

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py
.\.venv\Scripts\python.exe train_model.py
```

Labels are normalized from `spam`, `legitimate`, `legit`, or `ham` (`ham`
becomes `legitimate`); ambiguous numeric labels are rejected. Whitespace-only
text variants of duplicate messages are treated as exact duplicates. All
records with identical text but conflicting labels are excluded and counted;
same-label duplicates and empty/unusable messages are also removed and
reported.

The trainer uses a stratified held-out test split before fitting TF-IDF; only
the training portion is used to fit the vectorizer and Multinomial Naive Bayes
classifier. It reports accuracy, spam precision/recall/F1, and a confusion
matrix from the held-out portion. The classifier and fitted vectorizer are
saved separately as `classifier.joblib` and `vectorizer.joblib`. Until both
artifacts exist, `/api/email/analyze` returns HTTP 503 rather than fabricating
a prediction.

`PHISHLENS_DATASET_PATH`, `PHISHLENS_CLASSIFIER_PATH`, and
`PHISHLENS_VECTORIZER_PATH` can override these local defaults. Dataset
preparation refuses to write over the tracked root `emails.csv`.

### Measured Phase 2 preparation and evaluation

Results from the actual local run (Python 3.13, scikit-learn 1.9.1; held-out
spam is the positive class):

- 6,046 numbered source messages were parsed; no messages lacked extractable
  subject/body text.
- 130 duplicate text records were removed; no conflicting-label duplicates
  were found. One record became empty under the project's text preprocessing
  and was excluded before training.
- Model input: 5,915 unique usable messages — 4,108 legitimate and 1,807 spam.
- Stratified split: 4,732 training messages and 1,183 held-out test messages.
  TF-IDF was fit only on the training partition.
- Accuracy: **90.28%**; spam precision: **100.00%**; spam recall: **68.14%**;
  spam F1: **81.05%**.
- Confusion matrix (rows = actual, columns = predicted; order
  `legitimate, spam`): `[[822, 0], [115, 246]]`.
- An additional check on 10 held-out messages produced 9 correct predictions.
  Their message text is not included in this repository.

These are results on a historical SpamAssassin corpus, not a guarantee of
performance on current email. The perfect held-out spam precision accompanies
115 missed spam messages in this split; spam recall should be considered when
using this model. Counts, per-archive SHA-256 values, labels, and evaluation
details are recorded in the ignored local `.local_data/dataset_report.json`
and `.local_data/training_report.json`. Re-run training to reproduce metrics;
the source corpus is not distributed with this project.

## API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Backend homepage |
| GET | `/api/health` | Flask and MongoDB availability |
| POST | `/api/preprocess-email` | Backward-compatible email preprocessing |
| POST | `/api/email/analyze` | Trained TF-IDF/Naive Bayes email analysis |
| POST | `/api/url/analyze` | Rule-based URL analysis; never opens the submitted URL |
| GET | `/api/history` | Shared history (`type`, `risk`, `search`, `limit`, `offset` filters) |
| GET | `/api/dashboard/stats` | Shared analysis and risk statistics |
| GET | `/api/reports` | Shared report statistics |

Email analysis persists the subject and indicators, not the submitted message
body or sender address. Risk scores are explainable heuristics, not probabilities
or guarantees that a message or URL is malicious or safe.

## Tests

Run the backend unit tests without installing a test framework:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
