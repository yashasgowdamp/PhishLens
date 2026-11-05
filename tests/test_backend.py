import os
import tempfile
import unittest
import base64
import hashlib
import stat
from unittest.mock import patch
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock
from pymongo.errors import OperationFailure

import app
from app import create_app
import database
from email_analyzer import (
    EmailModelUnavailable,
    _decode_secret_model_files,
    _load_model,
    analyze_email_text,
    get_model_paths,
)
from prepare_dataset import DatasetPreparationError, prepare_dataset
from risk_engine import assess_risk, assess_url_risk, combine_email_risk
from train_model import DatasetError, load_labeled_dataset
from url_analyzer import URLAnalysisError, analyze_url


class URLAnalyzerTests(unittest.TestCase):
    def test_https_url_without_indicators_is_low_risk(self):
        result = analyze_url("https://example.com/contact")
        self.assertEqual(result["risk_level"], "low")
        self.assertEqual(result["indicators"], [])
        self.assertEqual(result["classification"], "No suspicious indicators detected")

    def test_indicators_are_reported_without_opening_the_url(self):
        result = analyze_url("http://192.0.2.1/login")
        self.assertIn("URL contains an IP address instead of a domain", result["indicators"])
        self.assertIn("URL uses an insecure HTTP scheme", result["indicators"])
        self.assertEqual(result["risk"]["is_probability"], False)

    def test_shortener_and_url_length_indicators(self):
        result = analyze_url(f"https://bit.ly/{'a' * 110}")
        self.assertIn("Unusually long URL", result["indicators"])
        self.assertIn("URL uses a known shortening service", result["indicators"])

    def test_invalid_scheme_is_rejected(self):
        with self.assertRaises(URLAnalysisError):
            analyze_url("javascript:alert(1)")

    def test_url_credentials_are_rejected(self):
        with self.assertRaises(URLAnalysisError):
            analyze_url("https://user:password@example.com/")


class RiskEngineTests(unittest.TestCase):
    def test_thresholds_are_bounded_and_explainable(self):
        self.assertEqual(assess_risk(30, []).get("risk_level"), "low")
        self.assertEqual(assess_risk(31, []).get("risk_level"), "medium")
        self.assertEqual(assess_risk(1000, ["indicator"])["risk_score"], 100)

    def test_email_risk_includes_model_and_link_reasons(self):
        url_result = analyze_url("http://192.0.2.1/login")
        result = combine_email_risk("Spam", 0.9, url_result)
        self.assertEqual(result["risk_level"], "high")
        self.assertTrue(any("email model" in reason for reason in result["warning_reasons"]))
        self.assertIn(
            "URL contains an IP address instead of a domain",
            result["warning_reasons"],
        )

    def test_empty_url_assessment_is_not_a_phishing_claim(self):
        result = assess_url_risk([])
        self.assertEqual(result["risk_level"], "low")
        self.assertIn("No configured suspicious URL indicators", result["warning_reasons"][0])


class DatasetValidationTests(unittest.TestCase):
    def test_empty_project_dataset_is_rejected(self):
        with self.assertRaisesRegex(DatasetError, "contains no records"):
            load_labeled_dataset(os.path.join(os.path.dirname(__file__), "..", "emails.csv"))

    def test_missing_dataset_columns_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "unlabeled.csv")
            with open(path, "w", encoding="utf-8") as dataset_file:
                dataset_file.write("message\n")
            with self.assertRaisesRegex(DatasetError, "text.*label"):
                load_labeled_dataset(path)

    def test_preparation_refuses_to_overwrite_tracked_emails_csv(self):
        project_root = os.path.dirname(os.path.dirname(__file__))
        with self.assertRaisesRegex(DatasetPreparationError, "Refusing to overwrite"):
            prepare_dataset(output_path=os.path.join(project_root, "emails.csv"))


class EmailModelTests(unittest.TestCase):
    def setUp(self):
        _load_model.cache_clear()
        _decode_secret_model_files.cache_clear()

    def tearDown(self):
        _load_model.cache_clear()
        _decode_secret_model_files.cache_clear()

    def test_missing_trained_model_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(EmailModelUnavailable):
                _load_model(
                    os.path.join(directory, "missing-classifier.joblib"),
                    os.path.join(directory, "missing-vectorizer.joblib"),
                )

    def _write_trained_model(self, directory):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB

        texts = [
            "meeting schedule project team",
            "team report project meeting",
            "win prize claim free cash",
            "claim cash prize now",
        ]
        labels = ["legitimate", "legitimate", "spam", "spam"]
        vectorizer = TfidfVectorizer()
        features = vectorizer.fit_transform(texts)
        classifier = MultinomialNB().fit(features, labels)
        classifier_path = os.path.join(directory, "classifier.joblib")
        vectorizer_path = os.path.join(directory, "vectorizer.joblib")
        import joblib

        joblib.dump(classifier, classifier_path)
        joblib.dump(vectorizer, vectorizer_path)
        return classifier_path, vectorizer_path

    def test_local_joblib_paths_continue_to_load(self):
        with tempfile.TemporaryDirectory() as directory:
            classifier_path, vectorizer_path = self._write_trained_model(directory)
            with (
                patch.dict(os.environ, {
                    "PHISHLENS_CLASSIFIER_PATH": classifier_path,
                    "PHISHLENS_VECTORIZER_PATH": vectorizer_path,
                }),
                patch.dict(os.environ, {
                    "PHISHLENS_CLASSIFIER_BASE64_FILE": "",
                    "PHISHLENS_VECTORIZER_BASE64_FILE": "",
                }),
            ):
                result = analyze_email_text("Win a free cash prize now")

        self.assertEqual(result["classification"], "Spam")

    def test_base64_secret_files_decode_and_load_models(self):
        with tempfile.TemporaryDirectory() as directory:
            classifier_path, vectorizer_path = self._write_trained_model(directory)
            hashes = {}
            secret_paths = []
            for name, path in (
                ("classifier", classifier_path),
                ("vectorizer", vectorizer_path),
            ):
                secret_path = path + ".b64"
                with open(path, "rb") as model_file, open(
                    secret_path,
                    "w",
                    encoding="ascii",
                ) as secret_file:
                    secret_file.write(
                        base64.b64encode(model_file.read()).decode("ascii") + "\n"
                    )
                secret_paths.append(secret_path)
                with open(path, "rb") as model_file:
                    hashes[name] = hashlib.sha256(model_file.read()).hexdigest()

            with patch.dict(os.environ, {
                "PHISHLENS_CLASSIFIER_BASE64_FILE": secret_paths[0],
                "PHISHLENS_VECTORIZER_BASE64_FILE": secret_paths[1],
                "PHISHLENS_CLASSIFIER_SHA256": hashes["classifier"],
                "PHISHLENS_VECTORIZER_SHA256": hashes["vectorizer"],
            }, clear=False):
                decoded_paths = get_model_paths()
                loaded_result = analyze_email_text("Win a free cash prize now")

            self.assertTrue(all(os.path.isfile(path) for path in decoded_paths))
            self.assertEqual(loaded_result["classification"], "Spam")
            if os.name == "posix":
                self.assertEqual(
                    stat.S_IMODE(os.stat(os.path.dirname(decoded_paths[0])).st_mode),
                    0o700,
                )
                self.assertTrue(all(
                    stat.S_IMODE(os.stat(path).st_mode) == 0o600
                    for path in decoded_paths
                ))

    def test_base64_secret_file_configuration_requires_both_files(self):
        with patch.dict(os.environ, {
            "PHISHLENS_CLASSIFIER_BASE64_FILE": "classifier.b64",
            "PHISHLENS_VECTORIZER_BASE64_FILE": "",
        }, clear=False):
            with self.assertRaisesRegex(EmailModelUnavailable, "Both"):
                get_model_paths()

    def test_invalid_base64_secret_file_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            classifier_secret = os.path.join(directory, "classifier.b64")
            vectorizer_secret = os.path.join(directory, "vectorizer.b64")
            with open(classifier_secret, "w", encoding="ascii") as secret_file:
                secret_file.write("not valid base64!")
            with open(vectorizer_secret, "w", encoding="ascii") as secret_file:
                secret_file.write("dGVzdA==")

            with patch.dict(os.environ, {
                "PHISHLENS_CLASSIFIER_BASE64_FILE": classifier_secret,
                "PHISHLENS_VECTORIZER_BASE64_FILE": vectorizer_secret,
                "PHISHLENS_CLASSIFIER_SHA256": hashlib.sha256(b"classifier").hexdigest(),
                "PHISHLENS_VECTORIZER_SHA256": hashlib.sha256(b"vectorizer").hexdigest(),
            }, clear=False):
                with self.assertRaisesRegex(EmailModelUnavailable, "valid Base64"):
                    get_model_paths()

    def test_base64_secret_files_reject_empty_files(self):
        with tempfile.TemporaryDirectory() as directory:
            secret_paths = []
            for name in ("classifier", "vectorizer"):
                secret_path = os.path.join(directory, name + ".b64")
                with open(secret_path, "w", encoding="ascii") as secret_file:
                    secret_file.write("")
                secret_paths.append(secret_path)

            with patch.dict(os.environ, {
                "PHISHLENS_CLASSIFIER_BASE64_FILE": secret_paths[0],
                "PHISHLENS_VECTORIZER_BASE64_FILE": secret_paths[1],
                "PHISHLENS_CLASSIFIER_SHA256": hashlib.sha256(b"classifier").hexdigest(),
                "PHISHLENS_VECTORIZER_SHA256": hashlib.sha256(b"vectorizer").hexdigest(),
            }, clear=False):
                with self.assertRaisesRegex(EmailModelUnavailable, "is empty"):
                    get_model_paths()

    def test_missing_base64_secret_file_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {
                "PHISHLENS_CLASSIFIER_BASE64_FILE": os.path.join(
                    directory,
                    "missing-classifier.b64",
                ),
                "PHISHLENS_VECTORIZER_BASE64_FILE": os.path.join(
                    directory,
                    "missing-vectorizer.b64",
                ),
                "PHISHLENS_CLASSIFIER_SHA256": hashlib.sha256(b"classifier").hexdigest(),
                "PHISHLENS_VECTORIZER_SHA256": hashlib.sha256(b"vectorizer").hexdigest(),
            }, clear=False):
                with self.assertRaisesRegex(
                    EmailModelUnavailable,
                    "could not be found",
                ):
                    get_model_paths()

    def test_base64_secret_files_require_expected_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            secret_paths = []
            for name in ("classifier", "vectorizer"):
                secret_path = os.path.join(directory, name + ".b64")
                with open(secret_path, "w", encoding="ascii") as secret_file:
                    secret_file.write(base64.b64encode(name.encode()).decode("ascii"))
                secret_paths.append(secret_path)

            with patch.dict(os.environ, {
                "PHISHLENS_CLASSIFIER_BASE64_FILE": secret_paths[0],
                "PHISHLENS_VECTORIZER_BASE64_FILE": secret_paths[1],
                "PHISHLENS_CLASSIFIER_SHA256": "",
                "PHISHLENS_VECTORIZER_SHA256": "",
            }, clear=False):
                with self.assertRaisesRegex(EmailModelUnavailable, "SHA-256"):
                    get_model_paths()

    def test_base64_secret_files_reject_oversized_file_before_reading(self):
        for oversized_model in ("classifier", "vectorizer"):
            with self.subTest(model=oversized_model):
                with tempfile.TemporaryDirectory() as directory:
                    secret_paths = {
                        "classifier": os.path.join(directory, "classifier.b64"),
                        "vectorizer": os.path.join(directory, "vectorizer.b64"),
                    }
                    for model_name, secret_path in secret_paths.items():
                        with open(secret_path, "wb") as secret_file:
                            if model_name == oversized_model:
                                secret_file.truncate(
                                    16 * 1024 * 1024 + 1
                                )
                            else:
                                secret_file.write(b"dGVzdA==")

                    with patch.dict(os.environ, {
                        "PHISHLENS_CLASSIFIER_BASE64_FILE": secret_paths["classifier"],
                        "PHISHLENS_VECTORIZER_BASE64_FILE": secret_paths["vectorizer"],
                        "PHISHLENS_CLASSIFIER_SHA256": hashlib.sha256(b"test").hexdigest(),
                        "PHISHLENS_VECTORIZER_SHA256": hashlib.sha256(b"test").hexdigest(),
                    }, clear=False):
                        original_open = open

                        def reject_oversized_read(path, *args, **kwargs):
                            if os.fspath(path) == secret_paths[oversized_model]:
                                raise AssertionError("Oversized Secret File was read")
                            return original_open(path, *args, **kwargs)

                        with patch(
                            "builtins.open",
                            side_effect=reject_oversized_read,
                        ):
                            with self.assertRaisesRegex(
                                EmailModelUnavailable,
                                "exceeds the .*byte size limit",
                            ):
                                get_model_paths()

    def test_base64_secret_files_reject_incorrect_hash_before_joblib_load(self):
        with tempfile.TemporaryDirectory() as directory:
            secret_paths = []
            for name in ("classifier", "vectorizer"):
                secret_path = os.path.join(directory, name + ".b64")
                with open(secret_path, "w", encoding="ascii") as secret_file:
                    secret_file.write(base64.b64encode(name.encode()).decode("ascii"))
                secret_paths.append(secret_path)

            with patch.dict(os.environ, {
                "PHISHLENS_CLASSIFIER_BASE64_FILE": secret_paths[0],
                "PHISHLENS_VECTORIZER_BASE64_FILE": secret_paths[1],
                "PHISHLENS_CLASSIFIER_SHA256": hashlib.sha256(
                    b"incorrect classifier"
                ).hexdigest(),
                "PHISHLENS_VECTORIZER_SHA256": hashlib.sha256(
                    b"vectorizer"
                ).hexdigest(),
            }, clear=False):
                with patch("email_analyzer.joblib.load") as joblib_load:
                    with self.assertRaisesRegex(
                        EmailModelUnavailable,
                        "integrity verification",
                    ):
                        analyze_email_text("Meeting schedule")

            joblib_load.assert_not_called()

    def test_invalid_joblib_artifact_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            classifier_path = os.path.join(directory, "classifier.joblib")
            vectorizer_path = os.path.join(directory, "vectorizer.joblib")
            for path in (classifier_path, vectorizer_path):
                with open(path, "wb") as artifact_file:
                    artifact_file.write(b"not a valid joblib artifact")

            with self.assertRaisesRegex(EmailModelUnavailable, "could not be loaded"):
                _load_model(classifier_path, vectorizer_path)


class BackendAPITests(unittest.TestCase):
    def setUp(self):
        self.client = create_app({"TESTING": True}).test_client()

    def test_home_and_health_are_available_without_mongodb(self):
        with patch("database.check_connection", side_effect=database.DatabaseUnavailable(
            "MONGODB_URI is not configured."
        )):
            self.assertEqual(self.client.get("/").status_code, 200)
            response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json["database"]["available"])

    def test_preprocessing_endpoint_validates_and_cleans_text(self):
        missing = self.client.post(
            "/api/preprocess-email",
            json={"email_text": "   "},
        )
        self.assertEqual(missing.status_code, 400)
        cleaned = self.client.post(
            "/api/preprocess-email",
            json={"email_text": "<b>URGENT</b> click https://example.com"},
        )
        self.assertEqual(cleaned.status_code, 200)
        self.assertEqual(cleaned.json["cleaned_text"], "urgent click")

    def test_non_json_and_non_object_requests_are_rejected(self):
        self.assertEqual(self.client.post("/api/url/analyze", data="url").status_code, 400)
        self.assertEqual(
            self.client.post(
                "/api/url/analyze",
                data="[]",
                content_type="application/json",
            ).status_code,
            400,
        )

    def test_url_analysis_returns_indicators_and_reports_database_failure(self):
        with patch("database.save_analysis", side_effect=database.DatabaseUnavailable(
            "MONGODB_URI is not configured."
        )) as save:
            response = self.client.post(
                "/api/url/analyze",
                json={"url": "http://192.0.2.1/login"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json["persistence"]["saved"])
        self.assertTrue(response.json["indicators"])
        save.assert_called_once()

    def test_email_analysis_returns_503_until_a_model_exists(self):
        with patch("app.analyze_email_text", side_effect=EmailModelUnavailable(
            "No trained model was found."
        )):
            response = self.client.post(
                "/api/email/analyze",
                json={"subject": "Hello", "body": "Meeting tomorrow."},
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json["status"], "error")

    def test_email_analysis_validates_sender_and_required_fields(self):
        response = self.client.post(
            "/api/email/analyze",
            json={"subject": "Hello", "body": "Meeting tomorrow.", "sender": "bad-address"},
        )
        self.assertEqual(response.status_code, 400)

    def test_email_analysis_combines_model_and_url_results(self):
        with (
            patch("app.analyze_email_text", return_value={
                "classification": "Spam",
                "confidence": 0.9,
            }),
            patch("database.save_analysis", side_effect=database.DatabaseUnavailable(
                "MONGODB_URI is not configured."
            )),
        ):
            response = self.client.post(
                "/api/email/analyze",
                json={
                    "subject": "Review this",
                    "body": "Please review the attached message.",
                    "url": "http://192.0.2.1/login",
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["classification"], "Spam")
        self.assertEqual(response.json["risk_level"], "high")
        self.assertFalse(response.json["persistence"]["saved"])
        self.assertIn(
            "URL contains an IP address instead of a domain",
            response.json["warning_reasons"],
        )

    def test_history_dashboard_and_reports_surface_database_unavailability(self):
        with patch("database.list_history", side_effect=database.DatabaseUnavailable("Mongo unavailable.")):
            history = self.client.get("/api/history")
        with patch("database.get_dashboard_stats", side_effect=database.DatabaseUnavailable("Mongo unavailable.")):
            dashboard = self.client.get("/api/dashboard/stats")
        with patch("database.get_report_statistics", side_effect=database.DatabaseUnavailable("Mongo unavailable.")):
            reports = self.client.get("/api/reports")
        self.assertEqual((history.status_code, dashboard.status_code, reports.status_code), (503, 503, 503))

    def test_history_filters_are_validated(self):
        self.assertEqual(self.client.get("/api/history?type=account").status_code, 400)
        self.assertEqual(self.client.get("/api/history?risk=critical").status_code, 400)
        self.assertEqual(self.client.get("/api/history?limit=1000").status_code, 400)

    def test_cors_is_restricted_to_the_configured_live_server_origin(self):
        allowed = self.client.get(
            "/api/health",
            headers={"Origin": "http://localhost:5500"},
        )
        rejected = self.client.get(
            "/api/health",
            headers={"Origin": "https://untrusted.example"},
        )
        self.assertEqual(allowed.headers.get("Access-Control-Allow-Origin"), "http://localhost:5500")
        self.assertIsNone(rejected.headers.get("Access-Control-Allow-Origin"))

    def test_startup_initializes_database_indexes(self):
        with patch("app.database.check_connection") as check_connection:
            app.initialize_database()

        check_connection.assert_called_once_with()

    def test_startup_logs_database_initialization_failure_and_continues(self):
        with (
            patch(
                "app.database.check_connection",
                side_effect=database.DatabaseUnavailable("MongoDB is unavailable."),
            ),
            patch("app.logger.warning") as warning,
        ):
            app.initialize_database()

        warning.assert_called_once()


class DatabaseBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.previous_client = database._client
        self.previous_uri = database._client_uri
        self.previous_initialized_collections = database._initialized_collections.copy()
        database._client = None
        database._client_uri = None
        database._initialized_collections.clear()

    def tearDown(self):
        database._client = self.previous_client
        database._client_uri = self.previous_uri
        database._initialized_collections.clear()
        database._initialized_collections.update(self.previous_initialized_collections)

    def test_collection_connection_creates_history_indexes_once(self):
        client = MagicMock()
        collection = MagicMock()
        client.__getitem__.return_value.__getitem__.return_value = collection
        with (
            patch.dict("os.environ", {
                "MONGODB_URI": "mongodb://127.0.0.1:27017/",
                "MONGODB_DATABASE": "phishlens",
                "MONGODB_COLLECTION": "analysis_history",
            }),
            patch("database.MongoClient", return_value=client) as mongo_client,
        ):
            first = database._database_collection()
            second = database._database_collection()

        self.assertIs(first, second)
        self.assertEqual(client.admin.command.call_count, 2)
        client.admin.command.assert_called_with("ping")
        mongo_client.assert_called_once_with(
            "mongodb://127.0.0.1:27017/",
            serverSelectionTimeoutMS=15_000,
            connectTimeoutMS=10_000,
        )
        self.assertEqual(collection.create_index.call_count, 4)
        self.assertEqual(
            collection.create_index.call_args_list[0].args[0],
            [("created_at", -1), ("_id", -1)],
        )
        retention_index = collection.create_index.call_args_list[3]
        self.assertEqual(retention_index.args[0], [("created_at", 1)])
        self.assertEqual(
            retention_index.kwargs["expireAfterSeconds"],
            database.ANALYSIS_RETENTION_SECONDS,
        )
        collection.find.assert_called_once_with(
            {"analysis_type": "url", "target": {"$regex": r"[?#]"}},
            {"_id": 1, "target": 1},
        )

    def test_collection_connection_redacts_legacy_url_targets(self):
        client = MagicMock()
        collection = MagicMock()
        collection.find.return_value = [{
            "_id": "record-1",
            "target": "https://example.com/path?token=private#session",
        }]
        client.__getitem__.return_value.__getitem__.return_value = collection
        with (
            patch.dict("os.environ", {"MONGODB_URI": "mongodb://127.0.0.1:27017/"}),
            patch("database.MongoClient", return_value=client),
        ):
            database._database_collection()

        collection.update_one.assert_called_once_with(
            {"_id": "record-1"},
            {"$set": {"target": "https://example.com/path"}},
        )

    def test_collection_connection_logs_ttl_index_creation_failure(self):
        client = MagicMock()
        collection = MagicMock()
        collection.create_index.side_effect = [
            "created_at_desc_id_desc",
            "analysis_type_created_at_desc_id_desc",
            "risk_level_created_at_desc_id_desc",
            OperationFailure("simulated index creation failure", code=85),
        ]
        client.__getitem__.return_value.__getitem__.return_value = collection
        with (
            patch.dict("os.environ", {"MONGODB_URI": "mongodb://127.0.0.1:27017/"}),
            patch("database.MongoClient", return_value=client),
            self.assertLogs("database", level="ERROR") as captured,
        ):
            with self.assertRaises(database.DatabaseUnavailable):
                database._database_collection()

        self.assertIn(database.ANALYSIS_RETENTION_INDEX, captured.output[0])
        self.assertIn("phishlens.analysis_history", captured.output[0])
        self.assertNotIn("mongodb://", captured.output[0])

    def test_connection_reports_unconfigured_database(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(
                database.DatabaseUnavailable,
                "MONGODB_URI is not configured",
            ):
                database._database_collection()

    def test_history_filters_are_escaped_and_paged_in_database(self):
        collection = MagicMock()
        collection.count_documents.return_value = 1
        cursor = MagicMock()
        cursor.sort.return_value = cursor
        cursor.skip.return_value = cursor
        cursor.limit.return_value = cursor
        cursor.__iter__.return_value = iter([{
            "_id": "record-1",
            "analysis_type": "email",
            "target": "Example subject",
            "classification": "Spam",
            "risk_level": "high",
            "risk_score": 70,
            "warning_reasons": ["Model classified as Spam"],
            "indicators": [],
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        }])
        collection.find.return_value = cursor

        with patch("database._database_collection", return_value=collection):
            records, total = database.list_history(
                limit=10,
                offset=5,
                analysis_type="email",
                risk_level="high",
                search="a.*",
            )

        query = collection.count_documents.call_args.args[0]
        self.assertEqual(query["analysis_type"], "email")
        self.assertEqual(query["risk_level"], "high")
        self.assertEqual(query["$or"][0]["target"]["$regex"], r"a\.\*")
        self.assertEqual(total, 1)
        self.assertEqual(records[0]["id"], "record-1")
        cursor.skip.assert_called_once_with(5)
        cursor.limit.assert_called_once_with(10)

    def test_saved_records_exclude_message_body_and_sender(self):
        collection = Mock()
        with patch("database._database_collection", return_value=collection):
            database.save_analysis({
                "analysis_type": "email",
                "target": "Example subject",
                "classification": "Spam",
                "risk_level": "high",
                "risk_score": 70,
                "warning_reasons": ["Model classified as Spam"],
                "indicators": [],
                "body": "Do not persist this message",
                "sender": "sender@example.com",
            })
        saved = collection.insert_one.call_args.args[0]
        self.assertNotIn("body", saved)
        self.assertNotIn("sender", saved)
        self.assertIsInstance(saved["created_at"], datetime)
        self.assertEqual(saved["created_at"].tzinfo, timezone.utc)

    def test_saved_url_targets_omit_query_and_fragment(self):
        collection = Mock()
        with patch("database._database_collection", return_value=collection):
            database.save_analysis({
                "analysis_type": "url",
                "target": "https://example.com/account?token=private&email=user@example.com#session",
                "classification": "Review required",
                "risk_level": "medium",
                "risk_score": 40,
                "warning_reasons": [],
                "indicators": [],
            })

        saved = collection.insert_one.call_args.args[0]
        self.assertEqual(saved["target"], "https://example.com/account")

    def test_history_redacts_legacy_url_targets(self):
        record = database._serialize_record({
            "_id": "record-1",
            "analysis_type": "url",
            "target": "https://example.com/path?access_token=private#session",
            "classification": "Review required",
            "risk_level": "medium",
            "risk_score": 40,
            "warning_reasons": [],
            "indicators": [],
        })

        self.assertEqual(record["target"], "https://example.com/path")

    def test_dashboard_statistics_are_derived_from_database_results(self):
        collection = Mock()
        today = datetime.now(timezone.utc).date().isoformat()
        collection.aggregate.side_effect = [
            [{
                "total_analyses": 4,
                "email_analyses": 2,
                "url_analyses": 2,
                "spam_emails": 1,
                "legitimate_emails": 1,
                "suspicious_urls": 1,
                "low_risk": 1,
                "medium_risk": 1,
                "high_risk": 2,
            }],
            [{"_id": today, "analyses": 4, "flagged_analyses": 3}],
        ]
        with patch("database._database_collection", return_value=collection):
            stats = database.get_dashboard_stats()
        self.assertEqual(stats["total_analyses"], 4)
        self.assertEqual(stats["suspicious_urls"], 1)
        self.assertEqual(stats["flagged_analyses"], 3)
        self.assertEqual(stats["flagged_percentage"], 75.0)
        self.assertEqual(stats["risk_distribution"]["high"]["percentage"], 50.0)
        self.assertEqual(len(stats["activity_last_7_days"]), 7)
        self.assertEqual(stats["activity_last_7_days"][-1]["analyses"], 4)
        self.assertEqual(stats["activity_last_7_days"][-1]["flagged_analyses"], 3)

    def test_report_statistics_include_classifications_and_risk_metrics(self):
        dashboard_stats = {
            "total_analyses": 5,
            "email_analyses": 3,
            "url_analyses": 2,
            "spam_emails": 1,
            "legitimate_emails": 2,
            "suspicious_urls": 1,
            "flagged_analyses": 2,
            "flagged_percentage": 40.0,
            "risk_distribution": {},
            "activity_last_7_days": [],
        }
        with patch("database.get_dashboard_stats", return_value=dashboard_stats):
            reports = database.get_report_statistics()
        self.assertEqual(reports["legitimate_emails"], 2)
        self.assertEqual(reports["suspicious_urls"], 1)
        self.assertEqual(reports["flagged_percentage"], 40.0)


if __name__ == "__main__":
    unittest.main()
