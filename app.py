import logging
import os
import re
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

import database
from email_analyzer import EmailModelUnavailable, analyze_email_text
from preprocessing import clean_email_text
from risk_engine import combine_email_risk
from url_analyzer import URLAnalysisError, analyze_url

logger = logging.getLogger(__name__)

MAX_EMAIL_CHARS = 100_000
MAX_SUBJECT_CHARS = 500
MAX_SENDER_CHARS = 254
MAX_URL_CHARS = 2_048
MAX_SEARCH_CHARS = 100


def create_app(test_config=None):
    application = Flask(__name__)
    application.config.update(MAX_CONTENT_LENGTH=1_000_000)
    if test_config:
        application.config.update(test_config)

    @application.after_request
    def add_cors_headers(response):
        allowed_origins = {
            origin.strip()
            for origin in os.environ.get(
                "CORS_ALLOWED_ORIGINS",
                "http://localhost:5500,http://127.0.0.1:5500",
            ).split(",")
            if origin.strip()
        }
        origin = request.headers.get("Origin")
        if origin in allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @application.errorhandler(HTTPException)
    def handle_http_error(error):
        return jsonify({"status": "error", "message": error.description}), error.code

    @application.errorhandler(Exception)
    def handle_unexpected_error(error):
        logger.exception("Unexpected error while handling a PhishLens request.")
        return jsonify({
            "status": "error",
            "message": "The server could not complete the request.",
        }), 500

    @application.get("/")
    def home():
        return jsonify({
            "project": "PhishLens",
            "message": "PhishLens Flask Backend is Running!",
            "status": "success",
        })

    @application.get("/api/health")
    def health_check():
        try:
            database.check_connection()
            database_status = {"available": True, "message": "MongoDB is connected."}
        except database.DatabaseUnavailable as error:
            database_status = {"available": False, "message": str(error)}

        return jsonify({
            "status": "success",
            "message": "PhishLens backend is running.",
            "database": database_status,
        })

    @application.post("/api/preprocess-email")
    def preprocess_email():
        data, error_response = read_json_object()
        if error_response:
            return error_response

        email_text = data.get("email_text")
        if not isinstance(email_text, str) or not email_text.strip():
            return json_error("Please provide valid email text.", 400)
        if len(email_text) > MAX_EMAIL_CHARS:
            return json_error("Email text exceeds the 100,000 character limit.", 413)

        return jsonify({
            "status": "success",
            "original_text": email_text,
            "cleaned_text": clean_email_text(email_text),
            "message": "Email preprocessing completed.",
        })

    @application.post("/api/email/analyze")
    def email_analysis():
        data, error_response = read_json_object()
        if error_response:
            return error_response

        subject = data.get("subject", "")
        body = data.get("body")
        sender = data.get("sender", "")
        optional_url = data.get("url", "")

        if not isinstance(subject, str) or not subject.strip():
            return json_error("Please provide a valid email subject.", 400)
        if len(subject) > MAX_SUBJECT_CHARS:
            return json_error("Email subject exceeds the 500 character limit.", 413)
        if not isinstance(body, str) or not body.strip():
            return json_error("Please provide valid email message text.", 400)
        if len(body) > MAX_EMAIL_CHARS:
            return json_error("Email text exceeds the 100,000 character limit.", 413)
        if not isinstance(sender, str) or len(sender) > MAX_SENDER_CHARS:
            return json_error("Please provide a valid sender address.", 400)
        if sender and not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", sender):
            return json_error("Please provide a valid sender address.", 400)
        if not isinstance(optional_url, str) or len(optional_url) > MAX_URL_CHARS:
            return json_error("Please provide a valid URL (maximum 2,048 characters).", 400)

        try:
            model_result = analyze_email_text(f"{subject}\n{body}")
            url_result = analyze_url(optional_url) if optional_url.strip() else None
        except EmailModelUnavailable as error:
            return json_error(str(error), 503)
        except URLAnalysisError as error:
            return json_error(str(error), 400)

        risk = combine_email_risk(
            model_result["classification"],
            model_result["confidence"],
            url_result,
        )
        record = {
            "analysis_type": "email",
            "target": subject.strip(),
            "classification": model_result["classification"],
            "risk_level": risk["risk_level"],
            "risk_score": risk["risk_score"],
            "warning_reasons": risk["warning_reasons"],
            "indicators": url_result["indicators"] if url_result else [],
            "model_confidence": model_result["confidence"],
        }
        persistence = persist_record(record)
        return jsonify({
            "status": "success",
            **record,
            "risk": risk,
            "persistence": persistence,
            "warning": "The risk score is indicative, not a probability or a guarantee of an attack.",
        })

    @application.post("/api/url/analyze")
    def url_analysis():
        data, error_response = read_json_object()
        if error_response:
            return error_response

        raw_url = data.get("url")
        if not isinstance(raw_url, str) or not raw_url.strip():
            return json_error("Please provide a valid URL.", 400)
        if len(raw_url) > MAX_URL_CHARS:
            return json_error("URL exceeds the 2,048 character limit.", 413)

        try:
            result = analyze_url(raw_url)
        except URLAnalysisError as error:
            return json_error(str(error), 400)

        record = {
            "analysis_type": "url",
            "target": result["url"],
            "classification": result["classification"],
            "risk_level": result["risk_level"],
            "risk_score": result["risk_score"],
            "warning_reasons": result["warning_reasons"],
            "indicators": result["indicators"],
        }
        persistence = persist_record(record)
        return jsonify({
            "status": "success",
            **record,
            "risk": result["risk"],
            "hostname": result["hostname"],
            "persistence": persistence,
            "warning": "URL checks are rule-based and cannot guarantee phishing detection.",
        })

    @application.get("/api/history")
    def detection_history():
        limit_value = request.args.get("limit", "50")
        if not limit_value.isdigit() or not 1 <= int(limit_value) <= 100:
            return json_error("The history limit must be between 1 and 100.", 400)

        analysis_type = request.args.get("type", "all")
        if analysis_type not in {"all", "email", "url"}:
            return json_error("The type filter must be all, email, or url.", 400)

        risk_level = request.args.get("risk", "all")
        if risk_level not in {"all", "low", "medium", "high"}:
            return json_error("The risk filter must be all, low, medium, or high.", 400)

        search = request.args.get("search", "").strip()
        if len(search) > MAX_SEARCH_CHARS:
            return json_error("Search text exceeds the 100 character limit.", 400)
        offset_value = request.args.get("offset", "0")
        if not offset_value.isdigit() or int(offset_value) > 10_000:
            return json_error("The history offset must be between 0 and 10,000.", 400)

        try:
            records, total = database.list_history(
                limit=int(limit_value),
                offset=int(offset_value),
                analysis_type=analysis_type,
                risk_level=risk_level,
                search=search,
            )
        except database.DatabaseUnavailable as error:
            return json_error(str(error), 503)

        return jsonify({"status": "success", "records": records, "total": total})

    @application.get("/api/dashboard/stats")
    def dashboard_statistics():
        try:
            stats = database.get_dashboard_stats()
        except database.DatabaseUnavailable as error:
            return json_error(str(error), 503)
        return jsonify({"status": "success", "statistics": stats})

    @application.get("/api/reports")
    def analysis_reports():
        try:
            reports = database.get_report_statistics()
        except database.DatabaseUnavailable as error:
            return json_error(str(error), 503)
        return jsonify({"status": "success", "reports": reports})

    return application


def read_json_object():
    if not request.is_json:
        return None, json_error("Request Content-Type must be application/json.", 400)

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, json_error("Request body must be a JSON object.", 400)
    return data, None


def json_error(message, status_code):
    return jsonify({"status": "error", "message": message}), status_code


def persist_record(record):
    try:
        database.save_analysis(record)
        return {"saved": True, "message": "Analysis saved to shared history."}
    except database.DatabaseUnavailable as error:
        logger.warning("Analysis was not saved: %s", error)
        return {"saved": False, "message": str(error)}


def initialize_database():
    try:
        database.check_connection()
    except database.DatabaseUnavailable as error:
        logger.warning(
            "MongoDB startup initialization failed; shared history is unavailable: %s",
            error,
        )


app = create_app()


if __name__ == "__main__":
    initialize_database()
    app.run(
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("FLASK_PORT", "5000")),
    )
