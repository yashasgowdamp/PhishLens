import logging
import os
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit, urlunsplit

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import PyMongoError

ANALYSIS_RETENTION_DAYS = 90
ANALYSIS_RETENTION_SECONDS = ANALYSIS_RETENTION_DAYS * 24 * 60 * 60
ANALYSIS_RETENTION_INDEX = "analysis_retention_ttl"
_MONGODB_URI_PATTERN = re.compile(r"mongodb(?:\+srv)?://[^\s\"'<>]+", re.IGNORECASE)
_AUTH_CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/-]+=*"
)
_SENSITIVE_VALUE_PATTERN = re.compile(
    r"(?i)\b(username|user|password|passwd|pwd|token|access[_ -]?token|"
    r"api[_ -]?key|secret|authorization)\b([=:])\s*([^\s,;]+)"
)

logger = logging.getLogger(__name__)
_client = None
_client_uri = None
_initialized_collections = set()


class DatabaseUnavailable(RuntimeError):
    """Raised when shared MongoDB persistence cannot be used."""


def _safe_database_error_detail(error):
    detail = str(error)
    detail = _MONGODB_URI_PATTERN.sub("[MongoDB URI redacted]", detail)
    detail = _AUTH_CREDENTIAL_PATTERN.sub("[credential redacted]", detail)
    return _SENSITIVE_VALUE_PATTERN.sub(r"\1\2[REDACTED]", detail)


def _database_collection():
    global _client, _client_uri, _initialized_collections

    uri = os.environ.get("MONGODB_URI", "").strip()
    if not uri:
        raise DatabaseUnavailable(
            "MONGODB_URI is not configured; analyses are not saved and shared history is unavailable."
        )

    database_name = os.environ.get("MONGODB_DATABASE", "phishlens").strip()
    collection_name = os.environ.get(
        "MONGODB_COLLECTION",
        "analysis_history",
    ).strip()
    if not database_name or not collection_name:
        raise DatabaseUnavailable(
            "MONGODB_DATABASE and MONGODB_COLLECTION must not be empty."
        )

    try:
        if _client is None or uri != _client_uri:
            new_client = MongoClient(
                uri,
                serverSelectionTimeoutMS=15_000,
                connectTimeoutMS=10_000,
            )
            if _client is not None:
                _client.close()
                _initialized_collections.clear()
            _client = new_client
            _client_uri = uri
        _client.admin.command("ping")
        collection = _client[database_name][collection_name]
        collection_key = (database_name, collection_name)
        if collection_key not in _initialized_collections:
            _create_index(
                collection,
                [("created_at", DESCENDING), ("_id", DESCENDING)],
                database_name,
                collection_name,
                name="created_at_desc_id_desc",
            )
            _create_index(
                collection,
                [
                    ("analysis_type", ASCENDING),
                    ("created_at", DESCENDING),
                    ("_id", DESCENDING),
                ],
                database_name,
                collection_name,
                name="analysis_type_created_at_desc_id_desc",
            )
            _create_index(
                collection,
                [
                    ("risk_level", ASCENDING),
                    ("created_at", DESCENDING),
                    ("_id", DESCENDING),
                ],
                database_name,
                collection_name,
                name="risk_level_created_at_desc_id_desc",
            )
            _create_index(
                collection,
                [("created_at", ASCENDING)],
                database_name,
                collection_name,
                name=ANALYSIS_RETENTION_INDEX,
                expireAfterSeconds=ANALYSIS_RETENTION_SECONDS,
            )
            logger.info(
                "Ensured MongoDB TTL index %s on %s.%s "
                "(created_at ascending, expireAfterSeconds=%s).",
                ANALYSIS_RETENTION_INDEX,
                database_name,
                collection_name,
                ANALYSIS_RETENTION_SECONDS,
            )
            _redact_legacy_url_targets(collection)
            _initialized_collections.add(collection_key)
        return collection
    except PyMongoError as error:
        logger.error(
            "MongoDB connection failed (%s): %s",
            type(error).__name__,
            _safe_database_error_detail(error),
        )
        raise DatabaseUnavailable(
            "MongoDB is unavailable. Check MONGODB_URI and ensure the database server is running."
        ) from error


def _create_index(collection, keys, database_name, collection_name, **options):
    index_name = options["name"]
    try:
        return collection.create_index(keys, **options)
    except PyMongoError:
        logger.exception(
            "MongoDB index creation failed for %s on %s.%s.",
            index_name,
            database_name,
            collection_name,
        )
        raise


def check_connection():
    _database_collection()


def save_analysis(record):
    target = record["target"]
    if record["analysis_type"] == "url":
        target = _redact_url_target(target)

    document = {
        "analysis_type": record["analysis_type"],
        "target": target,
        "classification": record["classification"],
        "risk_level": record["risk_level"],
        "risk_score": record["risk_score"],
        "warning_reasons": record["warning_reasons"],
        "indicators": record.get("indicators", []),
        "created_at": datetime.now(timezone.utc),
    }
    try:
        _database_collection().insert_one(document)
    except PyMongoError as error:
        raise DatabaseUnavailable(
            "MongoDB is unavailable. This analysis was not saved to shared history."
        ) from error


def list_history(limit=50, offset=0, analysis_type="all", risk_level="all", search=""):
    filters = {}
    if analysis_type != "all":
        filters["analysis_type"] = analysis_type
    if risk_level != "all":
        filters["risk_level"] = risk_level
    if search:
        escaped_search = re.escape(search)
        filters["$or"] = [
            {"target": {"$regex": escaped_search, "$options": "i"}},
            {"classification": {"$regex": escaped_search, "$options": "i"}},
            {"analysis_type": {"$regex": escaped_search, "$options": "i"}},
        ]

    try:
        collection = _database_collection()
        total = collection.count_documents(filters)
        documents = collection.find(filters).sort(
            [("created_at", -1), ("_id", -1)]
        ).skip(offset).limit(limit)
        records = [_serialize_record(document) for document in documents]
        return records, total
    except PyMongoError as error:
        raise DatabaseUnavailable(
            "MongoDB is unavailable. Shared detection history cannot be loaded."
        ) from error


def get_dashboard_stats():
    try:
        collection = _database_collection()
        aggregate = list(collection.aggregate([
            {
                "$group": {
                    "_id": None,
                    "total_analyses": {"$sum": 1},
                    "email_analyses": {
                        "$sum": {"$cond": [{"$eq": ["$analysis_type", "email"]}, 1, 0]}
                    },
                    "url_analyses": {
                        "$sum": {"$cond": [{"$eq": ["$analysis_type", "url"]}, 1, 0]}
                    },
                    "spam_emails": {
                        "$sum": {
                            "$cond": [
                                {
                                    "$and": [
                                        {"$eq": ["$analysis_type", "email"]},
                                        {"$eq": ["$classification", "Spam"]},
                                    ]
                                },
                                1,
                                0,
                            ]
                        }
                    },
                    "legitimate_emails": {
                        "$sum": {
                            "$cond": [
                                {
                                    "$and": [
                                        {"$eq": ["$analysis_type", "email"]},
                                        {"$eq": ["$classification", "Legitimate"]},
                                    ]
                                },
                                1,
                                0,
                            ]
                        }
                    },
                    "suspicious_urls": {
                        "$sum": {
                            "$cond": [
                                {
                                    "$and": [
                                        {"$eq": ["$analysis_type", "url"]},
                                        {"$in": ["$risk_level", ["medium", "high"]]},
                                    ]
                                },
                                1,
                                0,
                            ]
                        }
                    },
                    "low_risk": {
                        "$sum": {"$cond": [{"$eq": ["$risk_level", "low"]}, 1, 0]}
                    },
                    "medium_risk": {
                        "$sum": {"$cond": [{"$eq": ["$risk_level", "medium"]}, 1, 0]}
                    },
                    "high_risk": {
                        "$sum": {"$cond": [{"$eq": ["$risk_level", "high"]}, 1, 0]}
                    },
                }
            }
        ]))
        counts = aggregate[0] if aggregate else {
            "total_analyses": 0,
            "email_analyses": 0,
            "url_analyses": 0,
            "spam_emails": 0,
            "legitimate_emails": 0,
            "suspicious_urls": 0,
            "low_risk": 0,
            "medium_risk": 0,
            "high_risk": 0,
        }

        start = datetime.now(timezone.utc).date() - timedelta(days=6)
        start_timestamp = datetime.combine(start, datetime.min.time(), timezone.utc)
        activity_rows = collection.aggregate([
            {"$match": {"created_at": {"$gte": start_timestamp}}},
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$created_at",
                            "timezone": "UTC",
                        }
                    },
                    "analyses": {"$sum": 1},
                    "flagged_analyses": {
                        "$sum": {
                            "$cond": [
                                {"$in": ["$risk_level", ["medium", "high"]]},
                                1,
                                0,
                            ]
                        }
                    },
                }
            },
        ])
        activity_by_day = {row["_id"]: row for row in activity_rows}
    except PyMongoError as error:
        raise DatabaseUnavailable(
            "MongoDB is unavailable. Shared dashboard statistics cannot be loaded."
        ) from error

    total = int(counts["total_analyses"])
    distribution = {
        "low": int(counts["low_risk"]),
        "medium": int(counts["medium_risk"]),
        "high": int(counts["high_risk"]),
    }
    flagged_analyses = distribution["medium"] + distribution["high"]
    return {
        "total_analyses": total,
        "email_analyses": int(counts["email_analyses"]),
        "url_analyses": int(counts["url_analyses"]),
        "spam_emails": int(counts["spam_emails"]),
        "legitimate_emails": int(counts["legitimate_emails"]),
        "suspicious_urls": int(counts["suspicious_urls"]),
        "flagged_analyses": flagged_analyses,
        "flagged_percentage": round(flagged_analyses / total * 100, 1) if total else 0.0,
        "high_risk_analyses": distribution["high"],
        "risk_distribution": {
            level: {
                "count": count,
                "percentage": round(count / total * 100, 1) if total else 0.0,
            }
            for level, count in distribution.items()
        },
        "activity_last_7_days": [
            {
                "date": (start + timedelta(days=day)).isoformat(),
                "analyses": int(activity_by_day.get(
                    (start + timedelta(days=day)).isoformat(),
                    {},
                ).get("analyses", 0)),
                "flagged_analyses": int(activity_by_day.get(
                    (start + timedelta(days=day)).isoformat(),
                    {},
                ).get("flagged_analyses", 0)),
            }
            for day in range(7)
        ],
    }


def get_report_statistics():
    statistics = get_dashboard_stats()
    return {
        "total_analyses": statistics["total_analyses"],
        "email_analyses": statistics["email_analyses"],
        "url_analyses": statistics["url_analyses"],
        "spam_emails": statistics["spam_emails"],
        "legitimate_emails": statistics["legitimate_emails"],
        "suspicious_urls": statistics["suspicious_urls"],
        "flagged_analyses": statistics["flagged_analyses"],
        "flagged_percentage": statistics["flagged_percentage"],
        "risk_distribution": statistics["risk_distribution"],
        "activity_last_7_days": statistics["activity_last_7_days"],
        "risk_score_notice": (
            "Risk scores are indicative assessments, not probabilities or guarantees."
        ),
    }


def _serialize_record(document):
    created_at = document.get("created_at")
    if isinstance(created_at, datetime):
        created_at = created_at.isoformat()
    analysis_type = document.get("analysis_type", "")
    target = document.get("target", "")
    if analysis_type == "url":
        target = _redact_url_target(target)
    return {
        "id": str(document["_id"]),
        "analysis_type": analysis_type,
        "target": target,
        "classification": document.get("classification", ""),
        "risk_level": document.get("risk_level", ""),
        "risk_score": document.get("risk_score", 0),
        "warning_reasons": document.get("warning_reasons", []),
        "indicators": document.get("indicators", []),
        "created_at": created_at,
    }


def _redact_url_target(target):
    parsed = urlsplit(target)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _redact_legacy_url_targets(collection):
    legacy_urls = collection.find(
        {
            "analysis_type": "url",
            "target": {"$regex": r"[?#]"},
        },
        {"_id": 1, "target": 1},
    )
    for document in legacy_urls:
        target = document.get("target", "")
        redacted_target = _redact_url_target(target)
        if redacted_target != target:
            collection.update_one(
                {"_id": document["_id"]},
                {"$set": {"target": redacted_target}},
            )
