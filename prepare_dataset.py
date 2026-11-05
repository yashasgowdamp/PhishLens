import csv
import html
import hashlib
import io
import json
import os
import re
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from tarfile import TarError, open as open_tar

DEFAULT_OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    ".local_data",
    "emails.csv",
)
DEFAULT_REPORT_PATH = os.path.join(
    os.path.dirname(__file__),
    ".local_data",
    "dataset_report.json",
)
CORPUS_BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/"
CORPUS_ARCHIVES = (
    (
        "20030228_spam.tar.bz2",
        "spam",
        500,
        "c08debc32413804949a866be45ef78195cec2cbafd1da744ed76cdb860589743",
    ),
    (
        "20050311_spam_2.tar.bz2",
        "spam",
        1396,
        "44280a0e28bf7645b2279e8e42659271ad82c6f40bf8d364dbda7f1df344a765",
    ),
    (
        "20030228_easy_ham.tar.bz2",
        "legitimate",
        2500,
        "2b7b65904bcfcc31d2b5f51946f2d261370b257402cbbd62930b46ab83367438",
    ),
    (
        "20030228_hard_ham.tar.bz2",
        "legitimate",
        250,
        "ce2ce67880643dbde65ea7f85bffbfe4417349c4bd80b6b0de56262ae6b0a9c9",
    ),
    (
        "20030228_easy_ham_2.tar.bz2",
        "legitimate",
        1400,
        "b4bd3dc5ae5b40f38e99a0e41ad7d16b428b56e818e3a6ffa07330d62004dd38",
    ),
)
SOURCE_README_URL = (
    "https://spamassassin.apache.org/old/publiccorpus/readme.html"
)


class DatasetPreparationError(RuntimeError):
    """Raised when the official corpus cannot be converted safely."""


def _download_archive(archive_name):
    request = urllib.request.Request(
        CORPUS_BASE_URL + archive_name,
        headers={"User-Agent": "PhishLens academic dataset preparation"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise DatasetPreparationError(
                    f"Corpus download failed for {archive_name}: HTTP {response.status}."
                )
            return response.read()
    except (OSError, TimeoutError) as error:
        raise DatasetPreparationError(
            f"Could not download official corpus archive {archive_name}: {error}"
        ) from error


def _decode_part(part):
    try:
        content = part.get_content()
        if isinstance(content, str):
            return content
    except (LookupError, UnicodeError, TypeError):
        pass

    payload = part.get_payload(decode=True)
    if not payload:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _message_text(raw_message):
    message = BytesParser(policy=policy.default).parsebytes(raw_message)
    parts = message.walk() if message.is_multipart() else (message,)
    parts = [
        part for part in parts
        if part.get_content_maintype() != "multipart"
        and part.get_content_disposition() != "attachment"
    ]
    body_parts = []
    for part in parts:
        if part.get_content_type() != "text/plain":
            continue
        text = _decode_part(part).strip()
        if text:
            body_parts.append(text)

    if not body_parts:
        for part in parts:
            if part.get_content_type() != "text/html":
                continue
            text = html.unescape(_decode_part(part))
            text = re.sub(r"<[^>]*>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                body_parts.append(text)

    body = "\n".join(body_parts)
    subject = str(message.get("Subject", "")).strip()
    if subject and body:
        return f"Subject: {subject}\n\n{body}"
    return subject or body


def _normalize_exact_text(text):
    return re.sub(r"\s+", " ", text).strip()


def _read_archive(archive_name, label, expected_message_files, expected_sha256):
    archive_data = _download_archive(archive_name)
    archive_hash = hashlib.sha256(archive_data).hexdigest()
    if archive_hash != expected_sha256:
        raise DatasetPreparationError(
            f"SHA-256 mismatch for {archive_name}; refusing an unverified archive."
        )
    messages = []
    message_files = 0
    skipped_files = 0
    try:
        with open_tar(fileobj=io.BytesIO(archive_data), mode="r:bz2") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                if not re.fullmatch(
                    r"\d+\.[0-9a-fA-F]{32}",
                    member.name.rsplit("/", 1)[-1],
                ):
                    skipped_files += 1
                    continue
                message_files += 1
                message_file = archive.extractfile(member)
                if message_file is None:
                    continue
                raw_message = message_file.read()
                text = _message_text(raw_message)
                if text.strip():
                    messages.append({
                        "email_text": text,
                        "label": label,
                        "source_archive": archive_name,
                    })
    except (TarError, OSError) as error:
        raise DatasetPreparationError(
            f"Could not parse official corpus archive {archive_name}: {error}"
        ) from error
    if message_files != expected_message_files:
        raise DatasetPreparationError(
            f"Expected {expected_message_files} email files in {archive_name}, "
            f"found {message_files}."
        )

    return messages, {
        "name": archive_name,
        "url": CORPUS_BASE_URL + archive_name,
        "sha256": archive_hash,
        "message_files": message_files,
        "non_message_files_skipped": skipped_files,
        "labeled_messages_parsed": len(messages),
        "messages_without_extractable_text": message_files - len(messages),
        "label": label,
    }


def prepare_dataset(output_path=None, report_path=None):
    output_path = output_path or os.environ.get(
        "PHISHLENS_DATASET_PATH",
        DEFAULT_OUTPUT_PATH,
    )
    report_path = report_path or DEFAULT_REPORT_PATH
    tracked_placeholder = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "emails.csv")
    )
    if os.path.realpath(os.path.abspath(output_path)) == tracked_placeholder:
        raise DatasetPreparationError(
            "Refusing to overwrite the tracked emails.csv placeholder. "
            "Choose a local-only output path such as .local_data/emails.csv."
        )

    messages = []
    archives = []
    for archive_name, label, expected_count, expected_hash in CORPUS_ARCHIVES:
        parsed_messages, archive_metadata = _read_archive(
            archive_name,
            label,
            expected_count,
            expected_hash,
        )
        messages.extend(parsed_messages)
        archives.append(archive_metadata)

    label_by_text = defaultdict(set)
    for row in messages:
        normalized = _normalize_exact_text(row["email_text"])
        if normalized:
            label_by_text[normalized].add(row["label"])

    conflicting_texts = {
        text for text, labels in label_by_text.items() if len(labels) > 1
    }
    conflicting_records_removed = sum(
        1 for row in messages
        if _normalize_exact_text(row["email_text"]) in conflicting_texts
    )

    deduplicated = []
    seen_texts = set()
    duplicate_records_removed = 0
    for row in messages:
        normalized = _normalize_exact_text(row["email_text"])
        if not normalized or normalized in conflicting_texts:
            continue
        if normalized in seen_texts:
            duplicate_records_removed += 1
            continue
        seen_texts.add(normalized)
        deduplicated.append(row)

    if not deduplicated:
        raise DatasetPreparationError(
            "The official corpus yielded no unambiguous, non-duplicate email text."
        )

    output_directory = os.path.dirname(os.path.abspath(output_path))
    report_directory = os.path.dirname(os.path.abspath(report_path))
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(report_directory, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as dataset_file:
        writer = csv.DictWriter(dataset_file, fieldnames=("email_text", "label"))
        writer.writeheader()
        writer.writerows(
            {"email_text": row["email_text"], "label": row["label"]}
            for row in deduplicated
        )

    labels = Counter(row["label"] for row in deduplicated)
    report = {
        "dataset": "Apache SpamAssassin Public Mail Corpus",
        "source_readme": SOURCE_README_URL,
        "license": (
            "The source README does not specify an open dataset license. "
            "It states that copyright for message text remains with original senders."
        ),
        "use_restriction": (
            "Local academic research/training only by user approval; do not redistribute "
            "the downloaded messages or normalized email CSV."
        ),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_messages_with_nonempty_text": len(messages),
        "source_readme_claimed_total": 6047,
        "selected_archives_message_files": sum(
            archive["message_files"] for archive in archives
        ),
        "duplicate_records_removed": duplicate_records_removed,
        "conflicting_duplicate_texts_removed": len(conflicting_texts),
        "conflicting_records_removed": conflicting_records_removed,
        "unique_unambiguous_records": len(deduplicated),
        "class_counts": {
            "legitimate": labels.get("legitimate", 0),
            "spam": labels.get("spam", 0),
        },
        "output_columns": ["email_text", "label"],
        "archives": archives,
    }
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
        report_file.write("\n")

    return report


if __name__ == "__main__":
    print(json.dumps(prepare_dataset(), indent=2))
