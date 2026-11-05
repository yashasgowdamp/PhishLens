import atexit
import base64
import binascii
import hashlib
import hmac
import os
import string
from functools import lru_cache
import pickle
import shutil
import tempfile

import joblib

from preprocessing import clean_email_text

DEFAULT_CLASSIFIER_PATH = os.path.join(
    os.path.dirname(__file__),
    "classifier.joblib",
)
DEFAULT_VECTORIZER_PATH = os.path.join(
    os.path.dirname(__file__),
    "vectorizer.joblib",
)
MAX_MODEL_SECRET_FILE_BYTES = 16 * 1024 * 1024
SHA256_HEX_LENGTH = hashlib.sha256().digest_size * 2


class EmailModelUnavailable(RuntimeError):
    """Raised when a trained email model has not been installed."""


@lru_cache(maxsize=1)
def _decode_secret_model_files(
    classifier_secret_path,
    vectorizer_secret_path,
    classifier_expected_sha256,
    vectorizer_expected_sha256,
):
    secret_files = (
        ("classifier", classifier_secret_path, classifier_expected_sha256),
        ("vectorizer", vectorizer_secret_path, vectorizer_expected_sha256),
    )
    decoded_models = []
    try:
        for model_name, secret_path, expected_sha256 in secret_files:
            if not expected_sha256 or len(expected_sha256) != SHA256_HEX_LENGTH:
                raise EmailModelUnavailable(
                    f"A valid {model_name} model SHA-256 hash must be configured."
                )
            if any(character not in string.hexdigits for character in expected_sha256):
                raise EmailModelUnavailable(
                    f"A valid {model_name} model SHA-256 hash must be configured."
                )

            if os.path.getsize(secret_path) > MAX_MODEL_SECRET_FILE_BYTES:
                raise EmailModelUnavailable(
                    f"The Base64 {model_name} model Secret File exceeds the "
                    f"{MAX_MODEL_SECRET_FILE_BYTES}-byte size limit."
                )
            with open(secret_path, "rb") as secret_file:
                encoded_model = secret_file.read(MAX_MODEL_SECRET_FILE_BYTES + 1)
            if len(encoded_model) > MAX_MODEL_SECRET_FILE_BYTES:
                raise EmailModelUnavailable(
                    f"The Base64 {model_name} model Secret File exceeds the "
                    f"{MAX_MODEL_SECRET_FILE_BYTES}-byte size limit."
                )
            try:
                encoded_model = encoded_model.decode("ascii").strip()
            except UnicodeError as error:
                raise EmailModelUnavailable(
                    f"The Base64 {model_name} model Secret File must contain valid "
                    "ASCII Base64 data."
                ) from error
            if not encoded_model:
                raise EmailModelUnavailable(
                    f"The Base64 {model_name} model Secret File is empty."
                )
            try:
                model_data = base64.b64decode(encoded_model, validate=True)
            except binascii.Error as error:
                raise EmailModelUnavailable(
                    f"The Base64 {model_name} model Secret File could not be decoded. "
                    "Verify that it contains valid Base64 data."
                ) from error
            if not model_data:
                raise EmailModelUnavailable(
                    f"The Base64 {model_name} model Secret File decodes to an empty file."
                )
            actual_sha256 = hashlib.sha256(model_data).hexdigest()
            if not hmac.compare_digest(
                actual_sha256,
                expected_sha256.lower(),
            ):
                raise EmailModelUnavailable(
                    f"The {model_name} model Secret File failed SHA-256 integrity "
                    "verification."
                )
            decoded_models.append(model_data)
    except FileNotFoundError as error:
        raise EmailModelUnavailable(
            "A configured Base64 model Secret File could not be found."
        ) from error
    except (OSError, UnicodeError, binascii.Error) as error:
        raise EmailModelUnavailable(
            "A configured Base64 model Secret File could not be read or decoded. "
            "Verify that it contains valid Base64 data."
        ) from error

    model_directory = tempfile.mkdtemp(prefix="phishlens-models-")
    try:
        if os.name == "posix":
            os.chmod(model_directory, 0o700)

        model_paths = []
        for model_name, model_data in zip(
            ("classifier", "vectorizer"),
            decoded_models,
        ):
            model_path = os.path.join(model_directory, f"{model_name}.joblib")
            descriptor = os.open(
                model_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as model_file:
                model_file.write(model_data)
            model_paths.append(model_path)
        atexit.register(shutil.rmtree, model_directory, ignore_errors=True)
        return tuple(model_paths)
    except OSError as error:
        shutil.rmtree(model_directory, ignore_errors=True)
        raise EmailModelUnavailable(
            "Decoded model Secret Files could not be prepared for loading."
        ) from error


@lru_cache(maxsize=2)
def _load_model(classifier_path, vectorizer_path):
    if not os.path.isfile(classifier_path) or not os.path.isfile(vectorizer_path):
        raise EmailModelUnavailable(
            "Email classification is unavailable because the trained classifier or "
            "TF-IDF vectorizer was not found. Prepare a genuine labeled email dataset "
            "and run prepare_dataset.py followed by train_model.py."
        )
    try:
        return joblib.load(classifier_path), joblib.load(vectorizer_path)
    except (
        OSError,
        EOFError,
        ValueError,
        TypeError,
        ImportError,
        AttributeError,
        KeyError,
        pickle.UnpicklingError,
    ) as error:
        raise EmailModelUnavailable(
            "The trained classifier or TF-IDF vectorizer could not be loaded. "
            "Verify that both files contain valid model artifacts."
        ) from error


def get_model_paths():
    classifier_secret_path = os.environ.get(
        "PHISHLENS_CLASSIFIER_BASE64_FILE",
        "",
    ).strip()
    vectorizer_secret_path = os.environ.get(
        "PHISHLENS_VECTORIZER_BASE64_FILE",
        "",
    ).strip()
    if classifier_secret_path or vectorizer_secret_path:
        if not classifier_secret_path or not vectorizer_secret_path:
            raise EmailModelUnavailable(
                "Both PHISHLENS_CLASSIFIER_BASE64_FILE and "
                "PHISHLENS_VECTORIZER_BASE64_FILE must be configured."
            )
        return _decode_secret_model_files(
            classifier_secret_path,
            vectorizer_secret_path,
            os.environ.get("PHISHLENS_CLASSIFIER_SHA256", "").strip(),
            os.environ.get("PHISHLENS_VECTORIZER_SHA256", "").strip(),
        )

    return (
        os.path.abspath(os.environ.get(
            "PHISHLENS_CLASSIFIER_PATH",
            DEFAULT_CLASSIFIER_PATH,
        )),
        os.path.abspath(os.environ.get(
            "PHISHLENS_VECTORIZER_PATH",
            DEFAULT_VECTORIZER_PATH,
        )),
    )


def analyze_email_text(email_text):
    cleaned_text = clean_email_text(email_text)
    if not cleaned_text:
        raise ValueError("Email text must contain usable words.")

    classifier_path, vectorizer_path = get_model_paths()
    classifier, vectorizer = _load_model(classifier_path, vectorizer_path)
    features = vectorizer.transform([cleaned_text])
    prediction = str(classifier.predict(features)[0]).strip().lower()
    if prediction not in {"spam", "legitimate"}:
        raise ValueError(
            "The trained model returned an unsupported class; expected Spam or Legitimate."
        )

    if not hasattr(classifier, "predict_proba"):
        raise ValueError("The trained email model must support predict_proba.")
    probabilities = classifier.predict_proba(features)[0]
    class_labels = [str(label).strip().lower() for label in classifier.classes_]
    confidence = float(probabilities[class_labels.index(prediction)])

    return {
        "classification": "Spam" if prediction == "spam" else "Legitimate",
        "confidence": confidence,
    }
