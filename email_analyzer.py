import atexit
import base64
import binascii
import hashlib
import hmac
import http.client
import json
import os
import pickle
import re
import shutil
import string
import tempfile
from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

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
MAX_MODEL_DOWNLOAD_BYTES = 16 * 1024 * 1024
MAX_GITHUB_RESPONSE_BYTES = 1024 * 1024
GITHUB_REQUEST_TIMEOUT_SECONDS = 15
SHA256_HEX_LENGTH = hashlib.sha256().digest_size * 2
GITHUB_API_BASE_URL = "https://api.github.com"


class EmailModelUnavailable(RuntimeError):
    """Raised when a trained email model has not been installed."""


class _SafeGitHubRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        if urlsplit(new_url).scheme.lower() != "https":
            raise EmailModelUnavailable(
                "GitHub model downloads only allow HTTPS redirects."
            )

        redirected_request = super().redirect_request(
            request,
            response,
            code,
            message,
            headers,
            new_url,
        )
        if redirected_request is not None:
            for request_headers in (
                redirected_request.headers,
                redirected_request.unredirected_hdrs,
            ):
                for header_name in tuple(request_headers):
                    if header_name.lower() in {
                        "authorization",
                        "proxy-authorization",
                    }:
                        del request_headers[header_name]
        return redirected_request


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


def _validate_expected_sha256(expected_sha256, model_name):
    if not expected_sha256 or len(expected_sha256) != SHA256_HEX_LENGTH:
        raise EmailModelUnavailable(
            f"A valid {model_name} model SHA-256 hash must be configured."
        )
    if any(character not in string.hexdigits for character in expected_sha256):
        raise EmailModelUnavailable(
            f"A valid {model_name} model SHA-256 hash must be configured."
        )


def _verify_model_sha256(model_path, expected_sha256, model_name):
    _validate_expected_sha256(expected_sha256, model_name)
    digest = hashlib.sha256()
    try:
        with open(model_path, "rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise EmailModelUnavailable(
            f"The downloaded {model_name} model could not be read for verification."
        ) from error
    if not hmac.compare_digest(digest.hexdigest(), expected_sha256.lower()):
        raise EmailModelUnavailable(
            f"The {model_name} model failed SHA-256 integrity verification."
        )


def _github_request(url, token, accept):
    request = Request(
        url,
        headers={
            "Accept": accept,
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "PhishLens-model-loader",
        },
    )
    try:
        opener = build_opener(_SafeGitHubRedirectHandler())
        return opener.open(request, timeout=GITHUB_REQUEST_TIMEOUT_SECONDS)
    except HTTPError as error:
        if error.code in {401, 403}:
            raise EmailModelUnavailable(
                "GitHub rejected model-release authentication or access."
            ) from error
        if error.code == 404:
            raise EmailModelUnavailable(
                "The configured GitHub release or required model asset was not found."
            ) from error
        raise EmailModelUnavailable(
            f"GitHub model download failed with HTTP status {error.code}."
        ) from error
    except (TimeoutError, URLError, OSError) as error:
        if isinstance(error, URLError) and isinstance(error.reason, TimeoutError):
            message = "GitHub model download timed out."
        elif isinstance(error, TimeoutError):
            message = "GitHub model download timed out."
        else:
            message = "GitHub model download could not be completed."
        raise EmailModelUnavailable(message) from error


@lru_cache(maxsize=1)
def _download_release_model_files():
    repository = os.environ.get("PHISHLENS_GITHUB_REPOSITORY", "").strip()
    release_tag = os.environ.get("PHISHLENS_GITHUB_RELEASE_TAG", "").strip()
    token = os.environ.get("PHISHLENS_GITHUB_TOKEN", "").strip()
    expected_hashes = {
        "classifier": os.environ.get("PHISHLENS_CLASSIFIER_SHA256", "").strip(),
        "vectorizer": os.environ.get("PHISHLENS_VECTORIZER_SHA256", "").strip(),
    }

    if not repository or not release_tag or not token:
        raise EmailModelUnavailable(
            "GitHub model loading requires PHISHLENS_GITHUB_REPOSITORY, "
            "PHISHLENS_GITHUB_RELEASE_TAG, and PHISHLENS_GITHUB_TOKEN."
        )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise EmailModelUnavailable(
            "PHISHLENS_GITHUB_REPOSITORY must use the owner/repository format."
        )
    for model_name, expected_sha256 in expected_hashes.items():
        _validate_expected_sha256(expected_sha256, model_name)

    release_url = (
        f"{GITHUB_API_BASE_URL}/repos/{repository}/releases/tags/"
        f"{quote(release_tag, safe='')}"
    )
    try:
        with _github_request(
            release_url,
            token,
            "application/vnd.github+json",
        ) as response:
            release_data = response.read(MAX_GITHUB_RESPONSE_BYTES + 1)
        if len(release_data) > MAX_GITHUB_RESPONSE_BYTES:
            raise EmailModelUnavailable(
                "GitHub release metadata exceeded the allowed response size."
            )
        release = json.loads(release_data)
    except EmailModelUnavailable:
        raise
    except TimeoutError as error:
        raise EmailModelUnavailable("GitHub model download timed out.") from error
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        http.client.HTTPException,
    ) as error:
        raise EmailModelUnavailable(
            "GitHub returned invalid or unreadable release metadata."
        ) from error

    if not isinstance(release, dict):
        raise EmailModelUnavailable("GitHub returned invalid release metadata.")
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise EmailModelUnavailable("GitHub release metadata has no valid assets list.")
    required_assets = {}
    for model_name in ("classifier", "vectorizer"):
        asset_name = f"{model_name}.joblib"
        asset = next(
            (
                item for item in assets
                if isinstance(item, dict) and item.get("name") == asset_name
            ),
            None,
        )
        if asset is None:
            raise EmailModelUnavailable(
                f"The GitHub release is missing the required {asset_name} asset."
            )
        asset_id = asset.get("id")
        asset_size = asset.get("size")
        if (
            not isinstance(asset_id, int)
            or isinstance(asset_id, bool)
            or asset_id <= 0
            or not isinstance(asset_size, int)
            or isinstance(asset_size, bool)
            or not 0 < asset_size <= MAX_MODEL_DOWNLOAD_BYTES
        ):
            raise EmailModelUnavailable(
                f"The GitHub {asset_name} asset has invalid metadata or exceeds "
                f"the {MAX_MODEL_DOWNLOAD_BYTES}-byte download limit."
            )
        required_assets[model_name] = asset

    model_directory = tempfile.mkdtemp(prefix="phishlens-github-models-")
    try:
        if os.name == "posix":
            os.chmod(model_directory, 0o700)

        model_paths = []
        for model_name in ("classifier", "vectorizer"):
            asset = required_assets[model_name]
            asset_url = (
                f"{GITHUB_API_BASE_URL}/repos/{repository}/releases/assets/"
                f"{asset['id']}"
            )
            model_path = os.path.join(model_directory, f"{model_name}.joblib")
            try:
                with _github_request(
                    asset_url,
                    token,
                    "application/octet-stream",
                ) as response:
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None:
                        try:
                            content_length = int(content_length)
                        except ValueError as error:
                            raise EmailModelUnavailable(
                                f"The GitHub {model_name} model asset has invalid "
                                "download metadata."
                            ) from error
                        if not 0 < content_length <= MAX_MODEL_DOWNLOAD_BYTES:
                            raise EmailModelUnavailable(
                                f"The GitHub {model_name} model asset exceeds the "
                                f"{MAX_MODEL_DOWNLOAD_BYTES}-byte download limit."
                            )

                    descriptor = os.open(
                        model_path,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                    )
                    downloaded_size = 0
                    with os.fdopen(descriptor, "wb") as model_file:
                        while True:
                            chunk = response.read(
                                min(1024 * 1024, MAX_MODEL_DOWNLOAD_BYTES + 1 - downloaded_size)
                            )
                            if not chunk:
                                break
                            downloaded_size += len(chunk)
                            if downloaded_size > MAX_MODEL_DOWNLOAD_BYTES:
                                raise EmailModelUnavailable(
                                    f"The GitHub {model_name} model asset exceeds the "
                                    f"{MAX_MODEL_DOWNLOAD_BYTES}-byte download limit."
                                )
                            model_file.write(chunk)
                    if downloaded_size == 0:
                        raise EmailModelUnavailable(
                            f"The GitHub {model_name} model asset is empty."
                        )
            except EmailModelUnavailable:
                raise
            except (
                OSError,
                URLError,
                TimeoutError,
                http.client.HTTPException,
            ) as error:
                if isinstance(error, TimeoutError) or (
                    isinstance(error, URLError)
                    and isinstance(error.reason, TimeoutError)
                ):
                    raise EmailModelUnavailable(
                        "GitHub model download timed out."
                    ) from error
                raise EmailModelUnavailable(
                    f"The GitHub {model_name} model asset could not be downloaded."
                ) from error

            _verify_model_sha256(
                model_path,
                expected_hashes[model_name],
                model_name,
            )
            model_paths.append(model_path)

        atexit.register(shutil.rmtree, model_directory, ignore_errors=True)
        return tuple(model_paths)
    except OSError as error:
        shutil.rmtree(model_directory, ignore_errors=True)
        raise EmailModelUnavailable(
            "Downloaded GitHub model files could not be prepared for loading."
        ) from error
    except EmailModelUnavailable:
        shutil.rmtree(model_directory, ignore_errors=True)
        raise


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
    github_configuration = (
        os.environ.get("PHISHLENS_GITHUB_REPOSITORY", "").strip(),
        os.environ.get("PHISHLENS_GITHUB_RELEASE_TAG", "").strip(),
        os.environ.get("PHISHLENS_GITHUB_TOKEN", "").strip(),
    )
    if any(github_configuration):
        return _download_release_model_files()

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
