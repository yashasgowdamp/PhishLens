import os
from functools import lru_cache

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


class EmailModelUnavailable(RuntimeError):
    """Raised when a trained email model has not been installed."""


@lru_cache(maxsize=2)
def _load_model(classifier_path, vectorizer_path):
    if not os.path.isfile(classifier_path) or not os.path.isfile(vectorizer_path):
        raise EmailModelUnavailable(
            "Email classification is unavailable because the trained classifier or "
            "TF-IDF vectorizer was not found. Prepare a genuine labeled email dataset "
            "and run prepare_dataset.py followed by train_model.py."
        )
    return joblib.load(classifier_path), joblib.load(vectorizer_path)


def get_model_paths():
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
