import json
import math
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from preprocessing import clean_email_text

DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    ".local_data",
    "emails.csv",
)
DEFAULT_CLASSIFIER_PATH = os.path.join(
    os.path.dirname(__file__),
    "classifier.joblib",
)
DEFAULT_VECTORIZER_PATH = os.path.join(
    os.path.dirname(__file__),
    "vectorizer.joblib",
)
DEFAULT_EVALUATION_PATH = os.path.join(
    os.path.dirname(__file__),
    ".local_data",
    "training_report.json",
)
CLASS_LABELS = ("legitimate", "spam")


class DatasetError(ValueError):
    """Raised when a labeled email dataset cannot safely train the classifier."""


def load_labeled_dataset(dataset_path):
    try:
        dataset = pd.read_csv(dataset_path)
    except FileNotFoundError as error:
        raise DatasetError(f"Labeled email dataset not found: {dataset_path}") from error
    except pd.errors.EmptyDataError as error:
        raise DatasetError("The labeled email dataset contains no records.") from error
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise DatasetError(f"Could not read the labeled email dataset: {error}") from error

    text_column = "email_text" if "email_text" in dataset.columns else "text"
    if text_column not in dataset.columns or "label" not in dataset.columns:
        raise DatasetError(
            "Dataset must be a CSV with 'email_text' and 'label' columns "
            "(the legacy 'text' column is also accepted); "
            "labels must clearly identify spam or legitimate messages."
        )
    if dataset.empty:
        raise DatasetError("The labeled email dataset contains no records.")

    dataset = dataset.loc[:, [text_column, "label"]].rename(
        columns={text_column: "email_text"}
    )
    input_records = len(dataset)
    dataset["email_text"] = dataset["email_text"].where(
        dataset["email_text"].notna(),
        "",
    ).astype(str)
    dataset["label"] = dataset["label"].where(dataset["label"].notna(), "").astype(str)
    dataset["_normalized_text"] = dataset["email_text"].map(
        lambda value: " ".join(value.split())
    )
    dataset["_clean_text"] = dataset["_normalized_text"].map(clean_email_text)
    dataset["_label"] = dataset["label"].str.strip().str.lower().map({
        "ham": "legitimate",
        "legit": "legitimate",
        "legitimate": "legitimate",
        "spam": "spam",
    })

    invalid_labels = sorted(dataset.loc[dataset["_label"].isna(), "label"].unique())
    if invalid_labels:
        raise DatasetError(
            "Unrecognized labels found. Use explicit spam/legitimate labels "
            f"(ham is accepted as legitimate); invalid values: {invalid_labels[:10]}."
        )

    nonempty_text_records = dataset.loc[
        (dataset["_normalized_text"] != "") & (dataset["_clean_text"] != "")
    ].copy()
    empty_records_removed = len(dataset) - len(nonempty_text_records)
    dataset = nonempty_text_records
    if dataset.empty:
        raise DatasetError("No non-empty email messages remain after preprocessing.")

    conflicting_text = dataset.groupby("_normalized_text")["_label"].nunique()
    if (conflicting_text > 1).any():
        conflicting_texts = set(conflicting_text[conflicting_text > 1].index)
        conflicting_record_count = int(
            dataset["_normalized_text"].isin(conflicting_texts).sum()
        )
        dataset = dataset.loc[
            ~dataset["_normalized_text"].isin(conflicting_texts)
        ].copy()
    else:
        conflicting_texts = set()
        conflicting_record_count = 0

    if dataset.empty:
        raise DatasetError(
            "No records remain after removing duplicate text with conflicting labels."
        )

    records_before_duplicate_removal = len(dataset)
    dataset = dataset.drop_duplicates(subset="_normalized_text", keep="first")
    duplicate_records_removed = records_before_duplicate_removal - len(dataset)

    counts = dataset["_label"].value_counts()
    missing_classes = sorted(set(CLASS_LABELS) - set(counts.index))
    if missing_classes:
        raise DatasetError(
            "Training requires examples from both classes; missing: "
            f"{', '.join(missing_classes)}."
        )
    if any(counts[label] < 2 for label in CLASS_LABELS):
        raise DatasetError(
            "At least two unique labeled messages per class are required "
            "for a stratified train/test split."
        )

    return dataset, {
        "input_records": input_records,
        "empty_records_removed": empty_records_removed,
        "duplicate_records_removed": duplicate_records_removed,
        "conflicting_duplicate_texts_removed": len(conflicting_texts),
        "conflicting_records_removed": conflicting_record_count,
        "unique_records": len(dataset),
        "class_counts": {
            label: int(counts.get(label, 0))
            for label in CLASS_LABELS
        },
    }

def run_training(
    dataset_path=None,
    classifier_path=None,
    vectorizer_path=None,
    test_size=0.2,
    random_state=42,
):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    dataset_path = dataset_path or os.environ.get(
        "PHISHLENS_DATASET_PATH",
        DEFAULT_DATASET_PATH,
    )
    classifier_path = classifier_path or os.environ.get(
        "PHISHLENS_CLASSIFIER_PATH",
        DEFAULT_CLASSIFIER_PATH,
    )
    vectorizer_path = vectorizer_path or os.environ.get(
        "PHISHLENS_VECTORIZER_PATH",
        DEFAULT_VECTORIZER_PATH,
    )
    dataset, dataset_summary = load_labeled_dataset(dataset_path)
    labels = dataset["_label"]
    record_count = len(dataset)
    test_count = max(math.ceil(record_count * test_size), len(CLASS_LABELS))
    if test_count >= record_count or record_count - test_count < len(CLASS_LABELS):
        raise DatasetError(
            "Dataset is too small for a stratified training and held-out test split."
        )

    try:
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            dataset["_normalized_text"],
            labels,
            test_size=test_count,
            random_state=random_state,
            stratify=labels,
        )
    except ValueError as error:
        raise DatasetError(
            f"Could not create a stratified train/test split: {error}"
        ) from error

    vectorizer = TfidfVectorizer(
        preprocessor=clean_email_text,
        lowercase=False,
    )
    classifier = MultinomialNB()
    try:
        training_features = vectorizer.fit_transform(train_texts)
        classifier.fit(training_features, train_labels)
        predictions = classifier.predict(vectorizer.transform(test_texts))
    except ValueError as error:
        raise DatasetError(
            f"Could not fit the email model using this dataset: {error}"
        ) from error

    matrix = confusion_matrix(test_labels, predictions, labels=CLASS_LABELS)
    sample_count = min(10, len(test_texts))
    sample_texts = test_texts.iloc[:sample_count]
    sample_labels = test_labels.loc[sample_texts.index]
    sample_features = vectorizer.transform(sample_texts)
    sample_predictions = classifier.predict(sample_features)
    sample_probabilities = classifier.predict_proba(sample_features)
    class_indexes = {
        label: index
        for index, label in enumerate(classifier.classes_)
    }
    summary = {
        **dataset_summary,
        "training_records": len(train_texts),
        "test_records": len(test_texts),
        "unseen_prediction_smoke_test": {
            "checked": sample_count,
            "correct": int(
                sum(
                    predicted == actual
                    for predicted, actual in zip(
                        sample_predictions,
                        sample_labels,
                        strict=True,
                    )
                )
            ),
            "results": [
                {
                    "actual": str(actual),
                    "predicted": str(predicted),
                    "confidence": float(
                        probabilities[class_indexes[predicted]]
                    ),
                    "correct": bool(predicted == actual),
                }
                for actual, predicted, probabilities in zip(
                    sample_labels,
                    sample_predictions,
                    sample_probabilities,
                    strict=True,
                )
            ],
        },
        "evaluation": {
            "accuracy": float(accuracy_score(test_labels, predictions)),
            "precision": float(precision_score(
                test_labels, predictions, pos_label="spam", zero_division=0
            )),
            "recall": float(recall_score(
                test_labels, predictions, pos_label="spam", zero_division=0
            )),
            "f1_score": float(f1_score(
                test_labels, predictions, pos_label="spam", zero_division=0
            )),
            "confusion_matrix": {
                "labels": list(CLASS_LABELS),
                "values": matrix.tolist(),
            },
        },
    }

    for artifact_path in (classifier_path, vectorizer_path):
        os.makedirs(os.path.dirname(os.path.abspath(artifact_path)), exist_ok=True)
    joblib.dump(classifier, classifier_path)
    joblib.dump(vectorizer, vectorizer_path)
    evaluation_path = os.environ.get(
        "PHISHLENS_EVALUATION_PATH",
        DEFAULT_EVALUATION_PATH,
    )
    os.makedirs(os.path.dirname(os.path.abspath(evaluation_path)), exist_ok=True)
    with open(evaluation_path, "w", encoding="utf-8") as report_file:
        json.dump(summary, report_file, indent=2)
        report_file.write("\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_training(), indent=2))
