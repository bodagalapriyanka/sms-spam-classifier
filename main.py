"""Train and use a TF-IDF spam classifier."""
import argparse
import csv
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parent


def load_data(path):
    messages, labels, seen = [], [], {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"label", "message"}.issubset(reader.fieldnames):
            raise ValueError("CSV must contain label and message columns.")
        for line, row in enumerate(reader, start=2):
            label = (row.get("label") or "").strip().lower()
            message = (row.get("message") or "").strip()
            if label not in {"spam", "ham"} or not message:
                raise ValueError(f"Invalid row {line}: use spam/ham and a nonempty message.")
            key = " ".join(message.lower().split())
            if key in seen:
                if seen[key] != label:
                    raise ValueError(f"Conflicting labels on row {line}.")
                continue
            seen[key] = label
            messages.append(message)
            labels.append(label)
    if any(labels.count(label) < 5 for label in ("spam", "ham")):
        raise ValueError("Provide at least five unique messages per label.")
    return messages, labels


def split_data(messages, labels):
    """Create repeatable, stratified 60/20/20 training/validation/test sets."""
    x_dev, x_test, y_dev, y_test = train_test_split(
        messages, labels, test_size=0.2, random_state=42, stratify=labels
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_dev, y_dev, test_size=0.25, random_state=43, stratify=y_dev
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def candidates():
    """A fixed comparison chosen before inspecting test results."""
    classifiers = [
        ("Naive Bayes alpha=1", MultinomialNB(alpha=1.0)),
        ("Naive Bayes alpha=0.1", MultinomialNB(alpha=0.1)),
        ("Naive Bayes alpha=0.01", MultinomialNB(alpha=0.01)),
        ("Logistic regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Balanced logistic regression", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42)),
    ]
    models = [(name, Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", classifier),
    ])) for name, classifier in classifiers]
    # Alternative regularization settings and character fragments offer alternatives
    # to Naive Bayes' tendency to overreact to isolated words such as "free".
    for features in ("word", "word+char"):
        for c in (3.0, 10.0):
            for balanced in (False, True):
                word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
                vectorizer = word if features == "word" else FeatureUnion([
                    ("word", word),
                    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                            sublinear_tf=True, min_df=2)),
                ])
                name = f"LR {features} C={c:g}" + (" balanced" if balanced else "")
                models.append((name, Pipeline([
                    ("tfidf", vectorizer),
                    ("classifier", LogisticRegression(
                        C=c, class_weight="balanced" if balanced else None,
                        solver="liblinear", max_iter=1000, random_state=42)),
                ])))
    return models


def scores(actual, predicted):
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average="binary", pos_label="spam", zero_division=0
    )
    return {"accuracy": float(accuracy_score(actual, predicted)),
            "spam_precision": float(precision), "spam_recall": float(recall),
            "spam_f1": float(f1)}


def train(data_path, model_path):
    messages, labels = load_data(data_path)
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(messages, labels)
    print(f"Training: {len(x_train)} | Validation: {len(x_val)} | Test: {len(x_test)}")
    print("Validation comparison (choose highest spam F1; ties favor precision):")
    print(f"{'Candidate':32} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    results = []
    best_key, winner, model = (-1, -1), None, None
    for name, candidate in candidates():
        # Each candidate learns vocabulary and IDF from training examples only.
        candidate.fit(x_train, y_train)
        metrics = scores(y_val, candidate.predict(x_val))
        results.append({"name": name, **metrics})
        print(f"{name:32} {metrics['spam_precision']:10.3f} "
              f"{metrics['spam_recall']:10.3f} {metrics['spam_f1']:10.3f}")
        key = (metrics["spam_f1"], metrics["spam_precision"])
        if key > best_key:
            best_key, winner, model = key, name, candidate

    print(f"\nSelected: {winner}")
    # Refit the selected configuration on training + validation, then test once.
    model.fit(x_train + x_val, y_train + y_val)
    predictions = model.predict(x_test)
    final_scores = scores(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=["ham", "spam"])
    print(f"Test accuracy: {final_scores['accuracy']:.1%}\n")
    print(classification_report(y_test, predictions, labels=["ham", "spam"], zero_division=0))
    print("Confusion matrix (rows=actual, columns=predicted; order: ham, spam):")
    print(matrix)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    report_path = model_path.with_suffix(".metrics.json")
    report = {
        "dataset": str(data_path.resolve()), "unique_examples": len(messages),
        "split_sizes": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
        "split_seeds": {"test": 42, "validation": 43},
        "selection_rule": "highest validation spam F1; ties favor precision",
        "validation_results": results, "selected_model": winner,
        "test_scores": final_scores, "confusion_matrix_ham_spam": matrix.tolist(),
        "limitation": "Exploratory: this corpus was inspected in earlier experiments; a fresh external test set is needed.",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved model: {model_path}\nSaved metrics: {report_path}")
    print(report["limitation"])
    return model


def predict(model, message):
    if not message.strip():
        raise ValueError("Enter a nonempty message.")
    label = model.predict([message])[0]
    print(f"{'SPAM' if label == 'spam' else 'NOT SPAM'}: {message}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="train", choices=["train", "predict", "interactive"])
    parser.add_argument("message", nargs="?", help="Message to classify, in quotes")
    parser.add_argument("--data", type=Path, default=ROOT / "data/messages.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "models/spam_classifier.joblib")
    args = parser.parse_args()
    if args.command == "predict" and not args.message:
        parser.error('Provide a message: python main.py predict "Your message"')
    try:
        if args.command == "train":
            train(args.data, args.model)
            return
        if not args.model.exists():
            parser.error("Model not found. Run python main.py train first.")
        # Only load trusted joblib files: deserialization can execute code.
        model = joblib.load(args.model)
        if args.command == "predict":
            predict(model, args.message)
        else:
            print("Enter a message, or type quit to exit.")
            while True:
                try:
                    message = input("> ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if message.strip().lower() in {"quit", "exit"}:
                    break
                if message.strip():
                    predict(model, message)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
