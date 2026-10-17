# Model card: TF-IDF spam classifier

## Purpose and intended use

This local command-line project teaches supervised text classification of English messages into `ham` and `spam`. It supports learning, experiments, and portfolio demonstrations. Its measured results do not establish suitability for automatic filtering of important communications.

Input is a nonempty text message; output is a predicted label. The CLI does not expose a calibrated confidence score.

## Model and features

The implementation compares 13 configurations: three Multinomial Naive Bayes smoothing settings, two base logistic regression configurations, and eight logistic regression variants combining word/character features, regularization, and class weighting.

Word features use TF-IDF with one- and two-word sequences. Extended variants use sublinear term frequency and optionally combine word features with `char_wb` fragments of length 3–5, with `min_df=2` for the character vectorizer.

The saved training report selects balanced logistic regression with word plus character features and `C=10`. That configuration uses `liblinear`, `max_iter=1000`, and random seed 42. The serialized pipeline includes both feature processing and classification.

## Training and selection

Data comes from the UCI SMS Spam Collection; see [DATA.md](DATA.md). The saved run contains 5,159 deduplicated messages, split approximately 60/20/20 into training, validation, and test groups.

Each candidate fits on training data. Highest validation spam F1 selects the winner, with spam precision breaking ties. The selected configuration then fits on training plus validation and is evaluated on the test split. Its validation spam F1 is 94.78%.

## Recorded test results

| Metric | Value |
| --- | ---: |
| Test examples | 1,032 |
| Accuracy | 98.93% |
| Spam precision | 96.80% |
| Spam recall | 94.53% |
| Spam F1 | 95.65% |

| Actual / predicted | Ham | Spam |
| --- | ---: | ---: |
| Ham | 900 | 4 |
| Spam | 7 | 121 |

Source: the local `models/spam_classifier.metrics.json`, summarized in [LEARNING_JOURNEY.md](../LEARNING_JOURNEY.md). A subsequent clean-environment check on October 16, 2026, using Python 3.14.6 and `requirements-lock.txt` reproduced the test confusion matrix shown above. The original source report lacks a timestamp, dependency versions, and dataset hash. Generated model files and their local metrics are excluded from Git.

## Diagnostic evidence

The saved 60-message AI-authored challenge report records 66.67% accuracy, 85.71% spam precision, 40.00% recall, and 54.55% F1. Its exact model identity and generation command are missing, so it cannot establish a controlled comparison with the selected model. It is not representative real-world evidence. See the [report](../reports/fresh_synthetic_metrics.json) and [individual predictions](../reports/fresh_synthetic_predictions.csv).

## Limitations

- The development corpus was inspected in earlier experiments. Reported held-out results remain exploratory.
- Older SMS language may differ from modern scams, other communication channels, and other languages.
- Word-only features ignore unseen vocabulary; character features may recognize fragments without understanding intent.
- Legitimate messages can be flagged and spam can be missed. Precision and recall depend on the evaluation population.
- No subgroup evaluation, calibrated confidence assessment, or production monitoring is documented.

## Reproduce the workflow

After installing dependencies as described in the README:

```sh
python -m unittest discover -s tests -v
python main.py train
python main.py predict "Can we meet tomorrow?"
```

Training creates `models/spam_classifier.joblib` and `models/spam_classifier.metrics.json`, replacing files at those default paths. Use `--model models/experiment.joblib` to keep a separate experiment. Exact reproduction of the saved numbers requires matching the original data and environment, which are not fully fingerprinted in the report.

Load only trusted joblib artifacts because deserialization can execute code. Retrain after changing data or feature configuration. Future work includes environment pinning, report provenance, broader tests, and fresh external evaluation; see [PROJECT_PLAN.md](../PROJECT_PLAN.md).
