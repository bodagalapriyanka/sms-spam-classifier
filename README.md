# Project 1: Text Classification System

Learn to classify messages as **spam** or **not spam** with TF-IDF and scikit-learn classifiers. This replaces the keyword exercise.

New to text classification? Follow the [learning guide](LEARNING_GUIDE.md) for a walkthrough of the code, hands-on exercises, metric calculations, and checkpoints. See the [project plan](PROJECT_PLAN.md) for planned improvements and documentation.

Read [my learning journey and results](LEARNING_JOURNEY.md) for recorded model comparisons, observations, improvements, and remaining limitations.

The [data guide](docs/DATA.md) explains the datasets and CSV format. The [model card](docs/MODEL_CARD.md) describes the selected model, recorded evaluation, and intended use.

Before sharing the repository, review [which files to publish](docs/PUBLISHING.md).

## Setup

The development environment uses **Python 3.14.6**. A clean virtual environment on macOS was verified with installation, all three unit tests, training, and saved-model prediction on October 16, 2026. Training reproduced the recorded test confusion matrix. Other Python versions and operating systems have not yet been verified for this project. Use `requirements-lock.txt` for the recorded exact runtime dependency versions; `requirements.txt` contains broader direct dependency ranges for development.

In your Mac VS Code terminal, inside this folder:

```sh
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

The virtual environment already exists. For a fresh checkout create it first with `python3 -m venv .venv`. On Windows use `py -m venv .venv`, then activate in PowerShell with `.venv\Scripts\Activate.ps1`.

## Train and evaluate

```sh
python main.py
```

The program reads `data/messages.csv` and creates stratified 60% training, 20% validation, and 20% test splits using fixed seeds. Duplicate messages are removed before splitting.

It compares three Naive Bayes smoothing settings (`alpha=1`, `0.1`, `0.01`) and logistic regression with and without balanced class weights. Additional logistic regression candidates use sublinear term frequency, `C=3` or `C=10`, and either word features alone or words combined with character fragments of length 3–5. Character fragments can capture spelling patterns; the classifier learns how features combine rather than applying a keyword override. Each candidate learns TF-IDF from training data alone. Validation spam F1 chooses the winner, with precision breaking ties. F1 balances catching spam against incorrectly flagging legitimate messages; it does not guarantee zero false positives.

The selected configuration is refitted on training plus validation data, then evaluated once on the reserved test split. The saved model is exactly this evaluated model. The test set is never used for candidate selection or fitting. Results include precision, recall, accuracy, and a confusion matrix. The model is saved to `models/spam_classifier.joblib` and the comparison and test metrics to `models/spam_classifier.metrics.json`.

This evaluation is exploratory: the same corpus was inspected during earlier experiments, so a new split does not create truly unseen external evidence. Repeated tuning after reading test scores would further compromise the test set. Use a fresh external dataset for a stronger final check.

## Try your own messages

```sh
python main.py predict "You won a free prize"
python main.py predict "Are you free for lunch?"
python main.py interactive
```

Interactive mode accepts one message at a time; enter `quit` to finish. Retrain after editing the dataset. `python main.py --help` shows all options. Use `python main.py train --data path/to/messages.csv` for your own labeled CSV; `--model` selects a different model path.

## How the model works

1. The CSV labels each message `spam` or `ham` (legitimate).
2. TF-IDF turns words and two-word phrases into numerical features. Term frequency measures occurrence within a message; inverse document frequency downweights terms common across training messages. Vectors are normalized.
3. Naive Bayes and logistic regression learn associations between features and labels rather than checking a fixed keyword list. Validation results select the configuration.
4. Test examples are withheld throughout fitting. Candidate pipelines fit TF-IDF on training data; the selected pipeline then refits on training plus validation data before the final test evaluation.

Accuracy is the fraction of correct predictions. Spam precision measures how many flagged messages really are spam; spam recall measures how much actual spam was caught. F1 balances precision and recall. The confusion matrix orders rows (actual labels) and columns (predicted labels) as `ham`, `spam`: top-right counts false positives, bottom-left counts false negatives.

## Dataset and limitations

The default dataset is the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) by Tiago Almeida and Jose Maria Gomez Hidalgo (2011), DOI: 10.24432/C5CC84, distributed under CC BY 4.0. The downloaded archive contains 5,574 labeled messages (4,827 ham and 747 spam). Conversion preserves these rows in `data/messages.csv`; the loader removes duplicates before splitting, leaving 5,159 unique normalized messages (4,517 ham and 642 spam).

The original archive and its README are retained in `data/`. The original 60 synthetic examples are preserved in `data/teaching_messages.csv`. This older SMS corpus does not represent all modern spam. Since legitimate messages are the majority, inspect spam recall and precision alongside accuracy. Predictions can be wrong. Unknown words are ignored; messages with no recognized features are classified using class priors.

Improve this baseline using a larger representative labeled dataset, inspecting false positives/negatives, and tuning on validation data while keeping a final test set untouched. Duplicate examples are removed; conflicting labels are rejected. Keep private messages out of public repositories. Load only trusted model files because joblib deserialization can execute code.

## Project files

- `main.py`: CSV loading, training, evaluation, prediction, and CLI.
- `data/messages.csv`: editable examples.
- `data/teaching_messages.csv`: original beginner dataset.
- `data/sms+spam+collection.zip`: downloaded source archive.
- `data/UCI_README.txt`: original dataset documentation.
- `requirements.txt`: dependencies.
- `models/spam_classifier.joblib`: generated model (excluded from Git).

## Verification

Run `python -m unittest discover -s tests -v` to check split isolation, TF-IDF leakage prevention, and metric calculations.

The suite also checks CSV validation, duplicate handling, CLI errors, and training followed by saved-model prediction. All 10 tests pass locally. The GitHub Actions workflow in `.github/workflows/tests.yml` installs the pinned dependencies and runs the suite on pushes and pull requests; its first hosted run is pending publication.

## License

Original project code and documentation are available under the [MIT License](LICENSE), copyright 2026 Priyanka. Third-party datasets and their original documentation retain their own terms and attribution; see the [data guide](docs/DATA.md).

## Manual diagnostic checks

The seven conversation examples are stored in `data/diagnostic_messages.csv`. These are known examples for inspecting behavior, not an independent accuracy estimate. They are never added to training or used to select the model. The original dataset file is unchanged.
