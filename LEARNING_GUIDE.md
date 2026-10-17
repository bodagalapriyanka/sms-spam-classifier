# Learn Text Classification with This Project

Use this project to learn how a machine learning system turns labeled examples into predictions. You will train a spam classifier, inspect how it works, and practice evaluating it honestly.

You need basic Python knowledge: functions, lists, dictionaries, loops, and running commands in a terminal. No previous machine learning experience is required.

## 1. Run the project

Open a terminal in the project folder. Create and activate a virtual environment if you do not already have one:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

On Windows, use `py -m venv .venv` and activate with `.venv\Scripts\Activate.ps1` instead.

Train the classifier:

```sh
python main.py train
```

Training prints a validation comparison, selects a model, and prints test metrics. It saves a model and a JSON report under `models/`. The first run may take some time because it compares several configurations.

Try a few predictions:

```sh
python main.py predict "Congratulations! Claim your cash prize now"
python main.py predict "Can we meet for lunch tomorrow?"
python main.py predict "Are you free this evening?"
python main.py interactive
```

Enter `quit` to leave interactive mode. Predictions may be wrong. Before running each example, write down your expected label; compare it with the output without treating a few successful examples as an accuracy measurement.

**Checkpoint:** Explain the difference between training a model and using an already trained model to predict.

## 2. Understand the data

Open `data/messages.csv`. Each row has two fields:

| Field | Meaning |
| --- | --- |
| `label` | `spam` for unwanted promotional/scam messages, or `ham` for legitimate messages in this dataset. |
| `message` | The text the model receives as input. |

This is **supervised learning**: the training examples include the answers. The model learns statistical patterns associated with those labels.

Read `load_data()` in `main.py`. It checks the column names, rejects missing text and invalid labels, removes duplicate messages, and rejects conflicting labels for the same normalized text. Duplicate detection ignores case and repeated whitespace; it does not detect every paraphrase.

The default data comes from the UCI SMS Spam Collection; attribution and limitations are in the README. `data/teaching_messages.csv` contains the smaller original teaching dataset. `data/diagnostic_messages.csv` contains known examples for inspecting behavior. Diagnostic examples are not an independent test set.

**Exercise:** Make a copy of the teaching CSV and try an empty message, an invalid label, a duplicate row, and a duplicate with a conflicting label. Train using `--data` and observe which cases fail and which are deduplicated. Keep the original datasets intact.

**Checkpoint:** Why can putting duplicate messages in both training and test data make evaluation misleading?

## 3. Follow the training flow

Read these functions in this order:

| Function | Question it answers |
| --- | --- |
| `load_data()` | Are the examples valid and unique? |
| `split_data()` | Which examples are available at each stage? |
| `candidates()` | Which feature/model configurations will be compared? |
| `scores()` | How do predictions become evaluation numbers? |
| `train()` | How are models fitted, selected, evaluated, and saved? |
| `predict()` and `main()` | How does the user interact with the saved model? |

The data flow is:

```text
Labeled messages
    -> validate and deduplicate
    -> split into training / validation / test
    -> fit each candidate on training data
    -> select using validation spam F1, then precision for ties
    -> refit the winner on training + validation
    -> evaluate on test data
    -> save the evaluated pipeline and its metrics
```

The split is approximately 60% training, 20% validation, and 20% test. Stratification keeps label proportions similar across the groups. Fixed random seeds make the split repeatable for the same input data and environment.

Training data teaches the model. Validation data helps choose its configuration. Test data estimates the selected model's performance after those choices are complete. Repeatedly changing the model based on test results makes that test set part of development, weakening its value as independent evidence.

**Checkpoint:** If you choose a regularization setting because it improves test F1, which boundary have you crossed?

## 4. Understand how text becomes numbers

Read the `TfidfVectorizer` definitions in `candidates()`.

Models need numeric input. TF-IDF creates features from text and assigns weights based on how often a feature appears in a message and how common it is across the training collection. Common features generally receive less distinctive weight. Scikit-learn also applies normalization; TF-IDF is not simply a count of keywords.

`ngram_range=(1, 2)` includes individual words and pairs of consecutive words. For example, “claim your prize” can produce word features such as `claim`, `your`, `prize`, `claim your`, and `your prize`.

Some candidates also use character fragments of length 3–5 through `char_wb`. These features can capture parts of words and spelling patterns. `FeatureUnion` combines the word and character feature sets. Sublinear term frequency reduces the influence of repeated occurrences of a feature.

A `Pipeline` keeps the vectorizer and classifier together. During fitting, the vectorizer learns vocabulary and IDF from the fitting data. During prediction, it uses those learned values. Fitting a vectorizer on the entire dataset before splitting would let information from evaluation data enter training: this is **data leakage**.

**Exercise:** In a Python shell with the environment activated, inspect a small vectorizer:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
features = vectorizer.fit_transform([
    "claim your prize",
    "bring your lunch",
])
print(vectorizer.get_feature_names_out())
print(features.shape)
print(vectorizer.transform(["anunseenword"]).nnz)
```

The last output is `0`: that word has no feature in the learned vocabulary. The model cannot directly learn a meaning for a word it has never seen. Character features can sometimes recognize fragments of unfamiliar words, but that does not guarantee correct predictions.

## 5. Compare the classifiers

The project compares two classifier families:

| Classifier | Intuition | Setting explored here |
| --- | --- | --- |
| Multinomial Naive Bayes | Combines feature evidence using a simplifying conditional independence assumption. | `alpha` controls smoothing. |
| Logistic regression | Learns feature weights that help separate the two labels. | `C` controls inverse regularization strength; class balancing changes the relative penalty for mistakes on each class. |

For logistic regression, a larger `C` means weaker regularization. `class_weight="balanced"` gives more weight to the less frequent class according to label frequencies. It may change the precision/recall tradeoff; it does not guarantee a better model.

The selection rule favors validation spam F1 and uses spam precision to break ties. This is a design decision, not a universal rule: a different application might care more about avoiding false spam flags.

**Exercise:** Read the printed validation table. Identify the best candidate by the implemented selection rule. Compare its precision and recall with another candidate. Explain the tradeoff before looking at the test results.

## 6. Interpret evaluation correctly

In this project, spam is the positive class:

| Outcome | Meaning |
| --- | --- |
| True positive (TP) | Spam correctly flagged as spam. |
| False positive (FP) | A legitimate message incorrectly flagged as spam. |
| False negative (FN) | Spam incorrectly allowed as legitimate. |
| True negative (TN) | A legitimate message correctly allowed. |

The printed confusion matrix orders actual labels by rows and predicted labels by columns:

| Actual / predicted | Ham | Spam |
| --- | --- | --- |
| Ham | TN | FP |
| Spam | FN | TP |

- **Accuracy:** `(TP + TN) / all messages`.
- **Spam precision:** `TP / (TP + FP)`. Of the messages flagged, how many were spam?
- **Spam recall:** `TP / (TP + FN)`. Of the actual spam messages, how many were caught?
- **Spam F1:** the harmonic mean of spam precision and recall, `2PR / (P + R)`.

The code uses zero for undefined precision/recall/F1 cases. Because legitimate messages are the majority in this dataset, accuracy alone can hide poor spam detection.

**Exercise:** Suppose TN = 80, FP = 5, FN = 10, and TP = 5. Calculate accuracy, spam precision, recall, and F1 before checking the answers below.

**Answers:** Accuracy = 85%; precision = 50%; recall ≈ 33.3%; F1 = 40%. The classifier catches only one third of the spam despite its 85% accuracy.

## 7. Read and run the tests

```sh
python -m unittest discover -s tests -v
```

Read `tests/test_training.py`. The existing tests check that splits are disjoint and repeatable, label proportions are similar, prediction does not expand the learned vocabulary, and a known example produces the expected metrics.

Passing these tests supports specific implementation properties. It does not prove that all inputs are handled or that the model will perform well on new messages.

**Exercise:** Add a test for conflicting labels in a temporary CSV. Confirm that `load_data()` raises `ValueError`. Then consider a separate test for duplicate removal: what result should it assert?

## 8. Experiment without losing your baseline

Use a separate data copy and model path for experiments:

```sh
python main.py train --data data/teaching_messages.csv --model models/learning_experiment.joblib
python main.py predict "Your message here" --model models/learning_experiment.joblib
```

The small teaching dataset is useful for understanding the workflow; its scores are too sensitive to a handful of examples to support broad performance claims.

For each experiment, record:

| Item | What to write |
| --- | --- |
| Question | What do you want to understand? |
| Hypothesis | What do you expect, and why? |
| Change | One setting or feature choice you changed. |
| Evidence | Validation precision, recall, F1, and relevant observations. |
| Interpretation | What the result supports and what remains uncertain. |

Suggested experiments: compare words alone with word and character features, compare balanced and unbalanced logistic regression, or change the amount of training data while preserving evaluation boundaries. Keep the original baseline available and make selection decisions using validation data.

Use test results for a final evaluation after choices are fixed. If you repeatedly inspect and tune against a test set, report that limitation and obtain a fresh external evaluation before making strong generalization claims.

## 9. Recognize the limits

This is a learning baseline built on an older SMS corpus. It may struggle with modern scams, unfamiliar language, ambiguous messages, and contexts absent from training. A keyword such as “free” does not determine a label by itself.

The README notes that this corpus was inspected during earlier experiments. A held-out split within an already explored corpus does not erase that history. Synthetic challenge examples can reveal failures, but their scores do not establish real-world performance.

Keep private messages out of shared datasets. Load only trusted joblib model files because loading them can execute code. The saved artifact contains both feature processing and the classifier; retrain when changing training data or feature configuration.

## Learning completion checklist

- [ ] I can install, train, and predict using the CLI.
- [ ] I can explain features, labels, fitting, and inference.
- [ ] I can explain training, validation, and test responsibilities.
- [ ] I can describe word features, character features, and TF-IDF.
- [ ] I can explain how a pipeline helps keep fitting and prediction consistent.
- [ ] I can calculate precision, recall, F1, and accuracy from a confusion matrix.
- [ ] I can describe a false positive and false negative in this application.
- [ ] I can run the tests and explain what each one checks.
- [ ] I can record an experiment without tuning against test results.
- [ ] I can explain why this project does not establish production readiness.

For proposed engineering improvements and additional documentation, see [PROJECT_PLAN.md](PROJECT_PLAN.md). For command details and dataset attribution, see [README.md](README.md).
