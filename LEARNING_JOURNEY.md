# My Text Classification Project: Learning, Observations, and Results

This project explores how to classify messages as spam or legitimate text using TF-IDF and scikit-learn. This write-up connects my learning goals with the implementation and results recorded in the project files.

## What I built

I built a command-line classifier that loads labeled CSV messages, validates and deduplicates them, compares model configurations, saves the selected pipeline, and predicts labels for new messages.

The project includes Naive Bayes and logistic regression, word and character features, validation-based selection, and test evaluation. The README describes it as a replacement for an earlier keyword exercise; the earlier implementation and its measured results are not preserved here.

## My starting point

I used this project to connect the steps of a machine learning workflow: preparing examples, turning text into features, comparing models, and checking predictions. Spam classification gave me a concrete problem where I could read a message, consider its label, and compare that with the model's output.

My goal is to understand why a model makes mistakes and how to judge an improvement, as well as to build a working program. I want someone learning alongside me to be able to follow the code and question the results.

## What I explored and improved

These are comparisons supported by the current implementation and saved metrics. They describe configurations evaluated in one recorded run, rather than a verified chronological history of separate project versions.

| Area | What the project explores | What the recorded evidence shows |
| --- | --- | --- |
| Naive Bayes smoothing | Compare `alpha=1`, `0.1`, and `0.01`. | Validation spam recall increases from 38.76% to 89.15% between alpha 1 and 0.01. |
| Class imbalance | Compare ordinary and balanced logistic regression. | In the base configurations, balancing increases validation recall from 55.04% to 90.70%, while precision drops from 100% to 93.60%. |
| Text representation | Compare word features with word plus character features. | At balanced `C=10`, validation F1 is 92.86% for words and 94.78% for words plus characters. |
| Evaluation boundaries | Deduplicate before splitting and fit candidate vectorizers on training data. | The code separates training, validation, and test responsibilities; existing tests check specific split and vocabulary properties. |
| Reporting | Report precision, recall, F1, and a confusion matrix alongside accuracy. | These reveal missed spam and legitimate messages flagged incorrectly. |

These comparisons do not prove that any setting is universally better. The word-only and word-plus-character variants both use sublinear term frequency, making their paired comparison more informative than comparing them with a base configuration that changes multiple settings.

## My recorded results

### Dataset and selection

The saved training report records 5,159 unique messages, split into 3,095 training, 1,032 validation, and 1,032 test examples. Candidates are selected using validation spam F1, with precision breaking ties. The selected configuration is then refitted on training plus validation before test evaluation.

### Validation comparison

Selected rows from the saved comparison are shown below. All numbers are percentages rounded to two decimals.

| Configuration | Spam precision | Spam recall | Spam F1 |
| --- | ---: | ---: | ---: |
| Naive Bayes, alpha 1 | 100.00 | 38.76 | 55.87 |
| Naive Bayes, alpha 0.1 | 100.00 | 83.72 | 91.14 |
| Naive Bayes, alpha 0.01 | 98.29 | 89.15 | 93.50 |
| Base logistic regression | 100.00 | 55.04 | 71.00 |
| Base balanced logistic regression | 93.60 | 90.70 | 92.13 |
| Balanced LR, words, C=10 | 95.12 | 90.70 | 92.86 |
| Balanced LR, words + characters, C=3 | 98.32 | 90.70 | 94.35 |
| Balanced LR, words + characters, C=10 | 98.33 | 91.47 | 94.78 |

The winner was **balanced logistic regression with word and character features, C=10**. This selection used validation results, not test results.

### Held-out test results

| Metric | Recorded value |
| --- | ---: |
| Accuracy | 98.93% |
| Spam precision | 96.80% |
| Spam recall | 94.53% |
| Spam F1 | 95.65% |

| Actual / predicted | Ham | Spam |
| --- | ---: | ---: |
| Ham | 900 | 4 |
| Spam | 7 | 121 |

On these 1,032 messages, the classifier flagged 4 legitimate messages as spam and missed 7 spam messages. These are results for this particular split of the UCI corpus, not an expected accuracy for all future messages.

The source is the locally saved `models/spam_classifier.metrics.json`. That directory is excluded from Git, so this document preserves a human-readable snapshot. The report has no run timestamp, environment versions, or dataset hash; this write-up does not claim that the run was reproduced during document creation.

### Synthetic challenge results

A separate saved report contains results for 60 AI-authored challenge messages:

| Metric | Recorded value |
| --- | ---: |
| Accuracy | 66.67% |
| Spam precision | 85.71% |
| Spam recall | 40.00% |
| Spam F1 | 54.55% |

Its confusion matrix records 28 correctly allowed legitimate messages, 2 false spam flags, 18 missed spam messages, and 12 correctly detected spam messages.

Sources: [challenge metrics](reports/fresh_synthetic_metrics.json) and [individual predictions](reports/fresh_synthetic_predictions.csv). The exact model identity and generation command are not recorded in those files. Consequently, this is a separate diagnostic observation; it does not establish a controlled performance comparison with the selected model above. Synthetic examples also do not represent real-world prevalence or performance.

## What I observed

- High precision can coexist with low recall. Naive Bayes with alpha 1 flags spam very precisely in validation but misses most spam examples.
- Class balancing changes the tradeoff. In the base logistic regression comparison, more spam is caught at the cost of more false positives.
- Character features help in the recorded paired validation comparison. This supports exploring spelling fragments for this dataset, without assuming they always improve performance.
- Familiar words do not determine intent. In the saved challenge predictions, “Are you free after your appointment on Thursday?” is correctly allowed, but “The community clinic offers a free workshop on Saturday.” is incorrectly flagged.
- The challenge predictions miss several scam-style messages, including a parcel-fee message requesting card details. These cases suggest a need for broader evaluation; they do not establish the cause of the errors.

## Learning points illustrated by this project

1. Text needs a numeric representation before these classifiers can learn from it. TF-IDF provides weighted word and character features.
2. A pipeline keeps learned preprocessing and classification together when saving and predicting.
3. Training teaches parameters, validation guides configuration choices, and a final test evaluates the chosen configuration.
4. Accuracy alone is insufficient when class frequencies and mistake costs differ.
5. A strong score on one corpus can coexist with weaknesses on other examples.
6. Reproducibility requires recording data, environment, configuration, and evaluation commands as well as scores.

The biggest lesson I take from these results is to look beyond a single accuracy number. The recorded test accuracy is 98.93%, but the separate synthetic challenge report catches only 40% of its spam examples. Those reports are not a controlled comparison of the same identified model, but together they remind me to ask what was evaluated before drawing a conclusion.

The workshop message is a useful example for me: “The community clinic offers a free workshop on Saturday.” is legitimate in the challenge labels, yet it was flagged as spam. I can see why relying on an isolated word such as “free” would be fragile. I also need to avoid assuming that I know the exact cause of this prediction without investigating it.

The improvements represented in this project include comparing smoothing settings, balancing classes, combining word and character features, and selecting configurations with validation metrics. I can point to the recorded comparisons to explain those choices. My next improvement is to make the experiments easier to reproduce, so another learner can check the evidence rather than take my conclusions on trust.

## Limitations and next improvements

The UCI corpus was inspected during earlier experiments, as acknowledged in the training report and README. The recorded test score is therefore exploratory evidence; a new split does not undo earlier exposure to the corpus.

Next improvements are to record reproducible evaluation commands and model identity, pin the working environment, expand input/CLI tests, and obtain a fresh externally sourced evaluation after freezing model choices. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full checklist.

## My next experiment

*Complete before running another experiment:*

- **Question:** What specific weakness do I want to investigate?
- **Hypothesis:** Why might my proposed change help?
- **Change:** What single factor will I vary?
- **Evidence:** Which validation metrics and examples will I inspect?
- **Outcome:** What happened, and what can I reasonably conclude?

For a step-by-step introduction to the concepts and exercises, see [LEARNING_GUIDE.md](LEARNING_GUIDE.md).
