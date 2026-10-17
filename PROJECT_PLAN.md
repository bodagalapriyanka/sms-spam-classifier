# Text Classification Project: Completion Plan

This document tracks the additions needed to make this learning project easier to reproduce, review, and extend. The current CLI is ready to share as a baseline; the unchecked items below are improvements, not claims that they already exist.

## Scope and intended outcome

Classify English messages as `spam` or `ham` using TF-IDF and scikit-learn. A new contributor should be able to install dependencies, run tests, train a model, and classify a message using the README alone.

The current scope is a local CLI. A web interface, hosted API, user accounts, and production deployment are optional future work.

## Already implemented

- [x] CSV input validation, duplicate removal, and conflicting-label rejection.
- [x] Repeatable stratified training, validation, and test splits.
- [x] TF-IDF and Naive Bayes/logistic regression candidates.
- [x] Selection by validation spam F1, with precision breaking ties.
- [x] Refit on training plus validation and evaluation on the held-out test split.
- [x] Model persistence and JSON evaluation reports.
- [x] Prediction and interactive CLI commands.
- [x] Tests for split boundaries, prediction vocabulary isolation, and metric calculations.
- [x] README with setup, usage, dataset attribution, and evaluation limitations.
- [x] Git exclusions for the virtual environment, caches, and generated models.

Verification: all three existing unit tests passed on October 16, 2026. This does not establish performance on new real-world messages.

## Prioritized additions

| Priority | Addition | Purpose | Done when |
| --- | --- | --- | --- |
| P1 | Reproducible environment | Make a fresh installation predictable | A supported Python version is documented, exact dependency versions are recorded in a lock or constraints file, and installation is verified in a clean environment. |
| P1 | Automated checks | Catch regressions on future changes | GitHub Actions installs dependencies and runs the unit tests on pushes and pull requests. |
| P1 | Input and CLI coverage | Verify expected errors and the saved-model workflow | Tests cover invalid CSVs, duplicate/conflicting rows, blank messages, missing model files, and training followed by loading and predicting with a temporary model. |
| P1 | Evaluation report provenance | Make reported scores interpretable | Each checked-in report identifies its model/configuration, dataset, generation command, and limitations; unsupported freshness claims are removed or qualified. |
| P1 | Repository license | Clarify reuse of original project code | The owner chooses a code license and a root LICENSE file is added; dataset attribution and its separate terms remain documented. |
| P2 | Data documentation | Explain each dataset and its intended use | `docs/DATA.md` documents schema, sources, preprocessing, and whether each file is for training, diagnostics, or evaluation. |
| P2 | Model and evaluation documentation | Explain behavior and evidence | `docs/MODEL_CARD.md` describes inputs, selection, measured results, false-positive/negative tradeoffs, limitations, and intended use. |
| P2 | Experiment reproducibility | Associate an artifact with its training conditions | Metrics include dependency versions, dataset hash, and configuration details sufficient to identify the run. |
| P2 | Error analysis | Guide improvements using validation evidence | A documented process inspects validation mistakes without using test mistakes for repeated model tuning. |
| P3 | Fresh external evaluation | Measure generalization beyond the development corpus | A separately sourced, appropriately licensed dataset is evaluated with a frozen model and the collection process and results are documented. |
| P3 | Optional user interface | Let someone try the classifier without CLI commands | A separately scoped UI uses the saved pipeline and includes clear failure handling and usage instructions. |

P1 items improve publication quality and maintainability. P2 items strengthen the project as a portfolio artifact. P3 items extend its capabilities and evidence.

## Documentation to create or update

| Document | Action | Required content |
| --- | --- | --- |
| `PROJECT_PLAN.md` | Created | Scope, inventory, priorities, and completion criteria. |
| `README.md` | Update after implementation | Supported Python version, verified clean-install commands, links to data/model documentation, and CI instructions or badge once CI exists. |
| `docs/DATA.md` | Created | CSV schema and example; UCI attribution and license; duplicate normalization; roles of teaching, diagnostic, and evaluation data; handling of private messages. |
| `docs/MODEL_CARD.md` | Created; provenance improvements remain | Model purpose, preprocessing, candidates and selection rule, dataset/splits, recorded results, intended use, limitations, and retraining commands. |
| `LICENSE` | Created: MIT | License for original project code, without implying ownership of third-party data. |

Separate architecture, deployment, API, and contribution guides are unnecessary for the current small CLI. Add them if the project grows to need them.

## Evaluation issue to resolve

`reports/fresh_synthetic_metrics.json` reports 60 examples with accuracy 0.667, spam precision 0.857, spam recall 0.400, and spam F1 0.545. Its stated limitation is that this is an AI-authored synthetic challenge and is not representative real-world evidence.

Before presenting these numbers as reproducible results, identify the exact evaluation inputs, model artifact/configuration, and command that generated them. The filename and an empty `overlapping_ids` list alone do not establish independent evaluation. These results also show why claims of universal accuracy or production readiness would be inappropriate.

## Suggested implementation order

1. [x] Record the working Python and dependency versions and verify a clean installation. Completed with Python 3.14.6 on macOS using `requirements-lock.txt`; installation, three tests, training, and saved-model prediction passed on October 16, 2026.
2. [x] Add meaningful input/CLI tests and configure CI to run them. All 10 tests pass locally; the first GitHub Actions run is pending publication.
3. Audit report provenance and write `docs/DATA.md`.
4. Generate traceable evaluation results and write `docs/MODEL_CARD.md`.
5. Choose a code license, add `LICENSE`, and update README links and setup instructions.
6. Review the final tracked-file list, then initialize, commit, and push when requested.

## Publication checklist

- [x] A clean environment can install, test, train, and predict using documented commands (Python 3.14.6 on macOS).
- [ ] All automated checks pass.
- [ ] Reported metrics have traceable inputs and generation instructions.
- [x] Data sources and file purposes are documented.
- [x] The owner has selected MIT; the root LICENSE and README notice are added.
- [ ] No credentials, private messages, virtual environments, or generated model binaries are included unintentionally.
- [ ] README links and commands match the final repository contents.

This checklist tracks future work. Creating this plan does not implement the unchecked items.
