# Data guide

## Files and purposes

| File | Purpose |
| --- | --- |
| `data/messages.csv` | Default training and evaluation corpus, converted from the UCI SMS Spam Collection. |
| `data/sms+spam+collection.zip` | Retained source archive. |
| `data/UCI_README.txt` | Original corpus documentation and attribution. |
| `data/teaching_messages.csv` | Original small synthetic teaching dataset; useful for learning the workflow. |
| `data/diagnostic_messages.csv` | Known examples for manually inspecting behavior; not used by default training. |
| `reports/fresh_synthetic_predictions.csv` | Saved predictions for 60 AI-authored challenge messages; a report, not a training CSV. |

Paths in this table are relative to the repository root.

## Training CSV format

Use UTF-8 CSV with `label` and `message` columns:

```csv
label,message
ham,Can we meet tomorrow?
spam,Claim your cash prize now
```

Quote messages containing commas using standard CSV quoting. Labels are `ham` (legitimate) or `spam`. The loader strips surrounding whitespace and lowercases labels. It rejects invalid labels and empty messages. At least five unique messages per class are required by the loader; this minimum supports the splitting workflow, not a reliable evaluation.

For duplicate detection, the loader lowercases message text and collapses whitespace. Matching normalized text is kept once; conflicting labels raise an error. This normalization is a duplicate key, not a replacement for the original message text. It does not detect all near-duplicates or paraphrases.

## Default corpus and splits

The retained corpus documentation describes 5,574 English SMS messages: 4,827 ham and 747 spam. The current project records 5,159 unique normalized messages after deduplication.

The saved run uses 3,095 training, 1,032 validation, and 1,032 test examples. Splits are stratified, with test seed 42 and validation seed 43. Candidate vectorizers fit on training text only. After selection, the winning pipeline fits on training plus validation before test evaluation.

The corpus was inspected in earlier experiments. These splits are exploratory evidence, not a substitute for a fresh external evaluation.

## Source and attribution

The dataset is the UCI SMS Spam Collection by Tiago Almeida and Jose Maria Gomez Hidalgo (2011), DOI `10.24432/C5CC84`. The project README links to the [UCI dataset record](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) and identifies its distribution as CC BY 4.0. Preserve the original corpus documentation and source attribution when redistributing the data. A project code license is separate from third-party dataset terms.

## Using other data

Keep your original corpus intact and point training at a separate CSV:

```sh
python main.py train --data path/to/messages.csv --model models/custom_classifier.joblib
```

Record the source, permission to use it, collection method, class counts, and preprocessing. Do not publish private messages. Keep diagnostic examples distinct from evaluation evidence; once examples influence model choices, they are development data.

The synthetic challenge report has no recorded model identity or generation command. Its predictions can illustrate errors, but are not yet a reproducible external benchmark.
