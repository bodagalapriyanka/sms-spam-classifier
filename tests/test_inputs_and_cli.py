"""Exercise invalid inputs and the public train/save/load/predict workflow."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from main import ROOT, load_data


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'messages.csv'
        self.rows = [(label, f'{label} example {i}')
                     for label in ('ham', 'spam') for i in range(5)]

    def write_rows(self, rows):
        with self.path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['label', 'message'])
            writer.writerows(rows)

    def test_duplicate_normalization_preserves_first_message(self):
        self.write_rows(self.rows + [('ham', '  HAM   EXAMPLE 0  ')])
        messages, labels = load_data(self.path)
        self.assertEqual(len(messages), 10)
        self.assertEqual(messages[0], 'ham example 0')
        self.assertEqual(labels.count('ham'), 5)

    def test_conflicting_duplicate_is_rejected(self):
        self.write_rows(self.rows + [('spam', 'HAM EXAMPLE 0')])
        with self.assertRaisesRegex(ValueError, 'Conflicting labels'):
            load_data(self.path)

    def test_invalid_label_and_empty_message_are_rejected(self):
        for row in [('unknown', 'some text'), ('ham', '   ')]:
            with self.subTest(row=row):
                self.write_rows(self.rows + [row])
                with self.assertRaisesRegex(ValueError, 'Invalid row'):
                    load_data(self.path)

    def test_missing_columns_are_rejected(self):
        self.path.write_text('category,text\nham,hello\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'label and message columns'):
            load_data(self.path)

    def test_insufficient_unique_examples_are_rejected(self):
        self.write_rows(self.rows[:-1])
        with self.assertRaisesRegex(ValueError, 'at least five unique'):
            load_data(self.path)


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / 'main.py'), *map(str, args)],
                              capture_output=True, text=True, timeout=120)

    def test_missing_model_and_missing_message_have_actionable_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / 'missing.joblib'
            for args, expected in [
                (('predict', 'hello', '--model', missing), 'Model not found'),
                (('predict',), 'Provide a message'),
            ]:
                with self.subTest(args=args):
                    result = self.run_cli(*args)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(expected, result.stderr)

    def test_train_save_load_predict_and_blank_input(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model.joblib'
            result = self.run_cli('train', '--data', ROOT / 'data/teaching_messages.csv',
                                  '--model', model)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(model.is_file())
            report = json.loads(model.with_suffix('.metrics.json').read_text())
            self.assertEqual(sum(map(sum, report['confusion_matrix_ham_spam'])),
                             report['split_sizes']['test'])
            for message in ['Can we meet for lunch?', 'Claim your cash prize now']:
                predicted = self.run_cli('predict', message, '--model', model)
                self.assertEqual(predicted.returncode, 0, predicted.stderr)
                self.assertRegex(predicted.stdout, r'^(NOT SPAM|SPAM): ')
                self.assertIn(message, predicted.stdout)
            blank = self.run_cli('predict', '   ', '--model', model)
            self.assertEqual(blank.returncode, 2)
            self.assertIn('nonempty message', blank.stderr)


if __name__ == '__main__':
    unittest.main()
