"""Checks for the boundaries that keep evaluation meaningful."""
import unittest

from main import candidates, load_data, ROOT, scores, split_data


class TrainingTests(unittest.TestCase):
    def test_splits_are_disjoint_repeatable_and_stratified(self):
        messages, labels = load_data(ROOT / 'data/messages.csv')
        split = split_data(messages, labels)
        self.assertEqual(split, split_data(messages, labels))
        train, validation, test, y_train, y_validation, y_test = split
        self.assertFalse(set(train) & set(validation))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(validation) & set(test))
        self.assertEqual(len(train) + len(validation) + len(test), len(messages))
        ratio = labels.count('spam') / len(labels)
        for group in (y_train, y_validation, y_test):
            self.assertAlmostEqual(group.count('spam') / len(group), ratio, delta=0.01)

    def test_candidates_do_not_learn_prediction_vocabulary(self):
        for name, model in candidates():
            with self.subTest(name=name):
                model.fit(['your ordinary lunch meeting', 'your cash reward prize'], ['ham', 'spam'])
                before = list(model.named_steps['tfidf'].get_feature_names_out())
                model.predict(['validationonlytoken testonlytoken'])
                self.assertEqual(before, list(model.named_steps['tfidf'].get_feature_names_out()))
                self.assertNotIn('testonlytoken', before)

    def test_spam_metrics_count_false_positives_and_missed_spam(self):
        result = scores(['ham', 'ham', 'spam', 'spam'], ['ham', 'spam', 'ham', 'spam'])
        for value in result.values():
            self.assertEqual(value, 0.5)


if __name__ == '__main__':
    unittest.main()
