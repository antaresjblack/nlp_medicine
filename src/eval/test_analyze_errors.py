"""Offline tests for strict entity error classification."""

import unittest

try:
    from .analyze_errors import analyze, classify_record
except ImportError:
    from analyze_errors import analyze, classify_record


class ErrorAnalysisTests(unittest.TestCase):
    def test_classifies_four_categories(self):
        gold = {
            "id": "test-case",
            "text": "肺炎发热胃痛",
            "entities": [
                {"start_idx": 0, "end_idx": 1, "type": "dis", "entity": "肺炎"},
                {"start_idx": 2, "end_idx": 3, "type": "sym", "entity": "发热"},
                {"start_idx": 4, "end_idx": 5, "type": "bod", "entity": "胃痛"},
            ],
        }
        predicted = {
            "id": "test-case",
            "text": gold["text"],
            "entities": [
                {"start_idx": 0, "end_idx": 1, "type": "sym", "entity": "肺炎"},
                {"start_idx": 2, "end_idx": 2, "type": "sym", "entity": "发"},
                {"start_idx": 4, "end_idx": 6, "type": "ite", "entity": "胃痛"},
            ],
        }
        cases = classify_record(gold, predicted, 0)
        categories = [case["category"] for case in cases]
        self.assertEqual(categories.count("type_error"), 1)
        self.assertEqual(categories.count("boundary_error"), 1)
        self.assertEqual(categories.count("missed"), 1)
        self.assertEqual(categories.count("false_positive"), 1)
        report = analyze(
            [
                {
                    "record_index": 0,
                    "predicted_entities": predicted["entities"],
                }
            ],
            [gold],
            "test",
        )
        self.assertEqual(report["total_error_events"], 4)
        self.assertEqual(report["gold_error_entities"], 3)
        self.assertEqual(report["prediction_error_entities"], 3)
        self.assertEqual(
            report["pairing_identity"]["left"],
            report["pairing_identity"]["right"],
        )

    def test_report_limits_examples(self):
        gold = [
            {
                "id": f"case-{index}",
                "text": "痛",
                "entities": [
                    {"start_idx": 0, "end_idx": 0, "type": "sym", "entity": "痛"}
                ],
            }
            for index in range(5)
        ]
        errors = [
            {
                "record_index": index,
                "predicted_entities": [],
                "missing": [[0, 0, "sym"]],
                "extra": [],
            }
            for index in range(5)
        ]
        report = analyze(errors, gold, "test")
        self.assertEqual(report["categories"]["missed"]["count"], 5)
        self.assertEqual(len(report["examples"]["missed"]), 3)


if __name__ == "__main__":
    unittest.main()
