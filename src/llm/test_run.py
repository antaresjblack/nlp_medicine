"""Offline checks; never calls the API."""
import unittest
from src.llm.run import normalize
import json


class NormalizeTests(unittest.TestCase):
    def extract(self, text, entities):
        return normalize(text, json.dumps({'entities': entities}, ensure_ascii=False))

    def test_nested_and_duplicate(self):
        a = dict(entity='肺炎', type='dis', start_idx=0, end_idx=1)
        b = dict(entity='肺', type='bod', start_idx=0, end_idx=0)
        out, _ = self.extract('肺炎', [a, b, a])
        self.assertEqual(len(out), 2)

    def test_unique_repair(self):
        out, issues = self.extract('出现发热', [dict(entity='发热', type='sym', start_idx=0, end_idx=1)])
        self.assertEqual(out[0]['start_idx'], 2)
        self.assertEqual(issues[0]['reason'], 'unique_offset_repair')

    def test_ambiguous_missing_unknown(self):
        out, issues = self.extract('痛痛', [dict(entity='痛', type='sym'),
            dict(entity='发热', type='sym'), dict(entity='痛', type='invalid')])
        self.assertEqual(out, [])
        self.assertEqual(len(issues), 3)

    def test_ambiguous_with_valid_anchor_uses_nearest(self):
        out, issues = self.extract('痛痛', [
            dict(entity='痛', type='sym', start_idx=2, end_idx=2),
        ])
        self.assertEqual(out[0]['start_idx'], 1)
        self.assertEqual(issues[0]['reason'], 'ambiguous_position_nearest_offset')

    def test_invalid_json_and_empty(self):
        with self.assertRaises(ValueError):
            normalize('文本', 'not json')
        self.assertEqual(self.extract('文本', [])[0], [])


if __name__ == '__main__':
    unittest.main()
