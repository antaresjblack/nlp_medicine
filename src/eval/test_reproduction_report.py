"""Offline test for the four-method reproduction report."""

import unittest
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reproduction_report import build_reproduction_report


class ReproductionReportTests(unittest.TestCase):
    def test_report_contains_all_four_eval_test_methods(self):
        summary = {
            "status": "pass",
            "repository_commit": "test-commit",
            "working_tree_dirty": False,
            "started_at": "2026-10-29T00:00:00+08:00",
            "finished_at": "2026-10-29T00:01:40+08:00",
            "elapsed_seconds": 100.0,
            "environment": {
                "python": "3.10",
                "torch": "2.9",
                "transformers": "4.57",
                "numpy": "1.26",
                "scipy": "1.13",
                "scikit_learn": "1.7",
                "cuda_device": "test-device",
            },
            "data_hashes": {
                "data/data_cleaned/CMeEE_train_cleaned_final.json": "train",
                "data/data_raw/CMeEE_dev.json": "dev",
                "data/data_eval/eval_tune.jsonl": "tune",
                "data/data_eval/eval_test.jsonl": "test",
            },
            "data_preparation": {"status": "present"},
            "frozen_small_model_config": {
                "selected_candidate": {
                    "C": 0.1,
                    "class_weight": "none",
                    "dict_span_features": True,
                },
                "settings": {"seed": 42},
            },
            "bert_variant": {"label": "MacBERT + GlobalPointer + 词典特征"},
            "pure_rules": {
                "metrics_file": "outputs/eval_split/metrics/test_pure_rules_metrics.json"
            },
            "dictionary_baseline": {
                "metrics_file": (
                    "outputs/eval_split/metrics/"
                    "test_dictionary_longest_min_count_1_min_length_2_metrics.json"
                )
            },
            "small_model": {
                "metrics_file": "outputs/eval_split/small_model/test/test_metrics.json",
                "error_analysis_md": "small-error.md",
            },
            "bert_execution": {
                "metrics_file": "outputs/bert_gp/test_macbert_dict_5ep_v1/metrics.json",
                "manifest": "outputs/bert_gp/test_macbert_dict_5ep_v1/manifest.json",
                "checkpoint": "checkpoint.pt",
                "checkpoint_sha256": "checkpoint-sha",
                "error_analysis_md": "bert-error.md",
            },
            "output_dir": "reproduction_outputs/test",
            "report_file": "reproduction_outputs/test/reproduction_report.txt",
        }
        data_scope = {
            "train": {"records": 1, "entities": 1},
            "dev": {"records": 1, "entities": 1},
            "eval_tune": {"records": 1, "entities": 1},
            "eval_test": {"records": 2500, "entities": 10460},
        }

        report = build_reproduction_report(summary, data_scope)

        names = (
            "纯人工规则（9 条正则）",
            "最长匹配词典（过滤单字）",
            "字符 LinearSVC + 词典特征",
            "MacBERT + GlobalPointer + 词典特征",
        )
        results = report.split("[3] 冻结 eval_test 四方法结果", 1)[1]
        results = results.split("[4] 九类实体严格 F1", 1)[0]
        positions = [results.index(name) for name in names]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("[3] 冻结 eval_test 四方法结果", report)


if __name__ == "__main__":
    unittest.main()
