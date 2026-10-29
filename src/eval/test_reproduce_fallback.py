"""Offline tests for data-package source fallback ordering."""

import unittest
from pathlib import Path
import sys
from unittest.mock import patch

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reproduce import (
    DEFAULT_DATA_PACKAGE,
    DEFAULT_DATA_PACKAGE_FALLBACK,
    data_package_sources,
    prepare_required_data,
)


class DataPackageFallbackTests(unittest.TestCase):
    def test_explicit_source_has_no_fallback(self):
        self.assertEqual(data_package_sources("https://example.com/data.zip"), [
            "https://example.com/data.zip"
        ])

    def test_default_source_order(self):
        self.assertEqual(
            data_package_sources(None),
            [DEFAULT_DATA_PACKAGE, DEFAULT_DATA_PACKAGE_FALLBACK],
        )

    def test_falls_back_after_primary_failure(self):
        calls = []

        def fake_ensure(source):
            calls.append(source)
            if source == DEFAULT_DATA_PACKAGE:
                raise RuntimeError("primary unavailable")
            return {"status": "downloaded", "package": source}

        with patch("reproduce.ensure_required_data", side_effect=fake_ensure):
            result = prepare_required_data(None)
        self.assertEqual(result["package"], DEFAULT_DATA_PACKAGE_FALLBACK)
        self.assertEqual(calls, [DEFAULT_DATA_PACKAGE, DEFAULT_DATA_PACKAGE_FALLBACK])


if __name__ == "__main__":
    unittest.main()
