"""Offline tests for data-package source fallback ordering."""

import hashlib
import io
import unittest
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import reproduce
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

    def test_corrupt_data_cache_is_removed_and_replaced(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("data/proof.txt", "ok")
        valid_bytes = buffer.getvalue()
        expected_sha256 = hashlib.sha256(valid_bytes).hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            root = (Path(directory) / "源码").resolve()
            cache_dir = root / ".data_packages"
            cache_dir.mkdir(parents=True)
            target = cache_dir / reproduce.DATA_PACKAGE_FILENAME
            target.write_bytes(b"damaged cache")
            download_paths = []

            def fake_run(command, stdout):
                self.assertEqual(command[command.index("--output") + 1], "-")
                self.assertEqual(command[command.index("--max-time") + 1], "300")
                self.assertNotIn(str(target), command)
                download_paths.append(Path(stdout.name))
                stdout.write(valid_bytes)
                return SimpleNamespace(returncode=0)

            with (
                patch("reproduce.ROOT", root),
                patch("reproduce.DATA_PACKAGE_SHA256", expected_sha256),
                patch("reproduce.shutil.which", return_value="curl"),
                patch("reproduce.subprocess.run", side_effect=fake_run),
            ):
                result = reproduce.resolve_data_package(
                    "https://example.com/cmeee_frozen_data_v1.zip"
                )

            self.assertEqual(result, target)
            self.assertEqual(target.read_bytes(), valid_bytes)
            self.assertEqual(list(cache_dir.glob("*.part")), [])
            self.assertEqual(list(cache_dir.glob("*.incoming")), [])
            self.assertFalse(download_paths[0].exists())
            self.assertNotIn(root, download_paths[0].parents)

    def test_download_partial_names_are_unique_and_cleaned_after_failure(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("data/proof.txt", "ok")
        valid_bytes = buffer.getvalue()
        expected_sha256 = hashlib.sha256(valid_bytes).hexdigest()
        partial_paths = []

        with tempfile.TemporaryDirectory() as directory:
            root = (Path(directory) / "源码").resolve()

            def fake_run(command, stdout):
                self.assertEqual(command[command.index("--output") + 1], "-")
                partial_paths.append(Path(stdout.name))
                if len(partial_paths) == 1:
                    stdout.write(b"interrupted download")
                    return SimpleNamespace(returncode=1)
                stdout.write(valid_bytes)
                return SimpleNamespace(returncode=0)

            with (
                patch("reproduce.ROOT", root),
                patch("reproduce.DATA_PACKAGE_SHA256", expected_sha256),
                patch("reproduce.shutil.which", return_value="curl"),
                patch("reproduce.subprocess.run", side_effect=fake_run),
            ):
                with self.assertRaisesRegex(RuntimeError, "curl exit 1"):
                    reproduce.resolve_data_package(
                        "https://example.com/cmeee_frozen_data_v1.zip"
                    )
                cache_dir = root / ".data_packages"
                self.assertEqual(list(cache_dir.glob("*.part")), [])
                self.assertFalse(partial_paths[0].exists())

                target = reproduce.resolve_data_package(
                    "https://example.com/cmeee_frozen_data_v1.zip"
                )

            self.assertEqual(target.read_bytes(), valid_bytes)
            self.assertEqual(len(partial_paths), 2)
            self.assertNotEqual(partial_paths[0], partial_paths[1])
            self.assertEqual(list(cache_dir.glob("*.part")), [])
            self.assertEqual(list(cache_dir.glob("*.incoming")), [])
            self.assertFalse(partial_paths[1].exists())

    def test_hash_mismatch_cleans_partial_without_publishing_target(self):
        expected_sha256 = hashlib.sha256(b"expected archive").hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()

            def fake_run(command, stdout):
                self.assertEqual(command[command.index("--output") + 1], "-")
                stdout.write(b'{"message": "rate limit exceeded"}')
                return SimpleNamespace(returncode=0)

            with (
                patch("reproduce.ROOT", root),
                patch("reproduce.DATA_PACKAGE_SHA256", expected_sha256),
                patch("reproduce.shutil.which", return_value="curl"),
                patch("reproduce.subprocess.run", side_effect=fake_run),
            ):
                with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                    reproduce.resolve_data_package(
                        "https://example.com/cmeee_frozen_data_v1.zip"
                    )

            cache_dir = root / ".data_packages"
            self.assertFalse((cache_dir / reproduce.DATA_PACKAGE_FILENAME).exists())
            self.assertEqual(list(cache_dir.glob("*.part")), [])
            self.assertEqual(list(cache_dir.glob("*.incoming")), [])

    def test_checkpoint_download_uses_safe_transfer_under_non_ascii_root(self):
        valid_bytes = b"checkpoint payload"
        expected_sha256 = hashlib.sha256(valid_bytes).hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            root = (Path(directory) / "源码").resolve()
            target = root / ".checkpoints" / "checkpoint.pt"
            download_paths = []

            def fake_run(command, stdout):
                self.assertEqual(command[command.index("--output") + 1], "-")
                self.assertEqual(command[command.index("--max-time") + 1], "1800")
                self.assertNotIn(str(target), command)
                download_paths.append(Path(stdout.name))
                stdout.write(valid_bytes)
                return SimpleNamespace(returncode=0)

            with (
                patch("reproduce.ROOT", root),
                patch("reproduce.shutil.which", return_value="curl"),
                patch("reproduce.subprocess.run", side_effect=fake_run),
            ):
                result = reproduce.resolve_bert_checkpoint(
                    "https://example.com/checkpoint.pt",
                    {"filename": "checkpoint.pt", "sha256": expected_sha256},
                )

            self.assertEqual(result, target)
            self.assertEqual(target.read_bytes(), valid_bytes)
            self.assertEqual(list(target.parent.glob("*.part")), [])
            self.assertEqual(list(target.parent.glob("*.incoming")), [])
            self.assertFalse(download_paths[0].exists())
            self.assertNotIn(root, download_paths[0].parents)


if __name__ == "__main__":
    unittest.main()
