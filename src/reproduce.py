"""One-command reproduction entry for the submitted CMeEE experiment.

By default this script:
1. verifies the frozen data and selected small-model configuration;
2. runs the frozen handwritten-rule and longest-match dictionary baselines;
3. retrains the frozen character LinearSVC and evaluates it on eval_test;
4. downloads or reuses the pinned MacBERT dictionary-fusion checkpoint;
5. runs final BERT inference on eval_test and compares all four results.

Use ``--check-only`` to verify the environment, data, frozen configuration and
committed evidence without training or running either model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from reproduction_report import build_reproduction_report, collect_data_scope

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
SELECTED_CONFIG = (
    ROOT
    / "outputs"
    / "eval_split"
    / "small_model"
    / "tune"
    / "runs"
    / "20261022T141507+0800"
    / "selected_config.json"
)
BERT_VARIANTS = {
    "dict_5ep": {
        "run_name": "macbert_dict_5ep_v1",
        "label": "MacBERT + GlobalPointer + 词典特征",
        "filename": "macbert_dict_5ep_v1.pt",
        "sha256": "595eb45cbfd920f93e175668a7fe50a11f8b949222e882bb2f7fd4c91ae9b1d8",
        "url": (
            "https://github.com/antaresjblack/nlp_medicine/releases/download/"
            "macbert-dict-5ep-v1/macbert_dict_5ep_v1.pt"
        ),
        "fallback_url": (
            "https://api.github.com/repos/antaresjblack/nlp_medicine/"
            "releases/assets/684231335"
        ),
        "expected_f1": 0.6700473292765381,
        "trainer": "bert_gp/train_dict.py",
        "evidence": "test_macbert_dict_5ep_v1",
    },
    "full_v1": {
        "run_name": "full_v1",
        "label": "BERT-base-chinese + GlobalPointer（3轮）",
        "filename": "full_v1.pt",
        "sha256": "9eec198f9f590d74da93caa8da6baae4d1cbb7bd40ba59538c654dadcbb7cd84",
        "url": (
            "https://github.com/antaresjblack/nlp_medicine/releases/download/"
            "bert-legacy-checkpoints-v1/full_v1.pt"
        ),
        "fallback_url": (
            "https://api.github.com/repos/antaresjblack/nlp_medicine/"
            "releases/assets/684473140"
        ),
        "expected_f1": 0.6377233070211882,
        "trainer": "bert_gp/train.py",
        "evidence": "test_full_v1",
    },
    "macbert_base_v1": {
        "run_name": "macbert_base_v1",
        "label": "MacBERT + GlobalPointer（3轮）",
        "filename": "macbert_base_v1.pt",
        "sha256": "3c6affab84a54ada0d01b2c2fce1f0cab927319770deeb8580880d3e05857fe0",
        "url": (
            "https://github.com/antaresjblack/nlp_medicine/releases/download/"
            "bert-legacy-checkpoints-v1/macbert_base_v1.pt"
        ),
        "fallback_url": (
            "https://api.github.com/repos/antaresjblack/nlp_medicine/"
            "releases/assets/684476265"
        ),
        "expected_f1": 0.6570264360962036,
        "trainer": "bert_gp/train.py",
        "evidence": "test_macbert_base_v1",
    },
    "macbert_warmup_v1": {
        "run_name": "macbert_warmup_v1",
        "label": "MacBERT + GlobalPointer（3轮 + warmup）",
        "filename": "macbert_warmup_v1.pt",
        "sha256": "f74dbfea85b8d9b88676013b144906be105b687e43e58b2143c7ec5992677047",
        "url": (
            "https://github.com/antaresjblack/nlp_medicine/releases/download/"
            "bert-legacy-checkpoints-v1/macbert_warmup_v1.pt"
        ),
        "fallback_url": (
            "https://api.github.com/repos/antaresjblack/nlp_medicine/"
            "releases/assets/684476337"
        ),
        "expected_f1": 0.6621921729260228,
        "trainer": "bert_gp/train.py",
        "evidence": "test_macbert_warmup_v1",
    },
    "macbert_warmup_5ep_v1": {
        "run_name": "macbert_warmup_5ep_v1",
        "label": "MacBERT + GlobalPointer（5轮 + warmup）",
        "filename": "macbert_warmup_5ep_v1.pt",
        "sha256": "868fbeb8530125552ff06ef8ca4c29f9cfad8b25f087be1e1d61687b41038cf7",
        "url": (
            "https://github.com/antaresjblack/nlp_medicine/releases/download/"
            "macbert-warmup-5ep-v1/macbert_warmup_5ep_v1.pt"
        ),
        "fallback_url": (
            "https://api.github.com/repos/antaresjblack/nlp_medicine/"
            "releases/assets/683593804"
        ),
        "expected_f1": 0.6670225673380248,
        "trainer": "bert_gp/train.py",
        "evidence": "test_macbert_warmup_5ep_v1",
    },
}
DEFAULT_BERT_VARIANT = "dict_5ep"
LOCAL_BERT_MODEL_DIR = ROOT / "bert_gp" / "pretrained_config"
LOCAL_BERT_MODEL_FILES = (
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
)

EXPECTED_DATA_HASHES = {
    "data/data_cleaned/CMeEE_train_cleaned_final.json": "e0e0d1bba812040e8e7f3e1096234e361db7ccc048db9575943fcfd9b3eb86dc",
    "data/data_raw/CMeEE_dev.json": "bbe6be016bdb1a418362507024902b70587b951a884cf4a96a6d1246822cd07c",
    "data/data_eval/eval_tune.jsonl": "ff1e9fdc685a4b405a55cfece540eceb7cba7bfdb70492693c51c6e42d0f584b",
    "data/data_eval/eval_test.jsonl": "ca65235e83f5ba055af7498b25ea55c376747a64ab6b1d4e5bffc0dfb8e62778",
}

DEFAULT_DATA_PACKAGE = (
    "https://github.com/antaresjblack/nlp_medicine/releases/download/"
    "cmeee-frozen-data-v1/cmeee_frozen_data_v1.zip"
)
DEFAULT_DATA_PACKAGE_FALLBACK = (
    "https://api.github.com/repos/antaresjblack/nlp_medicine/releases/assets/684705120"
)
DATA_PACKAGE_FILENAME = "cmeee_frozen_data_v1.zip"
DATA_PACKAGE_SHA256 = "b9e48b664ab0135892e688db2ffb25df6d6a1a726fabdd6b5fd0a1611a41cfef"

SMALL_MODEL_EXPECTED_F1 = 0.4289685582822086
PURE_RULES_EXPECTED_F1 = 0.04915930464519806
DICTIONARY_EXPECTED_F1 = 0.407671721677074


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_https_asset(
    value: str,
    target: Path,
    expected_sha256: str,
    asset_name: str,
    max_time_seconds: int,
) -> Path:
    """Download to a system temp file, verify it, then publish atomically."""
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{target.stem}.",
        suffix=".part",
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    incoming = target.with_name(
        f"{target.name}.{os.getpid()}.{uuid.uuid4().hex}.incoming"
    )
    try:
        print(f"[INFO] Downloading {asset_name}: {value}", flush=True)
        curl = shutil.which("curl")
        if curl:
            command = [
                curl,
                "-L",
                "--http1.1",
                "--retry",
                "5",
                "--retry-delay",
                "2",
                "--retry-all-errors",
                "--connect-timeout",
                "30",
                "--max-time",
                str(max_time_seconds),
                "--speed-limit",
                "1024",
                "--speed-time",
                "60",
                "--silent",
                "--show-error",
                "--fail",
                "--continue-at",
                "-",
                "--output",
                str(temporary),
                "--header",
                "Accept: application/octet-stream",
                "--header",
                "User-Agent: Codex-CMeEE-Reproduction",
                value,
            ]
            completed = subprocess.run(command)
            require(
                completed.returncode == 0,
                f"Failed to download {asset_name} (curl exit {completed.returncode})",
            )
        else:
            request = Request(
                value,
                headers={
                    "User-Agent": "Codex-CMeEE-Reproduction",
                    "Accept": "application/octet-stream",
                },
            )
            with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

        digest = sha256_file(temporary)
        require(
            digest == expected_sha256,
            f"Downloaded {asset_name} hash mismatch: expected {expected_sha256}, got {digest}",
        )

        # The source may be on another drive. Copy on the destination drive first,
        # verify the copy, then use os.replace for an atomic same-volume publish.
        shutil.copyfile(temporary, incoming)
        require(
            sha256_file(incoming) == expected_sha256,
            f"Copied {asset_name} hash mismatch after publication staging",
        )
        os.replace(incoming, target)
        print(f"[PASS] Downloaded and verified {asset_name}: {display_path(target)}")
        return target
    finally:
        for path in (temporary, incoming):
            if path.exists():
                path.unlink()


def resolve_bert_checkpoint(value: str, variant: dict[str, Any]) -> Path:
    """Resolve a local checkpoint or download the pinned checkpoint from HTTPS."""
    parsed = urlparse(value)
    expected_sha256 = str(variant["sha256"])
    if parsed.scheme in {"http", "https"}:
        require(parsed.scheme == "https", "BERT checkpoint URL must use HTTPS")
        filename = str(variant["filename"])
        cache_dir = ROOT / ".checkpoints"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / filename
        if target.is_file():
            digest = sha256_file(target)
            if digest == expected_sha256:
                print(f"[INFO] Using cached checkpoint: {display_path(target)}")
                return target
            print(
                "[WARN] Cached checkpoint hash mismatch: "
                f"expected {expected_sha256}, got {digest}; removing it."
            )
            target.unlink()

        return download_https_asset(
            value,
            target,
            expected_sha256,
            asset_name="checkpoint",
            max_time_seconds=1800,
        )

    path = Path(value).expanduser().resolve()
    require(path.is_file(), f"BERT checkpoint not found: {path}")
    digest = sha256_file(path)
    require(
        digest == expected_sha256,
        f"BERT checkpoint hash mismatch: expected {expected_sha256}, got {digest}",
    )
    return path


def prepare_bert_checkpoint(value: str | None, variant: dict[str, Any]) -> tuple[Path, str]:
    """Resolve an explicit/local checkpoint or fall back from Release to API."""
    if value is not None:
        return resolve_bert_checkpoint(value, variant), value

    local_checkpoint = ROOT.parent / str(variant["filename"])
    if local_checkpoint.is_file():
        return resolve_bert_checkpoint(str(local_checkpoint), variant), str(local_checkpoint)

    sources = [str(variant["url"])]
    fallback_url = variant.get("fallback_url")
    if fallback_url:
        sources.append(str(fallback_url))

    last_error: Exception | None = None
    for index, source in enumerate(sources):
        try:
            return resolve_bert_checkpoint(source, variant), source
        except RuntimeError as error:
            last_error = error
            if index + 1 < len(sources):
                print(f"[WARN] BERT checkpoint source failed: {error}")
                print("[INFO] Retrying with the GitHub Release Asset API.", flush=True)
    if last_error is not None:
        raise last_error
    raise RuntimeError("No BERT checkpoint source was available")


def resolve_bert_model_dir(value: Path | None) -> Path | None:
    """Use an explicit model directory or the bundled config/tokenizer files."""
    path = value if value is not None else LOCAL_BERT_MODEL_DIR
    if value is None and not path.is_dir():
        return None
    path = path.expanduser().resolve()
    require(path.is_dir(), f"BERT model directory not found: {path}")
    missing = [name for name in LOCAL_BERT_MODEL_FILES if not (path / name).is_file()]
    require(
        not missing,
        f"BERT model directory is missing required files: {', '.join(missing)}",
    )
    print(f"[PASS] Using local BERT config/tokenizer: {display_path(path)}")
    return path


def resolve_data_package(value: str) -> Path:
    """Resolve or download the pinned frozen-data archive."""
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        require(parsed.scheme == "https", "Data package URL must use HTTPS")
        cache_dir = ROOT / ".data_packages"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / DATA_PACKAGE_FILENAME
        if target.is_file():
            digest = sha256_file(target)
            if digest == DATA_PACKAGE_SHA256:
                print(f"[INFO] Using cached data package: {display_path(target)}")
                return target
            print(
                "[WARN] Cached data package hash mismatch: "
                f"expected {DATA_PACKAGE_SHA256}, got {digest}; removing it."
            )
            target.unlink()

        return download_https_asset(
            value,
            target,
            DATA_PACKAGE_SHA256,
            asset_name="frozen data package",
            max_time_seconds=300,
        )

    path = Path(value).expanduser().resolve()
    require(path.is_file(), f"Data package not found: {path}")
    digest = sha256_file(path)
    require(
        digest == DATA_PACKAGE_SHA256,
        f"Data package hash mismatch: expected {DATA_PACKAGE_SHA256}, got {digest}",
    )
    return path


def ensure_required_data(package_source: str, root: Path = ROOT) -> dict[str, Any]:
    """Download and extract the frozen dataset only when required files are absent."""
    missing = [
        relative
        for relative in EXPECTED_DATA_HASHES
        if not (root / relative).is_file()
    ]
    if not missing:
        print("[PASS] Frozen data files are already present; download was not needed.")
        return {"status": "present", "package": None, "sha256": None}

    print("[INFO] Missing frozen data files:")
    for relative in missing:
        print(f"       - {relative}")
    package = resolve_data_package(package_source)
    with zipfile.ZipFile(package) as archive:
        for info in archive.infolist():
            relative = Path(info.filename)
            require(
                not relative.is_absolute() and ".." not in relative.parts,
                f"Unsafe path in data package: {info.filename}",
            )
            require(
                relative.parts and relative.parts[0] == "data",
                f"Unexpected top-level path in data package: {info.filename}",
            )
            target = (root / relative).resolve()
            require(
                target == root or root in target.parents,
                f"Unsafe extraction target: {target}",
            )
        archive.extractall(root)
    print(f"[PASS] Extracted frozen data package to: {display_path(root / 'data')}")
    return {
        "status": "downloaded",
        "package": str(package),
        "sha256": DATA_PACKAGE_SHA256,
    }


def data_package_sources(explicit_source: str | None) -> list[str]:
    if explicit_source is not None:
        return [explicit_source]
    return [DEFAULT_DATA_PACKAGE, DEFAULT_DATA_PACKAGE_FALLBACK]


def prepare_required_data(explicit_source: str | None) -> dict[str, Any]:
    sources = data_package_sources(explicit_source)
    last_error: Exception | None = None
    for index, source in enumerate(sources):
        try:
            return ensure_required_data(source)
        except (RuntimeError, zipfile.BadZipFile) as error:
            last_error = error
            if index + 1 < len(sources):
                print(f"[WARN] Data package source failed: {error}")
                print("[INFO] Retrying with the GitHub Release Asset API.", flush=True)
    if last_error is not None:
        raise last_error
    raise RuntimeError("No data package source was available")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def display_path(path: Path) -> str:
    """Return a source-root-relative path, or an absolute path for external output."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_working_tree_dirty() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def verify_environment() -> dict[str, Any]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    modules = (
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("sklearn", "scikit_learn"),
        ("torch", "torch"),
        ("transformers", "transformers"),
    )
    for module_name, key in modules:
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            raise RuntimeError(
                f"Missing dependency {module_name}. Run: "
                "python -m pip install -r requirements.txt"
            ) from error
        versions[key] = module.__version__
    torch = importlib.import_module("torch")
    versions["cuda_available"] = torch.cuda.is_available()
    if versions["cuda_available"]:
        versions["cuda_device"] = torch.cuda.get_device_name(0)
    return versions


def verify_data_hashes() -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, expected in EXPECTED_DATA_HASHES.items():
        path = ROOT / relative
        require(path.exists(), f"Missing required data file: {relative}")
        digest = sha256_file(path)
        require(digest == expected, f"Data hash mismatch: {relative}\nexpected={expected}\nactual={digest}")
        actual[relative] = digest
    return actual


def resolve_frozen_path(value: str) -> Path:
    """Resolve new runtime paths and legacy paths stored in frozen evidence."""
    path = Path(value)
    if path.is_absolute():
        return path
    direct = ROOT / path
    if direct.exists():
        return direct
    if path.parts and path.parts[0] in {"data_raw", "data_cleaned", "data_eval"}:
        migrated = ROOT / "data" / path
        if migrated.exists():
            return migrated
    return direct


def verify_frozen_config() -> dict[str, Any]:
    frozen = load_json(SELECTED_CONFIG)
    require(frozen.get("status") == "frozen", "Small-model configuration is not frozen")
    require(frozen.get("selection_split") == "eval_tune", "Unexpected small-model selection split")
    require(frozen.get("settings", {}).get("seed") == 42, "Small-model seed is not 42")

    files = frozen["files"]
    hashes = frozen["hashes"]
    checks = (
        (files["train"], hashes["train"]),
        (files["tune"], hashes["tune"]),
        (frozen["source"]["candidate_summary"], hashes["tune_summary"]),
        ("configs/small_model_split_candidates.json", hashes["candidate_config"]),
    )
    for relative, expected in checks:
        path = resolve_frozen_path(relative)
        require(path.exists(), f"Frozen-config dependency is missing: {relative}")
        actual = sha256_file(path)
        require(actual == expected, f"Frozen-config hash mismatch: {relative}")
    test_path = resolve_frozen_path(files["test"])
    expected_test_hash = EXPECTED_DATA_HASHES["data/data_eval/eval_test.jsonl"]
    require(
        sha256_file(test_path) == expected_test_hash,
        "Test split hash mismatch",
    )
    return frozen


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(str(part) for part in command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    repository_root = str(ROOT.parent)
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        repository_root
        if not existing_python_path
        else repository_root + os.pathsep + existing_python_path
    )
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def compare_f1(name: str, actual: float, expected: float, tolerance: float) -> bool:
    difference = abs(actual - expected)
    passed = difference <= tolerance
    print(
        f"[{'PASS' if passed else 'FAIL'}] {name}: actual={actual:.6f}, "
        f"expected={expected:.6f}, |delta|={difference:.6f}, tolerance={tolerance:.6f}"
    )
    return passed


def generate_error_analysis(
    errors_path: Path, output_dir: Path, method: str
) -> tuple[Path, Path]:
    require(errors_path.exists(), f"Error records are missing: {errors_path}")
    output_json = output_dir / "error_analysis.json"
    output_md = output_dir / "error_analysis.md"
    run_command(
        [
            sys.executable,
            "eval/analyze_errors.py",
            "--errors",
            str(errors_path),
            "--gold",
            str(ROOT / "data" / "data_eval" / "eval_test.jsonl"),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--method",
            method,
        ]
    )
    return output_json, output_md


def verify_bert_evidence(variant: dict[str, Any]) -> dict[str, Any]:
    evidence_name = str(variant["evidence"])
    evidence_dir = ROOT / "outputs" / "bert_gp" / evidence_name
    metrics_path = evidence_dir / "metrics.json"
    manifest_path = evidence_dir / "manifest.json"
    require(metrics_path.exists(), f"Recorded BERT metrics are missing: {metrics_path}")
    require(manifest_path.exists(), f"Recorded BERT manifest is missing: {manifest_path}")
    metrics = load_json(metrics_path)
    manifest = load_json(manifest_path)
    require(metrics.get("metric") == "strict_entity_micro", "BERT evidence uses an unexpected metric")
    run_name = str(manifest.get("args", {}).get("run_name", ""))
    require(
        manifest.get("evaluation_split") == "eval_test" or run_name.startswith("test_"),
        "BERT evidence was not evaluated on eval_test",
    )
    training_args = manifest.get("checkpoint_training_args", manifest.get("args", {}))
    require(
        training_args.get("seed") == 42,
        "BERT training seed is not 42",
    )
    require(
        manifest.get("tune_sha256")
        == EXPECTED_DATA_HASHES["data/data_eval/eval_test.jsonl"],
        "BERT evidence does not reference the frozen eval_test hash",
    )
    return {"metrics": metrics, "manifest": manifest, "metrics_path": metrics_path, "manifest_path": manifest_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for reproduction artifacts; defaults to src/reproduction_outputs/<timestamp>",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify data hashes, frozen configuration and committed evidence; run no training.",
    )
    parser.add_argument(
        "--bert-variant",
        choices=tuple(BERT_VARIANTS),
        default=DEFAULT_BERT_VARIANT,
        help=(
            "BERT checkpoint variant to reproduce. Defaults to the latest "
            "dictionary-fusion model; select a historical variant for comparison."
        ),
    )
    parser.add_argument(
        "--bert-checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint path or HTTPS URL for the selected BERT variant. "
            "Defaults to the local pinned checkpoint, then its GitHub Release asset."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Local directory containing config.json and tokenizer files for offline BERT inference.",
    )
    parser.add_argument(
        "--data-package",
        type=str,
        default=None,
        help="Local path or HTTPS URL for the frozen CMeEE data package.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Allowed absolute F1 difference for reproduced main results; default 0.01.",
    )
    args = parser.parse_args()

    require(args.tolerance >= 0, "tolerance must be non-negative")
    variant = BERT_VARIANTS[args.bert_variant]
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir = (args.out_dir or ROOT / "reproduction_outputs" / stamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    summary: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "repository_commit": git_commit(),
        "working_tree_dirty": git_working_tree_dirty(),
        "output_dir": str(output_dir),
        "check_only": args.check_only,
        "bert_variant": {
            "id": args.bert_variant,
            "run_name": variant["run_name"],
            "label": variant["label"],
            "checkpoint_filename": variant["filename"],
        },
    }

    try:
        summary["environment"] = verify_environment()
        summary["data_preparation"] = prepare_required_data(args.data_package)
        summary["data_hashes"] = verify_data_hashes()
        data_scope = collect_data_scope()
        summary["data_scope"] = data_scope
        summary["frozen_small_model_config"] = verify_frozen_config()
        print("[PASS] Environment, required data, hashes and frozen small-model configuration.")

        bert_evidence = verify_bert_evidence(variant)
        bert_f1_pass = compare_f1(
            f"Committed {variant['run_name']} eval_test evidence",
            float(bert_evidence["metrics"]["f1"]),
            float(variant["expected_f1"]),
            1e-12,
        )
        summary["bert_evidence"] = {
            "metrics_file": display_path(bert_evidence["metrics_path"]),
            "manifest": display_path(bert_evidence["manifest_path"]),
            "f1": float(bert_evidence["metrics"]["f1"]),
            "verified": bert_f1_pass,
            "executed_here": False,
        }

        if args.check_only:
            summary["status"] = "check-only-pass" if bert_f1_pass else "failed"
            print("[INFO] Check-only mode completed; no model was trained or executed.")
        else:
            pure_rules_output = output_dir / "pure_rules"
            run_command(
                [
                    sys.executable,
                    "eval/run_pure_rule_split.py",
                    "--split",
                    "test",
                    "--output-dir",
                    str(pure_rules_output),
                ]
            )
            pure_rules_metrics_path = (
                pure_rules_output / "metrics" / "test_pure_rules_metrics.json"
            )
            require(
                pure_rules_metrics_path.exists(),
                f"Pure-rule run did not create {pure_rules_metrics_path}",
            )
            pure_rules_metrics = load_json(pure_rules_metrics_path)
            pure_rules_f1_pass = compare_f1(
                "Frozen handwritten rules eval_test",
                float(pure_rules_metrics["f1"]),
                PURE_RULES_EXPECTED_F1,
                args.tolerance,
            )
            summary["pure_rules"] = {
                "metrics_file": display_path(pure_rules_metrics_path),
                "predictions_file": display_path(
                    pure_rules_output
                    / "predictions"
                    / "test_pure_rules_predictions.jsonl"
                ),
                "errors_file": display_path(
                    pure_rules_output / "errors" / "test_pure_rules_errors.json"
                ),
                "f1": float(pure_rules_metrics["f1"]),
                "expected_f1": PURE_RULES_EXPECTED_F1,
                "verified": pure_rules_f1_pass,
                "executed_here": True,
            }

            dictionary_output = output_dir / "dictionary"
            dictionary_method = (
                "dictionary_longest_min_count_1_min_length_2"
            )
            run_command(
                [
                    sys.executable,
                    "eval/run_dictionary_split.py",
                    "--method",
                    "longest",
                    "--min-count",
                    "1",
                    "--min-length",
                    "2",
                    "--split",
                    "test",
                    "--output-dir",
                    str(dictionary_output),
                ]
            )
            dictionary_metrics_path = (
                dictionary_output
                / "metrics"
                / f"test_{dictionary_method}_metrics.json"
            )
            require(
                dictionary_metrics_path.exists(),
                f"Dictionary run did not create {dictionary_metrics_path}",
            )
            dictionary_metrics = load_json(dictionary_metrics_path)
            dictionary_f1_pass = compare_f1(
                "Frozen longest-match dictionary eval_test",
                float(dictionary_metrics["f1"]),
                DICTIONARY_EXPECTED_F1,
                args.tolerance,
            )
            summary["dictionary_baseline"] = {
                "metrics_file": display_path(dictionary_metrics_path),
                "predictions_file": display_path(
                    dictionary_output
                    / "predictions"
                    / f"test_{dictionary_method}_predictions.jsonl"
                ),
                "errors_file": display_path(
                    dictionary_output
                    / "errors"
                    / f"test_{dictionary_method}_errors.json"
                ),
                "f1": float(dictionary_metrics["f1"]),
                "expected_f1": DICTIONARY_EXPECTED_F1,
                "verified": dictionary_f1_pass,
                "executed_here": True,
            }

            small_output = output_dir / "small_model"
            run_command(
                [
                    sys.executable,
                    "eval/run_small_model_split.py",
                    "--stage",
                    "test",
                    "--selected-config",
                    str(SELECTED_CONFIG),
                    "--output-dir",
                    str(small_output),
                ]
            )
            small_metrics_path = small_output / "test" / "test_metrics.json"
            require(small_metrics_path.exists(), f"Small-model run did not create {small_metrics_path}")
            small_metrics = load_json(small_metrics_path)
            small_f1_pass = compare_f1(
                "Retrained LinearSVC eval_test",
                float(small_metrics["f1"]),
                SMALL_MODEL_EXPECTED_F1,
                args.tolerance,
            )
            small_analysis_json, small_analysis_md = generate_error_analysis(
                small_output / "test" / "test_errors.json",
                small_output / "test",
                "Character LinearSVC + dictionary features",
            )
            summary["small_model"] = {
                "metrics_file": display_path(small_metrics_path),
                "predictions_file": display_path(small_output / "test" / "test_predictions.jsonl"),
                "errors_file": display_path(small_output / "test" / "test_errors.json"),
                "manifest": display_path(small_output / "test" / "test_manifest.json"),
                "error_analysis_json": display_path(small_analysis_json),
                "error_analysis_md": display_path(small_analysis_md),
                "f1": float(small_metrics["f1"]),
                "expected_f1": SMALL_MODEL_EXPECTED_F1,
                "verified": small_f1_pass,
                "executed_here": True,
            }

            checkpoint, checkpoint_source = prepare_bert_checkpoint(
                args.bert_checkpoint,
                variant,
            )
            print(
                f"[PASS] BERT checkpoint SHA-256 verified: {variant['sha256']}",
                flush=True,
            )
            model_dir = resolve_bert_model_dir(args.model_dir)
            bert_output = output_dir / "bert_gp"
            bert_run_name = "reproduction_" + stamp
            bert_command = [
                sys.executable,
                str(variant["trainer"]),
                "--run-name",
                bert_run_name,
                "--checkpoint",
                str(checkpoint),
                "--final",
                "--output-dir",
                str(bert_output),
                "--quiet",
            ]
            if model_dir is not None:
                bert_command.extend(["--model-dir", str(model_dir)])
            run_command(bert_command)
            bert_metrics_path = bert_output / "metrics.json"
            bert_manifest_path = bert_output / "manifest.json"
            require(bert_metrics_path.exists(), f"BERT run did not create {bert_metrics_path}")
            require(bert_manifest_path.exists(), f"BERT run did not create {bert_manifest_path}")
            bert_metrics = load_json(bert_metrics_path)
            bert_run_manifest = load_json(bert_manifest_path)
            require(
                bert_run_manifest.get("evaluation_split") == "eval_test",
                "Reproduced BERT run did not use eval_test",
            )
            bert_run_pass = compare_f1(
                f"{variant['run_name']} eval_test",
                float(bert_metrics["f1"]),
                float(variant["expected_f1"]),
                args.tolerance,
            )
            bert_analysis_json, bert_analysis_md = generate_error_analysis(
                bert_output / "errors.json",
                bert_output,
                str(variant["label"]),
            )
            summary["bert_execution"] = {
                "checkpoint": display_path(checkpoint),
                "checkpoint_source": checkpoint_source,
                "checkpoint_sha256": variant["sha256"],
                "model_dir": display_path(model_dir) if model_dir is not None else None,
                "metrics_file": display_path(bert_metrics_path),
                "predictions_file": display_path(bert_output / "predictions.jsonl"),
                "errors_file": display_path(bert_output / "errors.json"),
                "manifest": display_path(bert_manifest_path),
                "error_analysis_json": display_path(bert_analysis_json),
                "error_analysis_md": display_path(bert_analysis_md),
                "f1": float(bert_metrics["f1"]),
                "expected_f1": float(variant["expected_f1"]),
                "verified": bert_run_pass,
                "executed_here": True,
            }

            passed = (
                pure_rules_f1_pass
                and dictionary_f1_pass
                and small_f1_pass
                and bert_f1_pass
                and bert_run_pass
            )
            summary["status"] = "pass" if passed else "failed"

        summary["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        summary["elapsed_seconds"] = time.perf_counter() - started
        report_text = None
        if not args.check_only:
            report_path = output_dir / "reproduction_report.txt"
            summary["report_file"] = display_path(report_path)
            report_text = build_reproduction_report(summary, data_scope)
            report_path.write_text(report_text + "\n", encoding="utf-8")
        summary_path = output_dir / "reproduction_summary.json"
        write_json(summary_path, summary)
        if report_text is not None:
            print("\n" + report_text)
        print(f"\nReproduction status: {summary['status'].upper()}")
        print(f"Summary: {summary_path}")
        return 0 if summary["status"] in {"pass", "check-only-pass"} else 1
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = str(error)
        summary["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        summary["elapsed_seconds"] = time.perf_counter() - started
        write_json(output_dir / "reproduction_summary.json", summary)
        print(f"\n[FAIL] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
