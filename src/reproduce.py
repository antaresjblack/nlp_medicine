"""One-command reproduction entry for the submitted CMeEE experiment.

By default this script:
1. verifies the frozen data and selected small-model configuration;
2. retrains the frozen character LinearSVC and evaluates it on eval_test;
3. verifies the committed BERT result evidence.

If a trusted BERT checkpoint is supplied, it also runs final BERT inference on
eval_test and compares the resulting F1 with the recorded main result.
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
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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
BERT_EVIDENCE = ROOT / "outputs" / "bert_gp" / "test_macbert_warmup_5ep_v1" / "metrics.json"
BERT_TRAINING_PROVENANCE = (
    ROOT / "outputs" / "bert_gp" / "test_macbert_warmup_5ep_v1" / "training_provenance.json"
)

EXPECTED_DATA_HASHES = {
    "data/data_cleaned/CMeEE_train_cleaned_final.json": "e0e0d1bba812040e8e7f3e1096234e361db7ccc048db9575943fcfd9b3eb86dc",
    "data/data_raw/CMeEE_dev.json": "bbe6be016bdb1a418362507024902b70587b951a884cf4a96a6d1246822cd07c",
    "data/data_eval/eval_tune.jsonl": "ff1e9fdc685a4b405a55cfece540eceb7cba7bfdb70492693c51c6e42d0f584b",
    "data/data_eval/eval_test.jsonl": "ca65235e83f5ba055af7498b25ea55c376747a64ab6b1d4e5bffc0dfb8e62778",
}

SMALL_MODEL_EXPECTED_F1 = 0.4289685582822086
BERT_EXPECTED_F1 = 0.6670225673380248
BERT_CHECKPOINT_SHA256 = "868fbeb8530125552ff06ef8ca4c29f9cfad8b25f087be1e1d61687b41038cf7"
BERT_CHECKPOINT_FILENAME = "macbert_warmup_5ep_v1.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_bert_checkpoint(value: str) -> Path:
    """Resolve a local checkpoint or download the pinned checkpoint from HTTPS."""
    parsed = urlparse(value)
    if parsed.scheme:
        require(parsed.scheme == "https", "BERT checkpoint URL must use HTTPS")
        if "/releases/assets/" in parsed.path:
            filename = BERT_CHECKPOINT_FILENAME
        else:
            filename = Path(parsed.path).name
            require(
                filename.endswith(".pt") and filename not in {"", ".pt"},
                "Checkpoint URL must end with a .pt filename",
            )
        cache_dir = ROOT / ".checkpoints"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / filename
        if target.is_file():
            digest = sha256_file(target)
            require(
                digest == BERT_CHECKPOINT_SHA256,
                f"Cached checkpoint hash mismatch: expected {BERT_CHECKPOINT_SHA256}, got {digest}",
            )
            print(f"[INFO] Using cached checkpoint: {display_path(target)}")
            return target

        partial = target.with_name(target.name + ".part")
        print(f"[INFO] Downloading checkpoint: {value}", flush=True)
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
                "1800",
                "--silent",
                "--show-error",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                "--header",
                "Accept: application/octet-stream",
                "--header",
                "User-Agent: Codex-CMeEE-Reproduction",
                value,
            ]
            completed = subprocess.run(command)
            require(completed.returncode == 0, f"Failed to download checkpoint (curl exit {completed.returncode})")
        else:
            request = Request(
                value,
                headers={
                    "User-Agent": "Codex-CMeEE-Reproduction",
                    "Accept": "application/octet-stream",
                },
            )
            with urlopen(request, timeout=120) as response, partial.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
        digest = sha256_file(partial)
        require(
            digest == BERT_CHECKPOINT_SHA256,
            f"Downloaded checkpoint hash mismatch: expected {BERT_CHECKPOINT_SHA256}, got {digest}",
        )
        os.replace(partial, target)
        print(f"[PASS] Downloaded and verified checkpoint: {display_path(target)}")
        return target

    path = Path(value).expanduser().resolve()
    require(path.is_file(), f"BERT checkpoint not found: {path}")
    digest = sha256_file(path)
    require(
        digest == BERT_CHECKPOINT_SHA256,
        f"BERT checkpoint hash mismatch: expected {BERT_CHECKPOINT_SHA256}, got {digest}",
    )
    return path


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


def verify_environment() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for module_name, key in (("numpy", "numpy"), ("scipy", "scipy"), ("sklearn", "scikit_learn")):
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            raise RuntimeError(
                f"Missing dependency {module_name}. Run: python -m pip install -r requirements.txt"
            ) from error
        versions[key] = module.__version__
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
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def compare_f1(name: str, actual: float, expected: float, tolerance: float) -> bool:
    difference = abs(actual - expected)
    passed = difference <= tolerance
    print(
        f"[{'PASS' if passed else 'FAIL'}] {name}: actual={actual:.6f}, "
        f"expected={expected:.6f}, |delta|={difference:.6f}, tolerance={tolerance:.6f}"
    )
    return passed


def verify_bert_evidence() -> dict[str, Any]:
    require(BERT_EVIDENCE.exists(), f"Recorded BERT metrics are missing: {BERT_EVIDENCE}")
    require(
        BERT_TRAINING_PROVENANCE.exists(),
        f"BERT training provenance is missing: {BERT_TRAINING_PROVENANCE}",
    )
    metrics = load_json(BERT_EVIDENCE)
    provenance = load_json(BERT_TRAINING_PROVENANCE)
    require(metrics.get("metric") == "strict_entity_micro", "BERT evidence uses an unexpected metric")
    require(
        provenance.get("evaluation_split") == "eval_test",
        "BERT evidence was not evaluated on eval_test",
    )
    require(
        provenance.get("training_args", {}).get("seed") == 42,
        "BERT training seed is not 42",
    )
    return {"metrics": metrics, "provenance": provenance}


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
        "--bert-checkpoint",
        type=str,
        default=None,
        help="Trusted MacBERT checkpoint path or HTTPS URL for final eval_test inference. Requires torch and transformers.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Allowed absolute F1 difference for retrained baselines; default 0.01.",
    )
    args = parser.parse_args()

    require(args.tolerance >= 0, "tolerance must be non-negative")
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir = (args.out_dir or ROOT / "reproduction_outputs" / stamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    summary: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "repository_commit": git_commit(),
        "output_dir": str(output_dir),
        "check_only": args.check_only,
    }

    try:
        summary["environment"] = verify_environment()
        summary["data_hashes"] = verify_data_hashes()
        summary["frozen_small_model_config"] = verify_frozen_config()
        print("[PASS] Environment, required data, hashes and frozen small-model configuration.")

        bert_evidence = verify_bert_evidence()
        bert_f1_pass = compare_f1(
            "Committed BERT eval_test evidence",
            float(bert_evidence["metrics"]["f1"]),
            BERT_EXPECTED_F1,
            1e-12,
        )
        summary["bert_evidence"] = {
            "metrics_file": display_path(BERT_EVIDENCE),
            "training_provenance": display_path(BERT_TRAINING_PROVENANCE),
            "f1": float(bert_evidence["metrics"]["f1"]),
            "verified": bert_f1_pass,
            "executed_here": False,
        }

        if args.check_only:
            summary["status"] = "check-only-pass" if bert_f1_pass else "failed"
            print("[INFO] Check-only mode completed; no model was trained or executed.")
        else:
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
            summary["small_model"] = {
                "metrics_file": display_path(small_metrics_path),
                "f1": float(small_metrics["f1"]),
                "expected_f1": SMALL_MODEL_EXPECTED_F1,
                "verified": small_f1_pass,
                "executed_here": True,
            }

            if args.bert_checkpoint is not None:
                checkpoint = resolve_bert_checkpoint(args.bert_checkpoint)
                bert_run_name = "reproduction_" + stamp
                run_command(
                    [
                        sys.executable,
                        "bert_gp/train.py",
                        "--run-name",
                        bert_run_name,
                        "--checkpoint",
                        str(checkpoint),
                        "--final",
                    ]
                )
                bert_metrics_path = ROOT / "outputs" / "bert_gp" / bert_run_name / "metrics.json"
                require(bert_metrics_path.exists(), f"BERT run did not create {bert_metrics_path}")
                bert_metrics = load_json(bert_metrics_path)
                bert_run_pass = compare_f1(
                    "Checkpoint MacBERT eval_test",
                    float(bert_metrics["f1"]),
                    BERT_EXPECTED_F1,
                    args.tolerance,
                )
                summary["bert_execution"] = {
                    "checkpoint": display_path(checkpoint),
                    "checkpoint_source": args.bert_checkpoint,
                    "checkpoint_sha256": BERT_CHECKPOINT_SHA256,
                    "metrics_file": display_path(bert_metrics_path),
                    "f1": float(bert_metrics["f1"]),
                    "expected_f1": BERT_EXPECTED_F1,
                    "verified": bert_run_pass,
                    "executed_here": True,
                }
            else:
                bert_run_pass = True
                summary["bert_execution"] = {
                    "status": "skipped",
                    "reason": "No --bert-checkpoint supplied; only committed evidence was checked.",
                }
                print(
                    "[SKIP] BERT inference was not run because no checkpoint was supplied.\n"
                    "       Use --bert-checkpoint <best.pt> in an environment with torch/transformers "
                    "to reproduce the 0.6670 test result."
                )

            passed = small_f1_pass and bert_f1_pass and bert_run_pass
            summary["status"] = "pass" if passed else "failed"

        summary["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        summary["elapsed_seconds"] = time.perf_counter() - started
        summary_path = output_dir / "reproduction_summary.json"
        write_json(summary_path, summary)
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
