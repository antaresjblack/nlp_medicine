"""Run member 2's character LinearSVC baseline on the frozen eval split.

This is an additive integration layer. It never rewrites member 2's historical
``outputs/`` files or ``configs/final_baseline.json``.

Workflow:
    1. Tune candidates on ``data_eval/eval_tune.jsonl``.
    2. Freeze the selected candidate in ``tune/.../selected_config.json``.
    3. Run the selected candidate once on ``data_eval/eval_test.jsonl``.

Examples:
    python eval/run_small_model_split.py --stage tune --dry-run
    python eval/run_small_model_split.py --stage tune
    python eval/run_small_model_split.py --stage test --selected-config <path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from dict_span_features import DictSpanAnnotator
from metrics import evaluate_records
from small_model import char_features, decode_bio, labels_from_entities, sparse_int32


TRAIN_REL = Path("data_cleaned/CMeEE_train_cleaned_final.json")
TUNE_REL = Path("data_eval/eval_tune.jsonl")
TEST_REL = Path("data_eval/eval_test.jsonl")
CHINA_TZ = timezone(timedelta(hours=8))


def now_iso() -> str:
    return datetime.now(CHINA_TZ).isoformat(timespec="seconds")


def run_id() -> str:
    return datetime.now(CHINA_TZ).strftime("%Y%m%dT%H%M%S%z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record is not an object")
            if "id" not in record:
                raise ValueError(f"{path}:{line_number}: missing record id")
            records.append(record)
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_record() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
    }


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(f"[{now_iso()}] {message}\n")


@dataclass
class Candidate:
    id: str
    C: float
    class_weight: str
    dict_span_features: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "C": self.C,
            "class_weight": self.class_weight,
            "dict_span_features": self.dict_span_features,
        }


@dataclass
class TrainedCandidate:
    classifier: LinearSVC
    vectorizer: DictVectorizer
    annotator: DictSpanAnnotator | None
    dictionary_chars: set[str]
    feature_count: int
    train_characters: int
    runtime_seconds: float


def safe_candidate_id(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("candidate id must contain only letters, digits, dot, underscore, or hyphen")
    return value


def read_candidates(path: Path) -> tuple[dict[str, Any], list[Candidate]]:
    payload = load_json(path)
    seed = int(payload.get("seed", 42))
    projection = str(payload.get("projection", "longest_non_overlapping_gold_entities"))
    if projection != "longest_non_overlapping_gold_entities":
        raise ValueError(f"unsupported projection: {projection}")

    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("candidates must be a non-empty list")

    candidates: list[Candidate] = []
    seen: set[str] = set()
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            raise ValueError("each candidate must be an object")
        candidate_id = safe_candidate_id(raw.get("id"))
        if candidate_id in seen:
            raise ValueError(f"duplicate candidate id: {candidate_id}")
        seen.add(candidate_id)
        C = float(raw.get("C"))
        class_weight = str(raw.get("class_weight", "none")).lower()
        if C <= 0:
            raise ValueError(f"{candidate_id}: C must be positive")
        if class_weight not in {"none", "balanced"}:
            raise ValueError(f"{candidate_id}: class_weight must be none or balanced")
        dict_span_features = raw.get("dict_span_features")
        if not isinstance(dict_span_features, bool):
            raise ValueError(f"{candidate_id}: dict_span_features must be boolean")
        candidates.append(
            Candidate(
                id=candidate_id,
                C=C,
                class_weight=class_weight,
                dict_span_features=dict_span_features,
            )
        )

    settings = {
        "seed": seed,
        "projection": projection,
        "dict_min_count": int(payload.get("dict_min_count", 1)),
        "dict_min_length": int(payload.get("dict_min_length", 2)),
    }
    return settings, candidates


def train_candidate(
    train: list[dict[str, Any]],
    candidate: Candidate,
    settings: dict[str, Any],
) -> TrainedCandidate:
    started = time.perf_counter()
    dictionary_phrases = {
        entity["entity"]
        for record in train
        for entity in record.get("entities", [])
    }
    dictionary_chars = {character for phrase in dictionary_phrases for character in phrase}
    annotator = (
        DictSpanAnnotator(
            train,
            min_count=settings["dict_min_count"],
            min_length=settings["dict_min_length"],
        )
        if candidate.dict_span_features
        else None
    )

    train_features: list[dict[str, str]] = []
    train_labels: list[str] = []
    for record in train:
        text = record["text"]
        spans = annotator.annotate(text) if annotator is not None else None
        train_features.extend(
            char_features(
                text,
                index,
                dictionary_chars,
                spans[index] if spans is not None else None,
            )
            for index in range(len(text))
        )
        train_labels.extend(labels_from_entities(text, record.get("entities", [])))

    vectorizer = DictVectorizer()
    x_train = sparse_int32(vectorizer.fit_transform(train_features))
    class_weight = None if candidate.class_weight == "none" else candidate.class_weight
    classifier = LinearSVC(
        C=candidate.C,
        class_weight=class_weight,
        random_state=settings["seed"],
    )
    classifier.fit(x_train, train_labels)
    return TrainedCandidate(
        classifier=classifier,
        vectorizer=vectorizer,
        annotator=annotator,
        dictionary_chars=dictionary_chars,
        feature_count=len(vectorizer.vocabulary_),
        train_characters=len(train_labels),
        runtime_seconds=time.perf_counter() - started,
    )


def predict_records(
    records: list[dict[str, Any]],
    model: TrainedCandidate,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for record in records:
        text = record["text"]
        spans = model.annotator.annotate(text) if model.annotator is not None else None
        features = [
            char_features(
                text,
                index,
                model.dictionary_chars,
                spans[index] if spans is not None else None,
            )
            for index in range(len(text))
        ]
        labels = model.classifier.predict(
            sparse_int32(model.vectorizer.transform(features))
        ).tolist()
        prediction: dict[str, Any] = {
            "id": record["id"],
            "text": text,
            "entities": decode_bio(text, labels),
        }
        predictions.append(prediction)
    return predictions


def compact_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "errors"}


def run_tune(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    candidates_path = args.candidates_config.resolve()
    output_root = (args.output_dir or root / "outputs/eval_split/small_model").resolve()
    settings, candidates = read_candidates(candidates_path)

    train_path = root / TRAIN_REL
    tune_path = root / TUNE_REL
    for path in (train_path, tune_path):
        if not path.exists():
            raise FileNotFoundError(path)

    if args.dry_run:
        print(f"Train: {train_path}")
        print(f"Tune:  {tune_path}")
        print(f"Candidates: {len(candidates)}")
        print(f"Output: {output_root / 'tune/runs/<timestamp>'}")
        for candidate in candidates:
            print(f"  - {candidate.id}: {candidate.as_dict()}")
        return 0

    tune_run_dir = output_root / "tune" / "runs" / run_id()
    candidate_dir = tune_run_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=False)
    logger = RunLogger(tune_run_dir / "tune_run.log")
    logger.log(f"stage=tune root={root}")
    logger.log(f"candidates_config={candidates_path}")
    logger.log(f"train={train_path}")
    logger.log(f"tune={tune_path}")
    logger.log(f"environment={environment_record()}")

    train = load_json(train_path)
    tune = load_jsonl(tune_path)
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, 1):
        logger.log(f"[{index}/{len(candidates)}] training {candidate.id}")
        model = train_candidate(train, candidate, settings)
        predictions = predict_records(tune, model)
        result = evaluate_records(tune, predictions)
        metrics = compact_metrics(result)
        metrics.update(
            {
                "method": "character_linear_svm_split",
                "candidate": candidate.as_dict(),
                "settings": settings,
                "split": "eval_tune",
                "train_file": relative_path(train_path, root),
                "gold_file": relative_path(tune_path, root),
                "feature_count": model.feature_count,
                "train_characters": model.train_characters,
                "runtime_seconds": model.runtime_seconds,
            }
        )

        stem = candidate.id
        prediction_path = candidate_dir / f"{stem}_predictions.jsonl"
        metrics_path = candidate_dir / f"{stem}_metrics.json"
        errors_path = candidate_dir / f"{stem}_errors.json"
        write_jsonl(prediction_path, predictions)
        write_json(metrics_path, metrics)
        write_json(errors_path, result["errors"])
        logger.log(
            f"{candidate.id}: P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
            f"time={model.runtime_seconds:.1f}s"
        )
        rows.append(
            {
                "candidate_id": candidate.id,
                "C": candidate.C,
                "class_weight": candidate.class_weight,
                "dict_span_features": candidate.dict_span_features,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "token_entity_f1": metrics["token_entity_f1"],
                "char_multi_label_f1": metrics["char_multi_label_f1"],
                "predicted_entities": metrics["predicted_entities"],
                "runtime_seconds": model.runtime_seconds,
                "prediction_file": relative_path(prediction_path, root),
                "metrics_file": relative_path(metrics_path, root),
                "errors_file": relative_path(errors_path, root),
            }
        )

    rows.sort(key=lambda row: (-row["f1"], -row["precision"], row["runtime_seconds"]))
    best = rows[0]
    selected = next(candidate for candidate in candidates if candidate.id == best["candidate_id"])
    summary = {
        "stage": "tune",
        "created_at": now_iso(),
        "selection_metric": "strict_entity_micro_f1",
        "selection_split": "eval_tune",
        "candidate_config": relative_path(candidates_path, root),
        "candidate_config_sha256": sha256_file(candidates_path),
        "settings": settings,
        "ranking": rows,
        "selected_candidate": selected.as_dict(),
    }
    summary_path = tune_run_dir / "candidate_summary.json"
    write_json(summary_path, summary)

    markdown = [
        "# Small model tune results",
        "",
        f"- Created: {summary['created_at']}",
        f"- Selection metric: `{summary['selection_metric']}`",
        f"- Selection split: `{summary['selection_split']}`",
        "",
        "| Rank | Candidate | C | class_weight | Dict span | Entity P | Entity R | Entity F1 | Token F1 | Char-multi F1 | Runtime (s) |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        markdown.append(
            f"| {rank} | `{row['candidate_id']}` | {row['C']} | {row['class_weight']} | "
            f"{str(row['dict_span_features']).lower()} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['token_entity_f1']:.4f} | "
            f"{row['char_multi_label_f1']:.4f} | {row['runtime_seconds']:.1f} |"
        )
    markdown += ["", f"Selected: `{selected.id}`", ""]
    (tune_run_dir / "candidate_summary.md").write_text("\n".join(markdown), encoding="utf-8")

    selected_config = {
        "status": "frozen",
        "frozen_at": now_iso(),
        "selection_metric": "strict_entity_micro_f1",
        "selection_split": "eval_tune",
        "selected_candidate": selected.as_dict(),
        "settings": settings,
        "files": {
            "train": relative_path(train_path, root),
            "tune": relative_path(tune_path, root),
            "test": TEST_REL.as_posix(),
        },
        "hashes": {
            "train": sha256_file(train_path),
            "tune": sha256_file(tune_path),
            "candidate_config": sha256_file(candidates_path),
            "tune_summary": sha256_file(summary_path),
        },
        "tune_metrics": {
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "prediction_file": best["prediction_file"],
            "metrics_file": best["metrics_file"],
        },
        "source": {
            "repository_commit": git_commit(root),
            "candidate_summary": relative_path(summary_path, root),
        },
        "environment": environment_record(),
    }
    selected_path = tune_run_dir / "selected_config.json"
    write_json(selected_path, selected_config)
    logger.log(f"selected={selected.id} f1={best['f1']:.4f}")
    logger.log(f"selected_config={selected_path}")
    print(f"Selected config: {selected_path}")
    return 0


def resolve_frozen_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def run_test(args: argparse.Namespace) -> int:
    if args.selected_config is None:
        raise SystemExit("--selected-config is required for --stage test")
    root = args.root.resolve()
    selected_path = args.selected_config.resolve()
    frozen = load_json(selected_path)
    if frozen.get("status") != "frozen":
        raise SystemExit("selected config is not frozen")

    output_root = (args.output_dir or root / "outputs/eval_split/small_model").resolve()
    test_dir = output_root / "test"
    prediction_path = test_dir / "test_predictions.jsonl"
    metrics_path = test_dir / "test_metrics.json"
    errors_path = test_dir / "test_errors.json"
    log_path = test_dir / "test_run.log"
    manifest_path = test_dir / "test_manifest.json"
    existing = [
        path
        for path in (prediction_path, metrics_path, errors_path, log_path, manifest_path)
        if path.exists()
    ]
    if existing:
        raise SystemExit(f"test artifacts already exist; refusing a second run: {existing[0]}")

    files = frozen["files"]
    train_path = resolve_frozen_path(root, files["train"])
    test_path = resolve_frozen_path(root, files["test"])
    train = load_json(train_path)
    test = load_jsonl(test_path)
    logger = RunLogger(log_path)
    logger.log(f"stage=test root={root}")
    logger.log(f"selected_config={selected_path}")
    logger.log(f"train={train_path}")
    logger.log(f"test={test_path}")

    selected = frozen["selected_candidate"]
    candidate = Candidate(
        id=safe_candidate_id(selected["id"]),
        C=float(selected["C"]),
        class_weight=str(selected["class_weight"]).lower(),
        dict_span_features=bool(selected["dict_span_features"]),
    )
    settings = frozen["settings"]
    model = train_candidate(train, candidate, settings)
    predictions = predict_records(test, model)
    result = evaluate_records(test, predictions)
    metrics = compact_metrics(result)
    metrics.update(
        {
            "method": "character_linear_svm_split",
            "candidate": candidate.as_dict(),
            "settings": settings,
            "split": "eval_test",
            "train_file": relative_path(train_path, root),
            "gold_file": relative_path(test_path, root),
            "feature_count": model.feature_count,
            "train_characters": model.train_characters,
            "runtime_seconds": model.runtime_seconds,
            "selected_config": relative_path(selected_path, root),
        }
    )
    write_jsonl(prediction_path, predictions)
    write_json(metrics_path, metrics)
    write_json(errors_path, result["errors"])
    logger.log(
        f"test: P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
        f"F1={metrics['f1']:.4f} time={model.runtime_seconds:.1f}s"
    )

    manifest = {
        "stage": "test",
        "completed_at": now_iso(),
        "selected_config": relative_path(selected_path, root),
        "selected_config_sha256": sha256_file(selected_path),
        "files": {
            "train": relative_path(train_path, root),
            "test": relative_path(test_path, root),
            "predictions": relative_path(prediction_path, root),
            "metrics": relative_path(metrics_path, root),
            "errors": relative_path(errors_path, root),
        },
        "hashes": {
            "train": sha256_file(train_path),
            "test": sha256_file(test_path),
            "predictions": sha256_file(prediction_path),
            "metrics": sha256_file(metrics_path),
        },
        "selected_candidate": candidate.as_dict(),
        "metrics": {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        },
        "source": {"repository_commit": git_commit(root)},
        "environment": environment_record(),
    }
    write_json(manifest_path, manifest)
    logger.log(f"manifest={manifest_path}")
    print(f"Test metrics: {metrics_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["tune", "test"], required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--candidates-config",
        type=Path,
        default=ROOT / "configs" / "small_model_split_candidates.json",
        help="candidate list used only by the tune stage",
    )
    parser.add_argument(
        "--selected-config",
        type=Path,
        default=None,
        help="frozen selected_config.json produced by the tune stage",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="defaults to outputs/eval_split/small_model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="tune stage only: validate inputs and print candidates without training",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.stage == "tune":
        return run_tune(args)
    if args.dry_run:
        parser.error("--dry-run is only supported for --stage tune")
    return run_test(args)


if __name__ == "__main__":
    raise SystemExit(main())
