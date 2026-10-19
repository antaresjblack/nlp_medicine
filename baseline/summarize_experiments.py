"""Create a compact Markdown table from saved *_metrics.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiment_summary.md"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for path in sorted(args.output_dir.glob("*_metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        method = data.get("method", path.stem)
        if data.get("metric") == "strict_entity_micro" and method == path.stem:
            method = path.stem.removesuffix("_metrics")
        rows.append(
            {
                "method": method,
                "entity_f1": data.get("f1", 0.0),
                "entity_precision": data.get("precision", 0.0),
                "entity_recall": data.get("recall", 0.0),
                "token_f1": data.get("token_entity_f1", 0.0),
                "token_accuracy": data.get("token_accuracy", 0.0),
                "predicted_entities": data.get("predicted_entities", 0),
                "runtime_seconds": data.get("runtime_seconds", "-") ,
            }
        )

    lines = [
        "# Baseline experiment summary",
        "",
        "Token metrics use the documented longest-non-overlapping character projection; entity metrics use strict span matching.",
        "",
        "| Method | Entity P | Entity R | Entity F1 | Token F1 | Token accuracy | Predicted entities | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        runtime = row["runtime_seconds"]
        runtime_text = f"{runtime:.2f}" if isinstance(runtime, (int, float)) else str(runtime)
        lines.append(
            f"| {row['method']} | {row['entity_precision']:.4f} | {row['entity_recall']:.4f} | "
            f"{row['entity_f1']:.4f} | {row['token_f1']:.4f} | {row['token_accuracy']:.4f} | "
            f"{row['predicted_entities']} | {runtime_text} |"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
