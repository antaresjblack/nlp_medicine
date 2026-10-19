"""Re-evaluate a saved prediction file with the shared metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metrics import evaluate_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.gold.open("r", encoding="utf-8-sig") as file:
        gold = json.load(file)
    with args.predictions.open("r", encoding="utf-8-sig") as file:
        predictions = json.load(file)
    result = evaluate_records(gold, predictions)
    compact = {key: value for key, value in result.items() if key != "errors"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
