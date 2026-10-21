"""
运行最终基线配置。
用法： python baseline/run_final_baselines.py --config configs/final_baseline.json
运行程序将按顺序执行：纯规则、最优字典设置、字符级小型模型、对已启用方法的错误分析，以及实验汇总表。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command: list[str], root: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-small-model", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    config_path = (args.config or root / "configs" / "final_baseline.json").resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = root / config.get("output_dir", "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = config["methods"]

    if methods["pure_rules"].get("enabled", False):
        method = methods["pure_rules"]
        run([sys.executable, method["script"], "--root", str(root), "--output-dir", str(output_dir)], root)

    dictionary = methods["dictionary_best"]
    if dictionary.get("enabled", False):
        run(
            [
                sys.executable,
                dictionary["script"],
                "--root", str(root),
                "--output-dir", str(output_dir),
                "--method", dictionary["method"],
                "--min-count", str(dictionary["min_count"]),
                "--min-length", str(dictionary["min_length"]),
            ],
            root,
        )

    small_model = methods["small_model"]
    small_model_ran = False
    if small_model.get("enabled", False) and not args.skip_small_model:
        command = [
            sys.executable,
            small_model["script"],
            "--root", str(root),
            "--output-dir", str(output_dir),
            "--seed", str(small_model.get("seed", 42)),
            "--C", str(small_model.get("C", 1.0)),
            "--class-weight", str(small_model.get("class_weight", "balanced")),
        ]
        if small_model.get("dict_span_features", False):
            command.append("--dict-span-features")
        run(command, root)
        small_model_ran = True

    error_analysis = config.get("error_analysis", {})
    if error_analysis.get("enabled", False):
        gold_path = root / error_analysis.get("gold", "data_raw/CMeEE_dev.json")
        examples_per_kind = error_analysis.get("examples_per_kind", 5)
        stems_by_name = {}
        if dictionary.get("enabled", False):
            stems_by_name["dictionary_best"] = (
                f"dictionary_{dictionary['method']}_min_count_{dictionary['min_count']}_min_length_{dictionary['min_length']}"
            )
        if methods["pure_rules"].get("enabled", False):
            stems_by_name["pure_rules"] = "pure_rules"
        if small_model_ran:
            stems_by_name["small_model"] = "small_model"
        for target in error_analysis.get("targets", []):
            stem = stems_by_name.get(target["name"])
            if stem is None:
                continue
            run(
                [
                    sys.executable,
                    error_analysis.get("script", "baseline/error_analysis.py"),
                    "--gold", str(gold_path),
                    "--predictions", str(output_dir / f"{stem}_predictions.json"),
                    "--output", str(output_dir / target["output"]),
                    "--examples-per-kind", str(examples_per_kind),
                ],
                root,
            )

    run([sys.executable, "baseline/summarize_experiments.py", "--output-dir", str(output_dir)], root)
    print(f"Final baseline outputs: {output_dir}")


if __name__ == "__main__":
    main()
