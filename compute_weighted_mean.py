from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping


def load_values(path: Path) -> Iterable[float]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a list of summary records")
    values = []
    for record in data:
        if not isinstance(record, Mapping):
            continue
        value = record.get("weighted_score_sum_actual")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def main() -> None:
    summary_files = [Path("summary.json"), Path("summary2.json")]
    all_values = []
    for path in summary_files:
        try:
            values = list(load_values(path))
        except (FileNotFoundError, ValueError) as exc:
            print(f"{path}: skipped ({exc})")
            continue
        if values:
            file_mean = mean(values)
            print(f"{path}: mean weighted_score_sum_actual = {file_mean:.4f}")
            all_values.extend(values)
        else:
            print(f"{path}: no weighted_score_sum_actual values found")

    if all_values:
        overall = mean(all_values)
        print(f"Overall mean across files = {overall:.4f}")
    else:
        print("No values collected from any summary file.")


if __name__ == "__main__":
    main()
