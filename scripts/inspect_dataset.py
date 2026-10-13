#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fire_ura.data import (
    DatasetSchema,
    _read_table,
    expand_input_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect raw fire-sensor files and test automatic schema inference."
    )
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="CSV files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=20_000,
        help="Maximum number of rows read from each file.",
    )
    parser.add_argument(
        "--delimiter",
        default="auto",
        help="CSV delimiter, or 'auto'.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, object] = {"files": []}

    for path in expand_input_paths(args.data):
        frame = _read_table(
            path,
            delimiter=args.delimiter,
            encoding=args.encoding,
            nrows=args.rows,
        )
        file_report: dict[str, object] = {
            "path": str(path),
            "shape_read": list(frame.shape),
            "columns": list(frame.columns),
            "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        }

        try:
            schema = DatasetSchema.infer(frame)
            file_report["inferred_schema"] = schema.as_dict()
            for label_name in (
                "scenario_label",
                "fire_label",
                "ternary_label",
                "anomaly_label",
                "progress_label",
            ):
                column = getattr(schema, label_name)
                if column:
                    counts = (
                        frame[column]
                        .astype(str)
                        .value_counts(dropna=False)
                        .head(30)
                        .to_dict()
                    )
                    file_report[f"{label_name}_counts"] = counts
        except Exception as exc:
            file_report["schema_error"] = str(exc)

        missing_counts = frame.isna().sum()
        file_report["missing_values"] = {
            column: int(count)
            for column, count in missing_counts.items()
            if int(count) > 0
        }
        report["files"].append(file_report)

    text = json.dumps(report, indent=2, default=str)
    print(text)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
