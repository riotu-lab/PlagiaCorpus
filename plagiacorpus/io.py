"""JSON helpers with atomic writes, so an interrupted run never leaves a corrupt file."""

import json
import os
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    """Write `data` to `path` via a temporary file and an atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp, path)


def load_queries(*paths: str | Path) -> list[dict]:
    """Concatenate the query lists stored in several JSON files (e.g. train + test)."""
    queries: list[dict] = []
    for path in paths:
        queries.extend(load_json(path))
    return queries


def read_table(path: str | Path):
    """Read an annotation sheet (.xlsx or .csv) as strings, verbatim; blank cells become ''."""
    import pandas as pd

    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str, keep_default_na=False).fillna("")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_table(table, path: str | Path) -> None:
    """Write a pandas DataFrame as .xlsx or as .csv (UTF-8 with BOM so Excel shows Arabic)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".xlsx":
        table.to_excel(path, index=False)
    else:
        table.to_csv(path, index=False, encoding="utf-8-sig")
