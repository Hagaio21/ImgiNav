#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Iterable, Any


def read_manifest(path: Path) -> List[Dict[str, Any]]:
    """Read a CSV manifest into a list of dict rows."""
    import csv

    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def create_manifest(rows: Iterable[Dict[str, Any]], output: Path, fieldnames: List[str]) -> None:
    """Write rows (list of dict) to a CSV manifest with the given fieldnames."""
    import csv

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    """Read JSON file and return parsed object."""
    import json

    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write data to JSON file with utf-8 encoding."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding='utf-8')


def read_yaml(path: Path) -> Any:
    """Read YAML file and return parsed object."""
    import yaml

    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


