"""Canonical hashes for inputs, decisions and replay."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (tuple, list)):
        return [_normalize(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, frozenset):
        return sorted(_normalize(v) for v in value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def local_import_closure(entry_stem: str, package_dir) -> set[str]:
    import re
    from pathlib import Path

    package_dir = Path(package_dir)
    pattern = re.compile(r"^from \.(\w+) import|^from \. import (.+)$", re.M)
    seen: set[str] = set()
    queue = [entry_stem]
    while queue:
        stem = queue.pop()
        if stem in seen:
            continue
        seen.add(stem)
        path = package_dir / f"{stem}.py"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            if match.group(1):
                queue.append(match.group(1))
            elif match.group(2):
                queue.extend(name.strip() for name in match.group(2).split(","))
    return seen
