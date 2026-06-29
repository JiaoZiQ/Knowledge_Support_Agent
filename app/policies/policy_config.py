import json
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_POLICY_PATH = Path("config/policies.json")


@lru_cache
def load_policy_config(path: str | Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    policy_path = Path(path)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy config not found: {policy_path}")
    return json.loads(policy_path.read_text(encoding="utf-8"))


def get_category_terms() -> dict[str, list[str]]:
    return load_policy_config().get("category_terms", {})
