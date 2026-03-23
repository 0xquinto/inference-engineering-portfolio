from pathlib import Path

import yaml


def load_profile(name: str, profiles_dir: Path | None = None) -> dict:
    if profiles_dir is None:
        profiles_dir = Path("profiles")
    path = profiles_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)
