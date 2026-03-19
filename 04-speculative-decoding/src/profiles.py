from pathlib import Path

import yaml


def load_profile(name: str, profiles_dir: Path | None = None) -> dict:
    """Load a hardware profile by name (gpu or local).

    Profiles are YAML files in the profiles/ directory with the same
    schema as configs/*.yaml. This allows reuse of the existing
    load_config() function.
    """
    if profiles_dir is None:
        profiles_dir = Path("profiles")
    path = profiles_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)
