import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # store project root for path resolution
    config["_root"] = str(Path(path).resolve().parent.parent)
    return config


def resolve_path(config: dict, relative_path: str) -> str:
    return str(Path(config["_root"]) / relative_path)
