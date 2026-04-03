from pathlib import Path
from typing import Union
import yaml


def load_config(config_path: Union[str, Path]) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(project_root: Union[str, Path], path_str: str) -> str:
    project_root = Path(project_root)
    path_obj = Path(path_str)

    if path_obj.is_absolute():
        return str(path_obj)

    return str((project_root / path_obj).resolve())


def ensure_parent_dir(path_str: Union[str, Path]) -> None:
    path_obj = Path(path_str)
    path_obj.parent.mkdir(parents=True, exist_ok=True)