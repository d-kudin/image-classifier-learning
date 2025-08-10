# src/utils.py

from pathlib import Path

def list_classes(root: str) -> list[str]:
    """Returns a sorted list of class names based on the subdirectories in data/train."""
    cls_dirs = [p.name for p in Path(root).glob("*") if p.is_dir()]
    cls_dirs.sort()
    return cls_dirs

def ensure_dirs():
    for p in ["data/train","data/val","data/test","models","results"]:
        Path(p).mkdir(parents=True, exist_ok=True)
