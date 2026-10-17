import os
from pathlib import Path
from typing import List, Dict


def ingest_documents(directory_path: str) -> List[Dict[str, str]]:
    """
    Reads all .txt files from a specified directory.

    Args:
        directory_path: Path to the folder containing documents.

    Returns:
        List of dicts with keys 'id' (filename) and 'content' (full text).
    """
    docs: List[Dict[str, str]] = []

    dir_path = Path(directory_path)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Warning: directory '{directory_path}' does not exist or is not a directory.")
        return docs

    for entry in sorted(dir_path.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() != ".txt":
            continue

        content = None
        # Try utf-8 first, fallback to latin-1 / errors handling
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                with open(entry, "r", encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Warning: could not read '{entry.name}': {e}")
                content = None
                break

        if content is None:
            # Last resort: read with errors='replace'
            try:
                with open(entry, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception as e:
                print(f"Warning: skipping '{entry.name}': {e}")
                continue

        # Skip warning for empty files but still include? Spec says handle gracefully, don't crash.
        # We include empty content but warn.
        if not content or not content.strip():
            print(f"Warning: '{entry.name}' is empty.")

        docs.append({"id": entry.name, "content": content or ""})

    return docs
