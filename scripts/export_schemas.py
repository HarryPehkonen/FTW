#!/usr/bin/env python3
"""Regenerate schemas/*.schema.json from the Pydantic envelope models.

Usage: uv run scripts/export_schemas.py
"""

from pathlib import Path

from ftw.protocol import export_json_schemas

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    written = export_json_schemas(REPO_ROOT / "schemas")
    for message_type, path in sorted(written.items()):
        print(f"{message_type:10s} -> {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
