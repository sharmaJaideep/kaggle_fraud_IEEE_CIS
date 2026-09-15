"""Create the IEEE-CIS Fraud Detection project layout.

The script is intentionally idempotent and never creates, deletes, or modifies
anything below ``input_data/``.
"""

from pathlib import Path
from typing import Optional


DIRECTORIES = (
    "notebooks",
    "src",
    "src/data",
    "src/features",
    "src/models",
    "config",
    "models",
    "reports",
    "logs",
)


def create_project_layout(project_root: Optional[Path] = None) -> None:
    """Create the standard project directories and package markers."""
    root = Path(project_root) if project_root else Path(__file__).resolve().parent
    for directory in DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)

    for package in (root / "src", root / "src/data", root / "src/features", root / "src/models"):
        (package / "__init__.py").touch(exist_ok=True)

    for directory in (root / "models", root / "reports", root / "logs"):
        (directory / ".gitkeep").touch(exist_ok=True)

    print(f"Project layout ready at: {root}")


if __name__ == "__main__":
    create_project_layout()
