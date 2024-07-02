"""Run pytest with a small workaround for broken readline imports on this environment."""

from __future__ import annotations

import sys
import types
from pathlib import Path


def main() -> int:
    # On this macOS + Conda Python 3.12 environment, importing readline
    # segfaults. Pytest imports it during startup, so we pre-populate a
    # harmless stub module before importing pytest.
    sys.modules.setdefault("readline", types.ModuleType("readline"))
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    import pytest

    return pytest.main(sys.argv[1:] or ["-v", "tests/"])


if __name__ == "__main__":
    raise SystemExit(main())
