"""Run pytest with a small workaround for broken readline imports on this environment."""

from __future__ import annotations

import sys
import types


def main(argv: list[str] | None = None) -> int:
    # On this macOS + Conda Python 3.12 environment, importing readline
    # segfaults. Pytest imports it during startup, so we pre-populate a
    # harmless stub module before importing pytest.
    sys.modules.setdefault("readline", types.ModuleType("readline"))

    import pytest

    return pytest.main(argv or sys.argv[1:] or ["-v", "tests/"])


if __name__ == "__main__":
    raise SystemExit(main())
