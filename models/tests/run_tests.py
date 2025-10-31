import os
import sys
import pytest
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PYTHONPATH", str(root))

    print(f"Running full test suite under: {root}")
    print("-" * 60)

    result = pytest.main([
        str(root / "tests"),
        "-v",
        "--disable-warnings",
        "--maxfail=1"
    ])

    sys.exit(result)
