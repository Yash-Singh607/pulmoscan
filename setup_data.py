"""Thin wrapper kept for backward compatibility.

Prefer: ``python -m chestxray.cli setup-data``
"""

import sys

from chestxray.cli import main

if __name__ == "__main__":
    main(["setup-data", *sys.argv[1:]])
