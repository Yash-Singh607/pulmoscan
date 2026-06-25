"""Thin wrapper kept for backward compatibility.

Prefer: ``python -m chestxray.cli predict ...``
"""

import sys

from chestxray.cli import main

if __name__ == "__main__":
    main(["predict", *sys.argv[1:]])
