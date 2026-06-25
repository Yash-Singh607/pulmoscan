"""Thin wrapper kept for backward compatibility.

Prefer: ``python -m chestxray.cli train ...``
"""

import sys

from chestxray.cli import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
