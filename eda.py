"""Thin wrapper kept for backward compatibility.

Prefer: ``python -m chestxray.cli eda ...``
"""

import sys

from chestxray.cli import main

if __name__ == "__main__":
    main(["eda", *sys.argv[1:]])
