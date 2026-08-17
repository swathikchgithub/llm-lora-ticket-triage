import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for subdir in ["data", "eval"]:
    path = os.path.join(ROOT, subdir)
    if path not in sys.path:
        sys.path.insert(0, path)
