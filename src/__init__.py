# src package — inserts project root into sys.path so `config` is importable
# from any submodule regardless of how it is invoked.
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
