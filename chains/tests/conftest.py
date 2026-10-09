import sys
from pathlib import Path

# chains/genertation.py and chains/retirval_grader.py are plain top-level
# modules (no package __init__.py, no relative imports), imported elsewhere
# in the repo as `from chains.retirval_grader import ...` with Agentic_RAG as
# CWD. For these tests we import them directly as bare modules, so the
# `chains/` directory itself needs to be on sys.path.
CHAINS_DIR = Path(__file__).resolve().parent.parent
if str(CHAINS_DIR) not in sys.path:
    sys.path.insert(0, str(CHAINS_DIR))

# Every chain module does `from imports import ...` (Agentic_RAG/imports.py,
# the project's single import page), so the project root must be importable too.
ROOT_DIR = CHAINS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
