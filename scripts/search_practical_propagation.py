"""Search upstream PR history and validate later target changes at localized code."""
import sys
from pathlib import Path

# Keep the historical practical-search runtime independent of benchmark evaluation.
sys.path.insert(0,str(Path(__file__).resolve().parent/'history_runtime'))
from history_search import main

if __name__=='__main__':
    raise SystemExit(main())
