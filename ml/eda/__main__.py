"""``python -m ml.eda`` writes ``docs/eda.md``."""
import os

# Set before numpy is imported. Other entry points set the same caps.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["LOKY_MAX_CPU_COUNT"] = "2"

from ml.eda.eda import main

raise SystemExit(main())
