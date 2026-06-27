#!/bin/bash
# Open lab notebooks in browser Jupyter (works when Cursor kernel list is empty)
cd "$(dirname "$0")"
source .venv/bin/activate
echo "Starting Jupyter... open http://localhost:8888 in your browser"
jupyter notebook "${1:-M2L2_Lab.ipynb}"
