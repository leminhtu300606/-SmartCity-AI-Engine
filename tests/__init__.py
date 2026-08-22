"""Rule 15: test cho từng nhóm rule — false positive phải bị loại.

Chạy:  python -m unittest discover -s tests -t .
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))