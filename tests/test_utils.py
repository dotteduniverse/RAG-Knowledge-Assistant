import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest


from src.utils import get_text_splitter


def test_text_splitter():
    splitter = get_text_splitter()
    text = "This is a test. " * 100
    chunks = splitter.split_text(text)
    assert len(chunks) > 1