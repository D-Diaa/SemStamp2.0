"""Utility functions for text processing."""

import os


def load_file_by_line(path):
    """Load a text file and return list of lines."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


def path_wo_ext(path):
    """Return path without file extension."""
    return os.path.splitext(path)[0]


def break_text(texts):
    """Break texts into token lists (tokenize by whitespace)."""
    return [text.lower().split() for text in texts]


def chunks(lst, n):
    """Yield successive n-sized chunks from list or tensor."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
