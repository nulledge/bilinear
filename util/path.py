import os


def safe_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
