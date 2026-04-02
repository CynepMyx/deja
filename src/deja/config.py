import os
from platformdirs import user_data_dir

APP_NAME = "deja"
LEGACY_INDEX_DIR = os.path.join(os.path.expanduser("~"), ".claude", "deja")
DEFAULT_INDEX_DIR = user_data_dir(APP_NAME, appauthor=False, ensure_exists=False)
CLAUDE_PROJECTS_DIR = os.path.join(os.path.expanduser("~"), ".claude", "projects")


def get_index_dir() -> str:
    env = os.environ.get("DEJA_INDEX_PATH")
    if env:
        return os.path.dirname(env)

    legacy_db = os.path.join(LEGACY_INDEX_DIR, "index.db")
    if os.path.exists(legacy_db):
        return LEGACY_INDEX_DIR

    return DEFAULT_INDEX_DIR


def get_index_path() -> str:
    env = os.environ.get("DEJA_INDEX_PATH")
    if env:
        return env
    return os.path.join(get_index_dir(), "index.db")
