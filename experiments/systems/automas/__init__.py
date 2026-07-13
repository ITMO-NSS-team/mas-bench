import importlib.util

# The adapter imports `automas` lazily (inside its methods), so without this
# check it would register fine and only blow up mid-run. Discovery is meant to
# list a system only when its deps are installed -- fail the import instead.
if importlib.util.find_spec("automas") is None:
    raise ImportError(
        "automas is not installed (uv pip install -e automas-research/)"
    )

from .adapter import AutoMASAdapter

__all__ = ["AutoMASAdapter"]
