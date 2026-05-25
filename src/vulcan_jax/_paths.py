"""Package-root discovery and data-path resolution."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def resolve_data_path(rel_path: str) -> Path:
    """Resolve a relative path against the installed package root.

    Resolution order:
      1. If *rel_path* is absolute, return it unchanged.
      2. If it exists relative to the current working directory, return that
         (allows user overrides).
      3. Otherwise resolve against ``PACKAGE_ROOT`` (shipped data).
    """
    p = Path(rel_path)
    if p.is_absolute():
        return p
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate
    return PACKAGE_ROOT / p
