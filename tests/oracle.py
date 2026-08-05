"""Pinned, isolated access to an upstream VULCAN checkout for oracle tests.

Three rules (each guards against a failure that actually happened):

1. NO SILENT SIBLING. `VULCAN_MASTER_DIR` must name the checkout; never
   auto-detect `../VULCAN-master/` (an unversioned, hand-patched copy was once
   cited as evidence of upstream behavior).

2. EXACT REVISION, CLEAN TREE. `require_oracle()` verifies HEAD against
   `tests/science_sources.yaml` and a clean worktree BEFORE any calculation.
   Reaction indices are positional, so a one-commit-newer checkout must fail
   with one clear message, not a cascade of rate-index failures.

3. NEVER MUTATE THE ORACLE. Upstream setup rewrites files in place
   (make_chem_funs renumbers a network file; FastChem writes into
   fastchem_vulcan/). `oracle_worktree()` hands out a temporary COPY and
   proves the original unchanged afterwards.

Local runs may skip when no oracle is configured; a release CI job must set
`VULCAN_JAX_REQUIRE_ORACLE=1` so a missing oracle FAILS instead of skipping
(skipped != passed).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = Path(__file__).resolve().parent / "science_sources.yaml"

ENV_DIR = "VULCAN_MASTER_DIR"
ENV_REQUIRE = "VULCAN_JAX_REQUIRE_ORACLE"

_MANIFEST_CACHE: dict | None = None


def manifest() -> dict:
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        _MANIFEST_CACHE = yaml.safe_load(MANIFEST_PATH.read_text())
    return _MANIFEST_CACHE


def oracle_spec(family: str) -> dict:
    """The manifest entry for one oracle-test family."""
    oracles = manifest()["oracles"]
    if family not in oracles:
        raise KeyError(
            f"unknown oracle family {family!r}; science_sources.yaml defines "
            f"{sorted(oracles)}")
    return oracles[family]


def _git(repo: Path, *args: str) -> tuple[int, str]:
    r = subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=True, text=True, timeout=30)
    return r.returncode, r.stdout.strip()


def oracle_dir() -> Path | None:
    """The configured oracle checkout, or None when unset."""
    raw = os.environ.get(ENV_DIR)
    return Path(raw).expanduser().resolve() if raw else None


def _fail_or_skip(msg: str):
    """Fail when an oracle is REQUIRED, else skip with the same message."""
    if os.environ.get(ENV_REQUIRE) == "1":
        pytest.fail(msg)
    pytest.skip(msg)


def require_oracle(family: str) -> Path:
    """Return the oracle path for `family`, or skip/fail with one clear reason.

    Verifies the checkout exists, is a git repository, sits at the manifest's
    exact commit, and is clean. Never returns a path that fails any of those.
    """
    spec = oracle_spec(family)
    want = spec["commit"]
    path = oracle_dir()

    if path is None:
        _fail_or_skip(
            f"oracle family {family!r} needs an upstream checkout. Set "
            f"{ENV_DIR} to a clean clone of {spec['repo']} at commit "
            f"{want[:12]}:\n"
            f"    git clone {spec['repo']} /tmp/vulcan-oracle\n"
            f"    git -C /tmp/vulcan-oracle checkout {want}\n"
            f"    export {ENV_DIR}=/tmp/vulcan-oracle\n"
            "There is deliberately no default sibling path: comparing against "
            "whatever happens to sit in ../VULCAN-master is what this "
            "machinery exists to prevent.")

    if not path.is_dir():
        _fail_or_skip(f"{ENV_DIR}={path} does not exist")

    rc, head = _git(path, "rev-parse", "HEAD")
    if rc != 0 or not head:
        _fail_or_skip(
            f"{ENV_DIR}={path} is not a git checkout, so its revision cannot "
            "be verified. An unversioned copy cannot serve as an oracle: the "
            "one on this project's machine was hand-patched with VULCAN-JAX's "
            "own code and then cited as upstream evidence. Clone "
            f"{spec['repo']} and check out {want[:12]}.")

    if head != want:
        _fail_or_skip(
            f"UNSUPPORTED ORACLE REVISION for family {family!r}.\n"
            f"  expected: {want}\n"
            f"  actual:   {head}\n"
            f"  repo:     {spec['repo']}\n"
            f"  why this revision: {spec.get('why', '').strip()}\n"
            "Reaction indices are POSITIONAL, so a different revision shifts "
            "every rate/Gibbs comparison and produces failures unrelated to "
            "the ported kernels. Check out the pinned commit, or update "
            "tests/science_sources.yaml deliberately and re-measure.")

    rc, dirty = _git(path, "status", "--porcelain")
    if rc != 0:
        _fail_or_skip(f"cannot read git status of {path}")
    if dirty:
        _fail_or_skip(
            f"oracle checkout {path} is DIRTY:\n"
            + "\n".join(f"    {ln}" for ln in dirty.splitlines()[:20])
            + "\nAn oracle must be pristine. Note that upstream setup code "
              "rewrites files in place (make_chem_funs renumbers a network "
              "file; FastChem writes into fastchem_vulcan/), so a dirty tree "
              "is often the residue of an earlier test run that failed to use "
              "oracle_worktree(). Reset it: git -C "
            f"{path} checkout . && git -C {path} clean -fd")

    return path


def tree_fingerprint(path: Path) -> str:
    """SHA-256 over the checkout's tracked-file state (git-based, cheap)."""
    rc, head = _git(path, "rev-parse", "HEAD")
    rc2, dirty = _git(path, "status", "--porcelain")
    if rc != 0 or rc2 != 0:
        raise RuntimeError(f"cannot fingerprint {path}: not a git checkout")
    return hashlib.sha256(f"{head}\n{dirty}".encode()).hexdigest()


@contextmanager
def oracle_worktree(family: str):
    """Yield a temporary COPY of the oracle; the original must not be touched.

    Upstream data-generation code mutates its own checkout, so every test that
    runs it must work on a copy. The copy is deleted on exit and the original is
    proven unchanged.
    """
    src = require_oracle(family)
    before = tree_fingerprint(src)
    with tempfile.TemporaryDirectory(prefix="vulcan-oracle-") as tmp:
        dst = Path(tmp) / src.name
        # copy2 preserves mtimes; skip .git (large, and the copy is disposable)
        shutil.copytree(src, dst, symlinks=True,
                        ignore=shutil.ignore_patterns(".git", "__pycache__"))
        try:
            yield dst
        finally:
            after = tree_fingerprint(src)
            if after != before:
                rc, dirty = _git(src, "status", "--porcelain")
                raise AssertionError(
                    f"the oracle checkout at {src} CHANGED during the test. "
                    "Upstream code must only ever run against the temporary "
                    "copy.\nDirty files now:\n"
                    + "\n".join(f"    {ln}" for ln in dirty.splitlines()[:20]))


def assert_oracle_unchanged(path: Path, before: str) -> None:
    """Explicit check for tests that manage their own copy."""
    after = tree_fingerprint(path)
    assert after == before, (
        f"oracle checkout {path} changed during the test "
        f"(fingerprint {before[:12]} -> {after[:12]})")
