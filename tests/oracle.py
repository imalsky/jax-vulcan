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
import sys
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


def oracle_dir_or_skip(what: str) -> Path:
    """Oracle path for a RE-EXEC'D CHILD, skipping the module when unset.

    Call this at module scope in an oracle test that runs itself as a
    subprocess. It is deliberately weaker than `require_oracle`, and the split
    is the whole design:

    - The PARENT pytest process verifies the pin -- `run_oracle_subprocess`
      goes through `oracle_worktree` -> `require_oracle`, which checks the
      exact commit and a clean tree -- then copies the checkout and points
      `VULCAN_MASTER_DIR` at the COPY.
    - The CHILD reads that copy. It cannot re-verify the revision because the
      copy has no `.git` by construction (upstream setup rewrites files in
      place, so the child must never touch the real checkout).

    So a bare existence check is the correct check HERE, and only here. Under
    a plain `pytest tests/` with no oracle configured, this skips the module,
    which is how these files behave on a fresh clone. With
    `VULCAN_JAX_REQUIRE_ORACLE=1` it fails instead (skipped != passed).
    """
    raw = os.environ.get(ENV_DIR)
    path = Path(raw).expanduser().resolve() if raw else None
    if path is not None and path.is_dir():
        return path
    msg = (
        f"upstream oracle not configured (looked at "
        f"{path if path is not None else '$' + ENV_DIR + ' unset'}). Set "
        f"${ENV_DIR} to a clean clone at the commit pinned in "
        f"tests/science_sources.yaml; {what} requires the upstream repo.")
    if os.environ.get(ENV_REQUIRE) == "1":
        pytest.fail(msg, pytrace=False)
    pytest.skip(msg, allow_module_level=True)


UNSET_SENTINEL = Path("/nonexistent/VULCAN-oracle-unset")


def oracle_dir_or_sentinel() -> Path:
    """Oracle path for a re-exec'd child, or a path that cannot exist.

    Same parent-verifies / child-reads split as `oracle_dir_or_skip` (see its
    docstring), but for the files that skip PER TEST instead of at module
    scope, because they also hold tests needing no oracle. The sentinel keeps
    every `if not VULCAN_MASTER.is_dir(): skip` site free of a None branch.
    """
    return oracle_dir() or UNSET_SENTINEL


def run_oracle_subprocess(test_file, family: str,
                          config_rel: str | None = None, *,
                          fastchem_abundance: str | None = None,
                          timeout: float | None = None) -> None:
    """Run `test_file`'s `main()` in a fresh process against an oracle COPY.

    The master/JAX module-table swap only works from a cold Python start, so
    every upstream-comparison test re-execs itself. This is the shared body of
    those wrappers: verify + copy the oracle, point the child at the copy, and
    surface the child's output on failure.
    """
    with oracle_worktree(family, config_rel,
                         fastchem_abundance=fastchem_abundance) as master:
        env = os.environ.copy()
        env["VULCAN_MASTER_DIR"] = str(master)
        result = subprocess.run(
            [sys.executable, str(Path(test_file).resolve())],
            capture_output=True, text=True, env=env, timeout=timeout)
    assert result.returncode == 0, (
        f"subprocess exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


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
def oracle_worktree(
    family: str,
    config_rel: str | None = None,
    fastchem_abundance: str | None = None,
):
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
        if config_rel is not None:
            config_source = dst / config_rel
            if not config_source.is_file():
                raise FileNotFoundError(
                    f"oracle config does not exist: {config_source}")
            shutil.copy2(config_source, dst / "vulcan_cfg.py")
        jax_fastchem = ROOT / "src" / "vulcan_jax" / "fastchem_vulcan"
        oracle_fastchem = dst / "fastchem_vulcan"
        # Upstream ships FastChem SOURCE, never a binary, so any oracle test
        # reaching ini_mix='EQ' dies with exit 127. Stage our built one: the
        # C++ source, model_main and makefile are byte-identical between the
        # two trees (verified), so it is the same program.
        if oracle_fastchem.is_dir() and not (oracle_fastchem / "fastchem").exists():
            binary = jax_fastchem / "fastchem"
            if not binary.is_file():
                raise FileNotFoundError(
                    f"oracle FastChem needs the built JAX binary at {binary}; "
                    "run any ini_mix='EQ' config once to compile it")
            shutil.copy2(binary, oracle_fastchem / "fastchem")
        if fastchem_abundance is not None:
            abundance = jax_fastchem / "input" / fastchem_abundance
            if not abundance.is_file():
                raise FileNotFoundError(f"no such abundance preset: {abundance}")
            shutil.copy2(
                jax_fastchem / "input" / "nasa9_logK_SNCHOPTi.dat",
                oracle_fastchem / "input" / "nasa9_logK_SNCHOPTi.dat")
            shutil.copy2(
                abundance,
                oracle_fastchem / "input" / "solar_element_abundances.dat")
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
