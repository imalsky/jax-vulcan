"""Pinned, isolated access to an upstream VULCAN checkout for oracle tests.

Four rules:

1. `VULCAN_MASTER_DIR` names the checkout; there is no default path.
2. `require_oracle()` checks HEAD against tests/science_sources.yaml and a
   clean tree before any calculation (reaction indices are positional).
3. `oracle_worktree()` hands out a temporary copy, since upstream setup
   rewrites files in place, and proves the original unchanged afterwards.
4. An upstream defect corrected on both sides is listed once in
   `ORACLE_CODE_DELTAS` and applied to the copy only; each delta must match
   the pinned text once.

Local runs skip without an oracle; release CI sets
`VULCAN_JAX_REQUIRE_ORACLE=1` so a missing oracle fails.
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
    """Oracle path for a re-exec'd child; skip the module when unset.

    The parent (`run_oracle_subprocess` -> `oracle_worktree` -> `require_oracle`)
    verifies the pin and points `VULCAN_MASTER_DIR` at a copy with no `.git`, so
    the child can only check existence. Fails instead under
    `VULCAN_JAX_REQUIRE_ORACLE=1`.
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
            "There is no default path.")

    if not path.is_dir():
        _fail_or_skip(f"{ENV_DIR}={path} does not exist")

    rc, head = _git(path, "rev-parse", "HEAD")
    if rc != 0 or not head:
        _fail_or_skip(
            f"{ENV_DIR}={path} is not a git checkout, so its revision cannot "
            "be verified; an unversioned copy cannot serve as an oracle. Clone "
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
            "tests/science_sources.yaml and re-run the oracle tests.")

    rc, dirty = _git(path, "status", "--porcelain")
    if rc != 0:
        _fail_or_skip(f"cannot read git status of {path}")
    if dirty:
        _fail_or_skip(
            f"oracle checkout {path} is DIRTY:\n"
            + "\n".join(f"    {ln}" for ln in dirty.splitlines()[:20])
            + "\nAn oracle must be pristine: upstream setup rewrites files in "
              "place (make_chem_funs renumbers a network file; FastChem writes "
              "into fastchem_vulcan/), so a dirty tree is usually residue of a "
              "run outside oracle_worktree(). Reset it: git -C "
            f"{path} checkout . && git -C {path} clean -fd")

    return path


def tree_fingerprint(path: Path) -> str:
    """SHA-256 over the checkout's tracked-file state (git-based, cheap)."""
    rc, head = _git(path, "rev-parse", "HEAD")
    rc2, dirty = _git(path, "status", "--porcelain")
    if rc != 0 or rc2 != 0:
        raise RuntimeError(f"cannot fingerprint {path}: not a git checkout")
    return hashlib.sha256(f"{head}\n{dirty}".encode()).hexdigest()


# Rule 4. {relative file: ((OLD, NEW, tag), ...)}; OLD/NEW are exact upstream
# text (24-space indent, LF).
ORACLE_CODE_DELTAS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "op.py": ((
        ("                        nn = 0.75 - 1.27*np.log(Fc)\n"
         "                        ff = np.exp( np.log(Fc)/(1.+ (np.log(k[i]*M/k_inf)/nn)**2 ) )\n"),
        ("                        nn = 0.75 - 1.27*np.log10(Fc)\n"
         "                        ff = Fc**( 1./(1.+ (np.log10(k[i]*M/k_inf)/nn)**2 ) )\n"),
        "C20 OH+CH3+M Troe width in log10 (Visscher & Moses 2011 eq 14)",
    ),),
}


def apply_code_deltas(root: Path) -> list[str]:
    """Apply `ORACLE_CODE_DELTAS` to the COPY at `root`; return the applied tags."""
    applied: list[str] = []
    for rel, deltas in ORACLE_CODE_DELTAS.items():
        path = root / rel
        text = path.read_bytes().decode("utf-8")
        for old, new, tag in deltas:
            n = text.count(old)
            if n != 1:
                raise RuntimeError(
                    f"{path}: delta {tag!r} expects its OLD block exactly once, "
                    f"found {n}; the pinned upstream no longer matches "
                    "ORACLE_CODE_DELTAS -- re-derive or delete the entry")
            text = text.replace(old, new)
            applied.append(tag)
        path.write_bytes(text.encode("utf-8"))
    return applied


# One FastChem build per oracle family per session: `make` takes ~1 min and
# every EQ comparison needs the same binary. The build happens in a throwaway
# copy (make writes objects in place), never in the pinned checkout; the
# TemporaryDirectory handles stay alive so the binaries outlive each worktree.
_FASTCHEM_BUILDS: dict[str, Path] = {}
_FASTCHEM_BUILD_DIRS: list[tempfile.TemporaryDirectory] = []


def _oracle_fastchem_binary(family: str, src: Path) -> Path:
    """Build the oracle's own FastChem and return the binary."""
    cached = _FASTCHEM_BUILDS.get(family)
    if cached is not None and cached.is_file():
        return cached
    holder = tempfile.TemporaryDirectory(prefix="vulcan-oracle-fastchem-")
    _FASTCHEM_BUILD_DIRS.append(holder)
    tree = Path(holder.name) / "fastchem_vulcan"
    shutil.copytree(src / "fastchem_vulcan", tree, symlinks=True,
                    ignore=shutil.ignore_patterns(".git", "__pycache__"))
    (tree / "obj").mkdir(exist_ok=True)
    build = subprocess.run(["make"], cwd=str(tree), capture_output=True,
                           text=True, timeout=1800)
    binary = tree / "fastchem"
    if build.returncode != 0 or not binary.is_file():
        raise RuntimeError(
            f"building the oracle's FastChem in {tree} failed "
            f"({build.returncode}):\n{build.stdout[-2000:]}\n{build.stderr[-2000:]}")
    _FASTCHEM_BUILDS[family] = binary
    return binary


@contextmanager
def oracle_worktree(
    family: str,
    config_rel: str | None = None,
    fastchem_abundance: str | None = None,
):
    """Yield a temporary COPY of the oracle; the original must not be touched.

    Upstream data-generation code mutates its own checkout, so every test that
    runs it must work on a copy. The copy is deleted on exit and the original is
    proven unchanged. The copy carries `ORACLE_CODE_DELTAS` (rule 4).
    """
    src = require_oracle(family)
    before = tree_fingerprint(src)
    with tempfile.TemporaryDirectory(prefix="vulcan-oracle-") as tmp:
        dst = Path(tmp) / src.name
        # copy2 preserves mtimes; skip .git (large, and the copy is disposable)
        shutil.copytree(src, dst, symlinks=True,
                        ignore=shutil.ignore_patterns(".git", "__pycache__"))
        apply_code_deltas(dst)
        if config_rel is not None:
            config_source = dst / config_rel
            if not config_source.is_file():
                raise FileNotFoundError(
                    f"oracle config does not exist: {config_source}")
            shutil.copy2(config_source, dst / "vulcan_cfg.py")
        oracle_fastchem = dst / "fastchem_vulcan"
        # Upstream ships FastChem source only, so an ini_mix='EQ' oracle run needs a
        # build of upstream's own source (the oracle stays unmodified).
        if oracle_fastchem.is_dir() and not (oracle_fastchem / "fastchem").exists():
            shutil.copy2(_oracle_fastchem_binary(family, src),
                         oracle_fastchem / "fastchem")
        if fastchem_abundance is not None:
            abundance = (ROOT / "src" / "vulcan_jax" / "thermo"
                         / fastchem_abundance)
            if not abundance.is_file():
                raise FileNotFoundError(f"no such abundance preset: {abundance}")
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
