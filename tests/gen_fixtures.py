"""Obtain every numerical-regression fixture the test suite requires.

    python tests/gen_fixtures.py --all

That one command, run in a clean clone with no sibling repositories, produces
all four fixtures and writes `tests/data/FIXTURES.json` recording each file's
SHA-256 alongside the commit, config, and dependency versions that made it.

The fixtures are deliberately required -- `conftest.py` fails collection when
one is missing, because a skipped numerical oracle still reports green -- but
they are gitignored (~36 MB), so a clean clone must be able to regenerate them.

ONE PROCESS PER NETWORK. The reaction network is frozen at the first
`import vulcan_jax`, and the HD189 and W39b adjoint states need different
networks (C-H-N-O vs SNCHO). Each generator therefore runs as a subprocess.

COST. The two adjoint states each converge a real run; expect a few minutes
each on a laptop. The two photo fixtures are setup-only and fast.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "data"
MANIFEST = FIXTURE_DIR / "FIXTURES.json"

# fixture file -> (generator argv relative to ROOT, which fixtures it writes).
# Grouped by generator so one subprocess can emit several files.
GENERATORS = {
    "photo": {
        "argv": [sys.executable, "tests/_gen_photo_baseline.py"],
        "writes": ["photo_setup_hd189_baseline.npz",
                   "photo_setup_hd189_T_dep.npz"],
        "note": "HD189 photo cross-section setup (default and T-dependent)",
    },
    "adj_hd189": {
        "argv": [sys.executable, "tests/_gen_adj_state.py", "hd189"],
        "writes": ["adj_state_hd189.npz"],
        "note": "HD189 converged state, photo OFF, C-H-N-O network",
    },
    "adj_w39b": {
        "argv": [sys.executable, "tests/_gen_adj_state.py", "w39b"],
        "writes": ["adj_state_w39b.npz"],
        "note": "WASP-39 b converged state, photo ON (frozen), SNCHO network",
    },
}

ALL_FIXTURES = [f for g in GENERATORS.values() for f in g["writes"]]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        head = r.stdout.strip()
        if r.returncode != 0 or not head:
            return None
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return None


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for mod in ("jax", "jaxlib", "numpy", "scipy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    try:
        from importlib.metadata import version
        out["vulcan_jax"] = version("vulcan-jax")
    except Exception:
        out["vulcan_jax"] = None
    return out


def _config_identity() -> dict:
    """Hash the shipped configs the generators read.

    A fixture built from a different config is a different oracle, and the
    failure mode is a confusing numerical mismatch rather than a clear error.
    """
    cfg_dir = ROOT / "src" / "vulcan_jax" / "configs"
    out = {}
    for name in ("default.yaml", "W39b.yaml", "HD189.yaml"):
        p = cfg_dir / name
        if p.is_file():
            out[name] = sha256(p)[:16]
    return out


def run_generator(key: str) -> None:
    spec = GENERATORS[key]
    print(f"\n=== {key}: {spec['note']} ===", flush=True)
    env = dict(os.environ)
    # PYTHONSAFEPATH strips the cwd from sys.path in some sandboxes, which makes
    # the generators fail for reasons unrelated to the repo.
    env.pop("PYTHONSAFEPATH", None)
    t0 = time.time()
    r = subprocess.run(spec["argv"], cwd=ROOT, env=env)
    if r.returncode != 0:
        raise SystemExit(
            f"gen_fixtures: generator {key} failed (exit {r.returncode}). "
            f"Command: {' '.join(spec['argv'])} (cwd={ROOT})")
    print(f"=== {key}: done in {time.time() - t0:.1f}s ===", flush=True)


def write_manifest() -> dict:
    """Record hashes + provenance for every fixture present."""
    entries = {}
    for key, spec in GENERATORS.items():
        for name in spec["writes"]:
            p = FIXTURE_DIR / name
            if not p.is_file():
                continue
            entries[name] = {
                "sha256": sha256(p),
                "bytes": p.stat().st_size,
                "generator": " ".join(spec["argv"][1:]),
                "note": spec["note"],
            }
    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vulcan_jax_commit": _git_head(),
        "versions": _versions(),
        "config_sha256_16": _config_identity(),
        "fixtures": entries,
    }
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {MANIFEST}")
    return manifest


def verify() -> int:
    """Check every fixture against the manifest. Returns an exit code."""
    if not MANIFEST.is_file():
        print(f"gen_fixtures: no manifest at {MANIFEST}; run --all first",
              file=sys.stderr)
        return 1
    manifest = json.loads(MANIFEST.read_text())
    recorded = manifest.get("fixtures", {})
    bad = []
    for name in ALL_FIXTURES:
        p = FIXTURE_DIR / name
        if not p.is_file():
            bad.append(f"{name}: MISSING")
            continue
        if name not in recorded:
            bad.append(f"{name}: present but not in the manifest")
            continue
        got = sha256(p)
        if got != recorded[name]["sha256"]:
            bad.append(f"{name}: sha256 {got[:16]}... != recorded "
                       f"{recorded[name]['sha256'][:16]}...")
    if bad:
        print("gen_fixtures: verification FAILED", file=sys.stderr)
        for b in bad:
            print(f"  - {b}", file=sys.stderr)
        return 1
    print(f"gen_fixtures: all {len(ALL_FIXTURES)} fixtures match the manifest "
          f"(built from commit {manifest.get('vulcan_jax_commit')})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true",
                   help="build every missing fixture, then write the manifest")
    g.add_argument("--only", choices=sorted(GENERATORS),
                   help="run one generator")
    g.add_argument("--verify", action="store_true",
                   help="check the existing fixtures against the manifest")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even when the fixture is already present")
    args = ap.parse_args(argv)

    if args.verify:
        return verify()

    keys = sorted(GENERATORS) if args.all else [args.only]
    for key in keys:
        have = all((FIXTURE_DIR / n).is_file() for n in GENERATORS[key]["writes"])
        if have and not args.force:
            print(f"=== {key}: already present, skipping (--force to rebuild) "
                  "===", flush=True)
            continue
        run_generator(key)

    missing = [n for n in ALL_FIXTURES if not (FIXTURE_DIR / n).is_file()]
    if missing and args.all:
        print("gen_fixtures: generators ran but these are still missing: "
              + ", ".join(missing), file=sys.stderr)
        return 1
    write_manifest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
