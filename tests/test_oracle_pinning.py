"""The upstream oracle must be revision-pinned, verified, and never mutated.

Pins: the audit fails BEFORE any calculation on a one-commit-newer or dirty
checkout; the oracle hash/status is proven unchanged across a test; a wrong
revision gives one clear "unsupported oracle revision" message, not a cascade
of rate-index failures; no newer chemistry network is copied into this repo.

These run without any oracle present: the revision machinery is exercised on
throwaway git repositories built in tmp_path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

import oracle as orc

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tests" / "science_sources.yaml"


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=True, text=True, check=True,
                       env={"HOME": str(repo), "GIT_CONFIG_GLOBAL": "/dev/null",
                            "GIT_CONFIG_NOSYSTEM": "1", "PATH": "/usr/bin:/bin"})
    return r.stdout.strip()


@pytest.fixture
def fake_oracle(tmp_path):
    """A tiny git repo standing in for an upstream checkout."""
    repo = tmp_path / "VULCAN-oracle"
    (repo / "thermo").mkdir(parents=True)
    (repo / "thermo" / "NCHO_photo_network.txt").write_text("# network v1\n")
    _git_init = subprocess.run(
        ["git", "init", "-q", str(repo)], capture_output=True, text=True)
    assert _git_init.returncode == 0, _git_init.stderr
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo


# --- manifest shape ---------------------------------------------------------

def test_manifest_is_well_formed():
    """Commits are full pinned shas with reasons; newer-than-pin networks are
    declared excluded and none was copied in; every divergence tag resolves
    to a defined, recorded divergence."""
    m = yaml.safe_load(MANIFEST.read_text())
    assert m["schema_version"] == 1
    for family, spec in m["oracles"].items():
        assert "repo" in spec, family
        commit = spec.get("commit", "")
        assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), (
            f"{family}: commit must be a full 40-hex sha, got {commit!r}")
        assert spec.get("why", "").strip(), f"{family}: needs a reason"
    excluded = {n.upper() for n in m["not_in_this_release"]["networks"]}
    assert any("CRAHCNO" in n for n in excluded)
    thermo = ROOT / "src" / "vulcan_jax" / "thermo"
    bad = [q.name for q in thermo.glob("*")
           if "CRAHCNO" in q.name.upper() or "NCCN" in q.name.upper()]
    assert not bad, f"newer upstream network(s) present: {bad}"
    known = set(m["intentional_divergences"])
    for key, spec in m["intentional_divergences"].items():
        assert spec.get("reason", "").strip(), f"{key}: no reason"
        assert spec.get("recorded_in", "").strip(), f"{key}: no record location"
    for rel, spec in m["supported_inputs"].items():
        for tag in spec.get("diverges", []) or []:
            assert tag in known, f"{rel} references unknown divergence {tag!r}"


def test_every_supported_input_exists_with_its_recorded_hash():
    """The positive half of the audit, checkable with no oracle at all."""
    import hashlib

    m = yaml.safe_load(MANIFEST.read_text())
    pkg = ROOT / "src" / "vulcan_jax"
    for rel, spec in m["supported_inputs"].items():
        p = pkg / rel
        assert p.is_file(), f"supported input missing: {rel}"
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        assert got == spec["sha256"], (
            f"{rel}: vendored input changed (sha256 {got[:16]}... != recorded "
            f"{spec['sha256'][:16]}...)")


# --- revision + cleanliness enforcement -------------------------------------

def test_missing_oracle_dir_skips_locally_but_fails_in_release_ci(monkeypatch):
    monkeypatch.delenv(orc.ENV_DIR, raising=False)
    monkeypatch.delenv(orc.ENV_REQUIRE, raising=False)
    with pytest.raises(pytest.skip.Exception) as exc:
        orc.require_oracle("vulcan2_ncho")
    assert "VULCAN_MASTER_DIR" in str(exc.value)

    # ...but a release CI job sets VULCAN_JAX_REQUIRE_ORACLE=1, where a
    # missing oracle must FAIL: skipped != passed.
    monkeypatch.setenv(orc.ENV_REQUIRE, "1")
    with pytest.raises(pytest.fail.Exception) as exc:
        orc.require_oracle("vulcan2_ncho")
    assert "VULCAN_MASTER_DIR" in str(exc.value)


def test_wrong_revision_fails_with_expected_and_actual(fake_oracle, monkeypatch):
    """The headline behavior: one clear message, before any calculation."""
    monkeypatch.setenv(orc.ENV_DIR, str(fake_oracle))
    monkeypatch.setenv(orc.ENV_REQUIRE, "1")
    head = _git(fake_oracle, "rev-parse", "HEAD")
    with pytest.raises(pytest.fail.Exception) as exc:
        orc.require_oracle("vulcan2_ncho")
    msg = str(exc.value)
    assert "UNSUPPORTED ORACLE REVISION" in msg
    assert orc.oracle_spec("vulcan2_ncho")["commit"] in msg   # expected
    assert head in msg                                        # actual
    assert "POSITIONAL" in msg          # says WHY, not just that it differs


def test_dirty_checkout_fails_and_says_why(fake_oracle, monkeypatch):
    """A dirty tree is often the residue of a test that mutated the oracle."""
    monkeypatch.setenv(orc.ENV_DIR, str(fake_oracle))
    monkeypatch.setenv(orc.ENV_REQUIRE, "1")
    want = orc.oracle_spec("vulcan2_ncho")["commit"]
    monkeypatch.setitem(orc.manifest()["oracles"]["vulcan2_ncho"], "commit",
                        _git(fake_oracle, "rev-parse", "HEAD"))
    try:
        (fake_oracle / "thermo" / "NCHO_photo_network.txt").write_text("edited\n")
        with pytest.raises(pytest.fail.Exception) as exc:
            orc.require_oracle("vulcan2_ncho")
        msg = str(exc.value)
        assert "DIRTY" in msg
        assert "make_chem_funs" in msg      # names the real mutation source
    finally:
        monkeypatch.setitem(orc.manifest()["oracles"]["vulcan2_ncho"],
                            "commit", want)
    # ... and a plain directory (the local ../VULCAN-master) is refused too
    plain = fake_oracle.parent / "plain-copy"
    plain.mkdir()
    monkeypatch.setenv(orc.ENV_DIR, str(plain))
    with pytest.raises(pytest.fail.Exception) as exc:
        orc.require_oracle("vulcan2_ncho")
    assert "not a git checkout" in str(exc.value)


# --- non-mutation ------------------------------------------------------------

def test_worktree_gives_a_copy_and_proves_the_original_is_untouched(
        fake_oracle, monkeypatch):
    """Upstream setup code rewrites files in place; it must only see the copy."""
    monkeypatch.setenv(orc.ENV_DIR, str(fake_oracle))
    monkeypatch.setenv(orc.ENV_REQUIRE, "1")
    want = orc.oracle_spec("vulcan2_ncho")["commit"]
    monkeypatch.setitem(orc.manifest()["oracles"]["vulcan2_ncho"], "commit",
                        _git(fake_oracle, "rev-parse", "HEAD"))
    original = (fake_oracle / "thermo" / "NCHO_photo_network.txt").read_bytes()
    try:
        with orc.oracle_worktree("vulcan2_ncho") as work:
            assert work != fake_oracle
            # simulate make_chem_funs renumbering the network IN PLACE
            (work / "thermo" / "NCHO_photo_network.txt").write_text("renumbered\n")
        assert (fake_oracle / "thermo"
                / "NCHO_photo_network.txt").read_bytes() == original
        # ... and the before/after fingerprint actually catches a mutation
        with pytest.raises(AssertionError, match="CHANGED during the test"):
            with orc.oracle_worktree("vulcan2_ncho"):
                (fake_oracle / "thermo" / "leak.txt").write_text("oops\n")
    finally:
        monkeypatch.setitem(orc.manifest()["oracles"]["vulcan2_ncho"],
                            "commit", want)


def test_the_oracle_workflow_matches_the_manifest():
    """.github/workflows/oracle.yml must pin the manifest's exact commits.

    The workflow clones upstream itself, so its SHAs are a second copy of
    `science_sources.yaml`. If they drift, the release gate silently compares
    against the wrong revision -- and because reaction indices are positional,
    that produces failures (or passes) unrelated to the ported kernels.

    Also checks every selected path exists, so a renamed test file cannot
    quietly drop out of the gate.
    """
    import yaml as _yaml

    wf_path = ROOT / ".github" / "workflows" / "oracle.yml"
    assert wf_path.is_file(), f"missing {wf_path}"
    wf = _yaml.safe_load(wf_path.read_text())
    include = wf["jobs"]["oracle"]["strategy"]["matrix"]["include"]
    manifest = _yaml.safe_load(
        (ROOT / "tests" / "science_sources.yaml").read_text())["oracles"]

    covered = set()
    for entry in include:
        family = entry["family"]
        assert family in manifest, (
            f"oracle.yml runs family {family!r}, absent from "
            f"science_sources.yaml")
        assert entry["commit"] == manifest[family]["commit"], (
            f"oracle.yml pins {family} at {entry['commit']} but the manifest "
            f"says {manifest[family]['commit']}; update both together")
        covered.add(family)
        for rel in entry["select"].split():
            assert (ROOT / rel).is_file(), (
                f"oracle.yml selects {rel}, which does not exist")

    # vulcan3_vm_branch has no test file yet; only assert what is claimed.
    used_in_tests = set()
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        for fam in manifest:
            if f'"{fam}"' in f.read_text():
                used_in_tests.add(fam)
    missing = sorted(used_in_tests - covered)
    assert not missing, (
        f"oracle families {missing} are used by tests but no oracle.yml job "
        f"runs them, so the release gate would never exercise them")
