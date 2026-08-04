"""The upstream oracle must be revision-pinned, verified, and never mutated.

Acceptance for change 1 of the 2026-08 handoff:

  * the audit passes against the manifest revision, and fails BEFORE any
    calculation against a one-commit-newer or a dirty checkout;
  * a test records the oracle hash/status before and after and proves it did
    not change;
  * current public master fails with a clear "unsupported oracle revision"
    message, not a cascade of rate-index failures;
  * no newer chemistry network is copied into this repository.

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

def test_manifest_pins_a_full_commit_per_oracle_family():
    m = yaml.safe_load(MANIFEST.read_text())
    assert m["schema_version"] == 1
    for family, spec in m["oracles"].items():
        assert "repo" in spec, family
        commit = spec.get("commit", "")
        assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), (
            f"{family}: commit must be a full 40-hex sha, got {commit!r}")
        assert spec.get("why", "").strip(), f"{family}: needs a reason"


def test_manifest_states_the_networks_not_in_this_release():
    """A newer oracle must be recognisable as newer, not as a missing file."""
    m = yaml.safe_load(MANIFEST.read_text())
    excluded = {n.upper() for n in m["not_in_this_release"]["networks"]}
    assert any("CRAHCNO" in n for n in excluded)
    assert any("HNC" in n for n in excluded)
    assert any("NCCN" in n for n in excluded)


def test_no_excluded_network_was_copied_into_the_repo():
    """Acceptance: no newer chemistry network is copied into jax-vulcan."""
    thermo = ROOT / "src" / "vulcan_jax" / "thermo"
    bad = [p.name for p in thermo.glob("*")
           if "CRAHCNO" in p.name.upper() or "NCCN" in p.name.upper()]
    assert not bad, f"newer upstream network(s) present: {bad}"


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


def test_every_divergence_names_a_reason_and_a_record():
    m = yaml.safe_load(MANIFEST.read_text())
    for key, spec in m["intentional_divergences"].items():
        assert spec.get("reason", "").strip(), f"{key}: no reason"
        assert spec.get("recorded_in", "").strip(), f"{key}: no record location"


def test_divergence_references_resolve():
    """A `diverges:` tag on an input must name a defined divergence."""
    m = yaml.safe_load(MANIFEST.read_text())
    known = set(m["intentional_divergences"])
    for rel, spec in m["supported_inputs"].items():
        for tag in spec.get("diverges", []) or []:
            assert tag in known, f"{rel} references unknown divergence {tag!r}"


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


def test_one_commit_newer_still_fails(fake_oracle, monkeypatch):
    """Pinning is exact: 'nearly right' is not right.

    The real instance: current public master is one relevant commit past the
    pinned revision (39f1906 removed NH3 + CH), which shifts every reaction
    index after it.
    """
    monkeypatch.setenv(orc.ENV_DIR, str(fake_oracle))
    monkeypatch.setenv(orc.ENV_REQUIRE, "1")
    (fake_oracle / "thermo" / "NCHO_photo_network.txt").write_text("# v2\n")
    _git(fake_oracle, "commit", "-qam", "one commit newer")
    with pytest.raises(pytest.fail.Exception) as exc:
        orc.require_oracle("vulcan2_ncho")
    assert "UNSUPPORTED ORACLE REVISION" in str(exc.value)


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


def test_non_git_directory_is_refused(tmp_path, monkeypatch):
    """The local ../VULCAN-master is not a git checkout at all."""
    plain = tmp_path / "VULCAN-master"
    plain.mkdir()
    monkeypatch.setenv(orc.ENV_DIR, str(plain))
    monkeypatch.setenv(orc.ENV_REQUIRE, "1")
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
    finally:
        monkeypatch.setitem(orc.manifest()["oracles"]["vulcan2_ncho"],
                            "commit", want)


def test_worktree_raises_if_the_original_was_mutated(fake_oracle, monkeypatch):
    """The before/after fingerprint must actually catch a mutation."""
    monkeypatch.setenv(orc.ENV_DIR, str(fake_oracle))
    monkeypatch.setenv(orc.ENV_REQUIRE, "1")
    want = orc.oracle_spec("vulcan2_ncho")["commit"]
    monkeypatch.setitem(orc.manifest()["oracles"]["vulcan2_ncho"], "commit",
                        _git(fake_oracle, "rev-parse", "HEAD"))
    try:
        with pytest.raises(AssertionError, match="CHANGED during the test"):
            with orc.oracle_worktree("vulcan2_ncho"):
                # a test that (wrongly) writes into the real checkout
                (fake_oracle / "thermo" / "leak.txt").write_text("oops\n")
    finally:
        monkeypatch.setitem(orc.manifest()["oracles"]["vulcan2_ncho"],
                            "commit", want)
