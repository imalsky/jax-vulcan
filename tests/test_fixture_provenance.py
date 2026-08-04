"""Every required numerical oracle must be obtainable from THIS repository.

The four `tests/data/*.npz` fixtures are deliberately mandatory -- `conftest.py`
fails collection when one is missing, because a skipped numerical oracle still
reports a green suite. Until 2026-08-03 two of them could only be produced
through the unpublished sibling `jax_paper/`, so a clean clone could not run its
own default suite; and both of those generators had rotted, hardcoding a
`/Users/.../Desktop/Emulators/VULCAN_Project` root that no longer exists. The
fixtures were therefore unreproducible even WITH the sibling checked out.

These tests are cheap and structural: they pin that the generators live here,
take no path outside the repository, and that the manifest/verify machinery
works. They do NOT run the generators (each converges a real model).
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
GEN = TESTS / "gen_fixtures.py"


def _gen_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen_fixtures", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_expected_fixture_has_an_in_repo_generator():
    """conftest's required set and gen_fixtures' produced set must agree."""
    import conftest

    gen = _gen_module()
    expected = {Path(p).name for p in conftest._EXPECTED_FIXTURES}
    produced = set(gen.ALL_FIXTURES)
    assert expected == produced, (
        f"conftest requires {sorted(expected)} but gen_fixtures produces "
        f"{sorted(produced)}; a fixture with no generator cannot be rebuilt")


def test_generators_are_inside_this_repository():
    """No generator may point at a sibling checkout.

    This is the whole defect: `../jax_paper/scripts/adj_save_state.py` was the
    only route to two fixtures, and jax_paper is not published.
    """
    import conftest

    for rel, how in conftest._EXPECTED_FIXTURES.items():
        script = how.split()[0]
        assert not script.startswith(".."), (
            f"{rel} is regenerated from {script}, outside this repository")
        assert (ROOT / script).is_file(), f"{rel}: missing generator {script}"


@pytest.mark.parametrize("script", ["tests/_gen_adj_state.py",
                                    "tests/_gen_photo_baseline.py",
                                    "tests/gen_fixtures.py"])
def test_no_generator_hardcodes_an_absolute_user_path(script):
    """The exact rot that made the old generators unrunnable.

    Both jax_paper generators opened with
    `ROOT = Path("/Users/imalsky/Desktop/Emulators/VULCAN_Project")`. Checks
    string literals in the AST, so prose in a docstring describing the old bug
    does not trip it.
    """
    tree = ast.parse((ROOT / script).read_text(), script)
    bad = [
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and (n.value.startswith("/Users/") or n.value.startswith("/home/"))
    ]
    assert not bad, f"{script} hardcodes absolute user path(s): {bad}"


def test_gen_fixtures_verify_reports_missing_manifest(tmp_path, monkeypatch):
    gen = _gen_module()
    monkeypatch.setattr(gen, "MANIFEST", tmp_path / "nope.json")
    assert gen.verify() == 1


def test_gen_fixtures_verify_detects_a_tampered_fixture(tmp_path, monkeypatch):
    """A wrong-bytes fixture must fail verification, not pass quietly."""
    gen = _gen_module()
    data = tmp_path / "data"
    data.mkdir()
    for name in gen.ALL_FIXTURES:
        (data / name).write_bytes(b"placeholder-" + name.encode())
    monkeypatch.setattr(gen, "FIXTURE_DIR", data)
    monkeypatch.setattr(gen, "MANIFEST", data / "FIXTURES.json")

    manifest = gen.write_manifest()
    assert set(manifest["fixtures"]) == set(gen.ALL_FIXTURES)
    assert gen.verify() == 0

    victim = data / gen.ALL_FIXTURES[0]
    victim.write_bytes(victim.read_bytes() + b"!")
    assert gen.verify() == 1


def test_manifest_records_provenance_not_just_hashes(tmp_path, monkeypatch):
    """A hash alone does not say what produced the file."""
    gen = _gen_module()
    data = tmp_path / "data"
    data.mkdir()
    (data / gen.ALL_FIXTURES[0]).write_bytes(b"x")
    monkeypatch.setattr(gen, "FIXTURE_DIR", data)
    monkeypatch.setattr(gen, "MANIFEST", data / "FIXTURES.json")

    m = gen.write_manifest()
    assert "vulcan_jax_commit" in m
    assert m["versions"]["jax"] and m["versions"]["numpy"]
    assert m["config_sha256_16"], "shipped config identity must be recorded"
    entry = m["fixtures"][gen.ALL_FIXTURES[0]]
    assert entry["sha256"] and entry["generator"] and entry["note"]


def test_gen_fixtures_cli_help_runs():
    """The one documented command must at least be well-formed."""
    r = subprocess.run([sys.executable, str(GEN), "--help"],
                       capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "--all" in r.stdout and "--verify" in r.stdout


def test_allow_missing_fixtures_is_documented_as_developer_only():
    """The escape hatch must not read as a supported CI setting."""
    text = (TESTS / "conftest.py").read_text()
    assert "VULCAN_JAX_ALLOW_MISSING_FIXTURES" in text
    assert "must not be set in release CI" in text
    assert "gen_fixtures.py --all" in text
