"""Every required numerical oracle must be obtainable from THIS repository.

The four `tests/data/*.npz` fixtures are mandatory (conftest fails
collection when one is missing, since a skipped oracle still reports
green), so their generators must live in this repo and use no path outside
it. These structural tests pin the generator locations and the
manifest/verify machinery; they do not run the generators.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
GEN = TESTS / "gen_fixtures.py"


def _gen_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen_fixtures", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fixture_generators_live_in_repo():
    """conftest's required set and gen_fixtures' produced set must agree; no
    generator may point outside this repository or hardcode an absolute user
    path. String literals are checked in the AST, so docstring prose does not
    trip it."""
    import conftest

    gen = _gen_module()
    expected = {Path(q).name for q in conftest._EXPECTED_FIXTURES}
    produced = set(gen.ALL_FIXTURES)
    assert expected == produced, (
        f"conftest requires {sorted(expected)} but gen_fixtures produces "
        f"{sorted(produced)}; a fixture with no generator cannot be rebuilt")
    for rel, how in conftest._EXPECTED_FIXTURES.items():
        script = how.split()[0]
        assert not script.startswith(".."), (
            f"{rel} is regenerated from {script}, outside this repository")
        assert (ROOT / script).is_file(), f"{rel}: missing generator {script}"
    for script in ("tests/_gen_adj_state.py", "tests/_gen_photo_baseline.py",
                   "tests/gen_fixtures.py"):
        tree = ast.parse((ROOT / script).read_text(), script)
        bad = [
            n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and (n.value.startswith("/Users/") or n.value.startswith("/home/"))
        ]
        assert not bad, f"{script} hardcodes absolute user path(s): {bad}"


def test_fixture_manifest_verifies_and_records_provenance(tmp_path, monkeypatch):
    """verify() fails on a missing manifest and on tampered bytes, and the
    manifest records what produced each file, not just its hash."""
    gen = _gen_module()
    monkeypatch.setattr(gen, "MANIFEST", tmp_path / "nope.json")
    assert gen.verify() == 1

    data = tmp_path / "data"
    data.mkdir()
    for name in gen.ALL_FIXTURES:
        (data / name).write_bytes(b"placeholder-" + name.encode())
    monkeypatch.setattr(gen, "FIXTURE_DIR", data)
    monkeypatch.setattr(gen, "MANIFEST", data / "FIXTURES.json")

    m = gen.write_manifest()
    assert set(m["fixtures"]) == set(gen.ALL_FIXTURES)
    assert gen.verify() == 0
    assert "vulcan_jax_commit" in m
    assert m["versions"]["jax"] and m["versions"]["numpy"]
    assert m["config_sha256_16"], "shipped config identity must be recorded"
    entry = m["fixtures"][gen.ALL_FIXTURES[0]]
    assert entry["sha256"] and entry["generator"] and entry["note"]

    victim = data / gen.ALL_FIXTURES[0]
    victim.write_bytes(victim.read_bytes() + b"!")
    assert gen.verify() == 1
