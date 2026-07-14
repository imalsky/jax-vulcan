"""YAML-backed configuration for VULCAN-JAX.

The config surface is authored as YAML. Canonical configs ship inside the
package (``vulcan_jax/configs/*.yaml``) and are overridable by a ``./configs/``
directory in the current working directory (CWD-first), so a run can be tuned
locally without editing packaged files.

``load_config(name_or_path, **overrides)`` reads a YAML file, folds the
import-frozen environment overrides, applies caller overrides, resolves the
handful of values that are functions of other knobs, and returns a ``Config``
namespace whose attribute surface is exactly what the runtime reads
(``cfg.nz``, ``cfg.network``, ...). ``default_config()`` is the process-wide
default, loaded once from ``configs/default.yaml`` at first ``import
vulcan_jax`` — it resolves the import-frozen knobs (``network`` / ``atom_list``
/ ``com_file``), honoring the ``$VULCAN_JAX_*`` overrides, exactly as the old
``vulcan_cfg`` module did at import time.
"""

from __future__ import annotations

import copy
import os
import re
import types
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


class _StrictLoader(yaml.SafeLoader):
    """Strict YAML loader for config files.

    Two departures from ``yaml.safe_load``, both to fail loudly instead of
    silently doing the wrong thing:

    1. Unsigned-exponent scientific notation (``1e22``, ``5.0e21``, ``1e-14``)
       parses as a float. Stock PyYAML (YAML 1.1 core schema) treats these as
       *strings*, which would silently poison numeric knobs a user hand-authors
       in a ``./configs`` override.
    2. Duplicate keys raise instead of last-wins.
    """

    def construct_mapping(self, node, deep=False):
        seen: set = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate config key {key!r}", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


# Float regex extended with the `1e22` (no dot / unsigned exponent) form.
_FLOAT_RE = re.compile(
    r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?\.[0-9][0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
    re.X,
)
_StrictLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float", _FLOAT_RE, list("-+0123456789.")
)

# Import-frozen knobs (read at the first `import vulcan_jax`): each honors an
# environment override so a non-default network / atom set / composition can be
# selected before import. Mirrors the historical vulcan_cfg module behavior;
# `com_file` is newly overridable for symmetry.
_FROZEN_ENV = {
    "network": "VULCAN_JAX_NETWORK",
    "atom_list": "VULCAN_JAX_ATOM_LIST",
    "com_file": "VULCAN_JAX_COM_FILE",
}

# Values that are functions of other knobs. Authored YAML omits them; an
# explicit YAML/override value always wins (they are only filled when absent).
_DERIVED = (
    ("dt_max", lambda d: d["runtime"] * 1e-5),
    ("photo_switch_longdy_thresh", lambda d: d["yconv_min"] * 10.0),
    ("save_movie_rate", lambda d: d["live_plot_frq"]),
    ("para_anaTP", lambda d: copy.deepcopy(d["para_warm"])),
)


class Config(types.SimpleNamespace):
    """Free-attribute config namespace (drop-in for the old ``vulcan_cfg`` module).

    A ``SimpleNamespace`` subclass so ``vars(cfg)``, ``getattr(cfg, name,
    default)``, ``setattr``, dynamic keys (``getattr(cfg, sp + "_H")``), and
    ``copy.deepcopy`` all behave exactly as the reflection layer
    (``make_config`` / ``state._cfg_overlay`` / ``legacy_io.Output.save_cfg``)
    expects.
    """


def _read_text(name_or_path: str | os.PathLike) -> str:
    """Resolve a config reference to YAML text.

    Resolution order: an explicit path or ``*.yaml``/``*.yml`` file; else a bare
    name resolved CWD-first (``./configs/<name>.yaml``) then from the packaged
    ``vulcan_jax/configs/<name>.yaml``.
    """
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") or p.exists():
        if not p.is_file():
            raise FileNotFoundError(f"config file not found: {name_or_path}")
        return p.read_text()

    fname = f"{name_or_path}.yaml"
    cwd = Path.cwd() / "configs" / fname
    if cwd.is_file():
        return cwd.read_text()

    pkg = resources.files("vulcan_jax").joinpath("configs", fname)
    if pkg.is_file():
        return pkg.read_text()

    raise FileNotFoundError(
        f"no config named {name_or_path!r}: looked for ./configs/{fname} and "
        f"the packaged vulcan_jax/configs/{fname}."
    )


def _apply_frozen_env(d: dict[str, Any]) -> None:
    """Overlay the import-frozen knobs from ``$VULCAN_JAX_*`` (env wins)."""
    for key, env in _FROZEN_ENV.items():
        val = os.environ.get(env)
        if val is None:
            continue
        if key == "atom_list":
            d[key] = [s for s in val.split(",") if s]
        else:
            d[key] = val


def _resolve_derived(d: dict[str, Any]) -> None:
    """Fill derived values (only when absent) and normalize a couple of types."""
    for key, fn in _DERIVED:
        if d.get(key) is None:
            d[key] = fn(d)
    if d.get("count_max") is not None:
        # Accept scientific-notation literals (e.g. 1e4) as an integer step cap.
        d["count_max"] = int(d["count_max"])


def load_config(name_or_path: str | os.PathLike = "default", /, **overrides: Any) -> Config:
    """Load a YAML config, apply env + caller overrides, resolve derived values.

    Args:
        name_or_path: a bare name (``"default"``, ``"W39b"``; resolved CWD-first
            then packaged) or an explicit path to a ``.yaml`` file.
        **overrides: attribute overrides applied after the YAML (and after the
            frozen-env fold), before derived resolution.

    Returns:
        A ``Config`` namespace with the full runtime attribute surface.
    """
    raw = yaml.load(_read_text(name_or_path), Loader=_StrictLoader)
    if not isinstance(raw, dict):
        raise ValueError(f"config {name_or_path!r} did not parse to a mapping.")
    d: dict[str, Any] = dict(raw)
    _apply_frozen_env(d)
    d.update(overrides)
    _resolve_derived(d)
    return Config(**d)


def dump_config(cfg: Config, path: str | os.PathLike) -> None:
    """Write a resolved Config back to YAML so a run can be reproduced exactly.

    Dumps the full attribute set (including the derived values), so re-loading
    the file with ``load_config`` reconstructs the same run inputs verbatim.
    """
    d = {
        k: v
        for k, v in vars(cfg).items()
        if not k.startswith("_") and not callable(v)
    }
    Path(path).write_text(
        yaml.safe_dump(d, sort_keys=False, default_flow_style=False, width=100)
    )


_DEFAULT: Config | None = None


def default_config() -> Config:
    """The process-wide default Config (loaded once from ``configs/default.yaml``).

    Resolves the import-frozen knobs, folding any ``$VULCAN_JAX_*`` overrides,
    exactly as the old ``vulcan_cfg`` module did at import time. Cached: the
    same object is returned every call, so ``state._cfg_overlay`` can identity-
    compare against it for its no-op fast path.
    """
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = load_config("default")
    return _DEFAULT
