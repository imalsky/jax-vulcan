"""Typed pre-loop input pytrees + host-side stellar-flux loader.

Phase 19 schema. Defines the public groupings of pre-loop setup output —
`AtmInputs` (atmosphere structure), `RateInputs` (rate constants),
`PhotoInputs` (stellar flux extents) — that the JAX runner ultimately
consumes. They are JAX-friendly NamedTuples so `jit`/`vmap` see them as
standard pytrees with no registration boilerplate.

These pytrees coexist with the legacy mutable `store.Variables` /
`store.AtmData` containers; callers fill them via `pytree_from_store`.
Phases 20+ will fill them directly from JAX-native setup code, at which
point the legacy containers become a thin compatibility layer.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

import chem_funs


class AtmInputs(NamedTuple):
    """Atmosphere structure read by the runner.

    Field set covers every `AtmData` array that `outer_loop.OuterLoop` /
    `jax_step.jax_ros2_step` / `chem.chem_rhs` / diffusion kernels read.
    Shapes assume `nz` vertical layers and `ni` species; `(nz-1)` arrays
    live on staggered interfaces.
    """
    pco:        jnp.ndarray   # (nz,)    — pressure on cells (dyne/cm^2)
    pico:       jnp.ndarray   # (nz+1,)  — pressure on interfaces
    Tco:        jnp.ndarray   # (nz,)    — temperature (K)
    Kzz:        jnp.ndarray   # (nz-1,)  — eddy diffusion (cm^2/s)
    vz:         jnp.ndarray   # (nz-1,)  — vertical advection (cm/s)
    M:          jnp.ndarray   # (nz,)    — third-body number density
    n_0:        jnp.ndarray   # (nz,)    — total number density
    mu:         jnp.ndarray   # (nz,)    — mean molecular weight
    g:          jnp.ndarray   # (nz,)    — local gravity
    Hp:         jnp.ndarray   # (nz,)    — pressure scale height
    dz:         jnp.ndarray   # (nz,)    — layer thickness
    dzi:        jnp.ndarray   # (nz-1,)  — interface thickness
    zco:        jnp.ndarray   # (nz+1,)  — height at interfaces
    zmco:       jnp.ndarray   # (nz,)    — height at cell centers
    ms:         jnp.ndarray   # (ni,)    — molecular weight per species
    alpha:      jnp.ndarray   # (ni,)    — thermal diffusion factor
    Dzz:        jnp.ndarray   # (nz-1, ni) — molecular diffusion (interface)
    Dzz_cen:    jnp.ndarray   # (nz, ni)   — molecular diffusion (cell)
    vm:         jnp.ndarray   # (nz, ni)   — thermal diffusion velocity
    vs:         jnp.ndarray   # (nz-1, ni) — settling velocity
    top_flux:   jnp.ndarray   # (ni,) — top BC flux (#/cm^2/s)
    bot_flux:   jnp.ndarray   # (ni,) — bottom BC flux
    bot_vdep:   jnp.ndarray   # (ni,) — bottom deposition velocity
    bot_fix_sp: jnp.ndarray   # (ni,) — bottom fixed-mixing-ratio mask


class RateInputs(NamedTuple):
    """Rate constants packed for the runner.

    `k` is a dense `(nr+1, nz)` array where `k[i]` is the rate constant
    of reaction `i` at every layer. Index 0 is unused (reactions are
    1-based throughout VULCAN); kept as a row so dict→array indexing
    needs no offset.
    """
    k:          jnp.ndarray   # (nr+1, nz) — forward rate constants


class IniAbunOutputs(NamedTuple):
    """JAX-native initial-abundance solver output (Phase 21).

    Produced by `ini_abun.compute_initial_abundance(cfg, atm_inputs)`,
    one of `ini_mix ∈ {EQ, const_mix, vulcan_ini, table, const_lowT}`.
    Mirrors the fields the legacy `InitialAbun.ini_y` / `ele_sum` wrote
    onto `data_var`; the facade in `ini_abun.py` reads from this and
    pushes the values into `store.Variables` for back-compat.

    Shapes assume `nz` vertical layers, `ni` species, `n_atoms` elements
    (same length as `composition.atom_list`).
    """
    y:           jnp.ndarray         # (nz, ni)    — number density
    ymix:        jnp.ndarray         # (nz, ni)    — mixing ratio
    y_ini:       jnp.ndarray         # (nz, ni)    — initial-state snapshot
    atom_ini:    jnp.ndarray         # (n_atoms,)  — sum of stoich * y per atom
    atom_loss:   jnp.ndarray         # (n_atoms,)  — zeros at init
    atom_conden: jnp.ndarray         # (n_atoms,)  — zeros at init
    charge_list: tuple[str, ...]     # charged species (empty when use_ion=False)


class PhotoInputs(NamedTuple):
    """Stellar / photo inputs that the pre-loop setup produces.

    `sflux_top` is the binned stellar flux at the top of atmosphere; it
    is shape `(nbin,)` when `use_photo=True` and shape `(0,)` otherwise.
    `def_bin_min`/`def_bin_max` are the wavelength extents established
    by the stellar flux read (later narrowed by the photo cross-section
    setup).
    """
    sflux_top:   jnp.ndarray   # (nbin,) — TOA stellar flux; empty if no photo
    def_bin_min: float
    def_bin_max: float


class PhotoStaticInputs(NamedTuple):
    """Dense photo cross-section pytree (Phase 22e).

    Carries everything the runtime kernels (`photo.compute_tau_jax` /
    `compute_flux_jax` / `compute_J_jax_flat`) and the `.vul` writer
    need; nothing else. Replaces the legacy `var.cross` / `var.cross_J` /
    `var.cross_T` / `var.cross_J_T` / `var.cross_Jion` / `var.cross_scat`
    Python-dict surface that Phase 22b retained for back-compat.

    `absp_cross` carries a row for *every* species in
    `absp_sp ∪ absp_T_sp` -- T-dep species also keep their 1D room-T
    cross. This matches the legacy `var.cross[sp]` semantics (1D for
    all photo+ion species, including T-dep ones) and removes a special
    case from the `.vul` synthesizer.

    `din12_indx` is `-1` until the post-photo `read_sflux` write fires;
    use `with_din12_indx(idx)` to attach it.
    """
    bins:           jnp.ndarray   # (nbin,)  float64 wavelength grid (nm)
    nbin:           int
    dbin1:          float
    dbin2:          float
    din12_indx:     int           # -1 sentinel pre-read_sflux

    absp_sp:           tuple        # (str, ...)  ordered absorbers (photo ∪ ion)
    absp_T_sp:         tuple        # (str, ...)  T-dep subset of absp_sp
    scat_sp:           tuple        # (str, ...)  Rayleigh scatterers
    branch_keys:       tuple        # ((sp, branch_idx), ...) non-T dissociation
    branch_T_keys:     tuple        # ((sp, branch_idx), ...) T-dep counterparts
    ion_branch_keys:   tuple        # ((sp, branch_idx), ...) ion branches

    absp_cross:    jnp.ndarray   # (n_absp, nbin)        room-T cross per absorber
    absp_T_cross:  jnp.ndarray   # (n_absp_T, nz, nbin)  T-dep total cross
    scat_cross:    jnp.ndarray   # (n_scat, nbin)
    cross_J:       jnp.ndarray   # (n_br, nbin)
    cross_J_T:     jnp.ndarray   # (n_br_T, nz, nbin)
    cross_Jion:    jnp.ndarray   # (n_ion_br, nbin)  -- (0, nbin) when use_ion=False

    def with_din12_indx(self, idx: int) -> "PhotoStaticInputs":
        """Return a copy with `din12_indx` set; asserts `idx >= 0` to
        catch the "used before `read_sflux`" ordering bug early."""
        if int(idx) < 0:
            raise ValueError(
                f"PhotoStaticInputs.din12_indx must be >= 0; got {idx}"
            )
        return self._replace(din12_indx=int(idx))


class RunState(NamedTuple):
    """Umbrella pytree carrying every pre-loop input the runner reads.

    Phase 19 schema; round-tripped through `pytree_from_store` /
    `apply_pytree_to_store`. Phases 20+ will produce one of these
    directly from JAX-native setup, at which point the legacy
    `(var, atm)` containers become a compatibility shim.
    """
    atm:   AtmInputs
    rate:  RateInputs
    photo: PhotoInputs


class StellarFlux(NamedTuple):
    """Pre-loop stellar-flux read result.

    Returned by `load_stellar_flux(cfg)`. Phase 19 moves the
    `np.genfromtxt(cfg.sflux_file, ...)` call out of `Variables.__init__`
    and into an explicit host-side function so the constructor stays
    free of file I/O.
    """
    wavelength_nm: np.ndarray  # (n_lambda,) — raw stellar wavelength grid (nm)
    flux:          np.ndarray  # (n_lambda,) — raw stellar flux density
    def_bin_min:   float       # bin floor: max(lambda[0], 2.0)
    def_bin_max:   float       # bin ceil:  min(lambda[-1], 700.0)


def load_stellar_flux(cfg) -> StellarFlux:
    """Read the stellar flux file and compute the spectral-bin extents.

    Mirrors `store.Variables.__init__`'s historical np.genfromtxt block
    (vendored from VULCAN-master). Returns a `StellarFlux` payload that
    callers pass to `Variables(stellar_flux=...)`.

    Returns an empty payload (`def_bin_min=0`, `def_bin_max=0`,
    zero-length arrays) when `cfg.use_photo` is False so callers can
    unconditionally call this without branching.
    """
    if not bool(getattr(cfg, "use_photo", False)):
        return StellarFlux(
            wavelength_nm=np.zeros((0,), dtype=np.float64),
            flux=np.zeros((0,), dtype=np.float64),
            def_bin_min=0.0,
            def_bin_max=0.0,
        )
    sflux_data = np.genfromtxt(
        cfg.sflux_file,
        dtype=float,
        skip_header=1,
        names=["lambda", "flux"],
    )
    wavelength = np.asarray(sflux_data["lambda"], dtype=np.float64)
    flux = np.asarray(sflux_data["flux"], dtype=np.float64)
    # Same clamp as the original constructor: spectral bin in [2, 700] nm.
    def_bin_min = float(max(wavelength[0], 2.0))
    def_bin_max = float(min(wavelength[-1], 700.0))
    return StellarFlux(
        wavelength_nm=wavelength,
        flux=flux,
        def_bin_min=def_bin_min,
        def_bin_max=def_bin_max,
    )


# ---------------------------------------------------------------------------
# Adapters between the legacy mutable containers and the typed pytree.
# ---------------------------------------------------------------------------

def pytree_from_store(var, atm) -> RunState:
    """Snapshot the runner-input fields of a `(var, atm)` pair into a
    typed pytree.

    Used today as the bridge between the legacy `store.Variables` /
    `store.AtmData` containers and the typed schema. Phases 20+ will
    produce a `RunState` directly from JAX-native setup, at which point
    this function becomes the bridge for legacy callers (tests etc.).
    """
    nr = int(chem_funs.nr)
    nz = int(atm.Tco.shape[0])

    atm_inputs = AtmInputs(
        pco=jnp.asarray(atm.pco),
        pico=jnp.asarray(atm.pico),
        Tco=jnp.asarray(atm.Tco),
        Kzz=jnp.asarray(atm.Kzz),
        vz=jnp.asarray(atm.vz),
        M=jnp.asarray(atm.M),
        n_0=jnp.asarray(atm.n_0),
        mu=jnp.asarray(atm.mu),
        g=jnp.asarray(atm.g),
        Hp=jnp.asarray(atm.Hp),
        dz=jnp.asarray(atm.dz),
        dzi=jnp.asarray(atm.dzi),
        zco=jnp.asarray(atm.zco),
        zmco=jnp.asarray(atm.zmco),
        ms=jnp.asarray(atm.ms),
        alpha=jnp.asarray(atm.alpha),
        Dzz=jnp.asarray(atm.Dzz),
        Dzz_cen=jnp.asarray(atm.Dzz_cen),
        vm=jnp.asarray(atm.vm),
        vs=jnp.asarray(atm.vs),
        top_flux=jnp.asarray(atm.top_flux),
        bot_flux=jnp.asarray(atm.bot_flux),
        bot_vdep=jnp.asarray(atm.bot_vdep),
        bot_fix_sp=jnp.asarray(atm.bot_fix_sp),
    )

    # Phase 22d: read the dense `(nr+1, nz)` array directly off
    # `var.k_arr`. The legacy `var.k` dict surface was retired; the
    # rate setup path (`rates.setup_var_k`) and all subsequent writers
    # (photo / conden) target `var.k_arr` in place.
    k_dense = np.asarray(var.k_arr, dtype=np.float64)
    if k_dense.shape != (nr + 1, nz):
        raise ValueError(
            f"var.k_arr shape {k_dense.shape} != expected ({nr+1}, {nz})"
        )
    rate_inputs = RateInputs(k=jnp.asarray(k_dense))

    sflux_top = getattr(var, "sflux_top", None)
    if sflux_top is None:
        sflux_top_arr = jnp.zeros((0,), dtype=jnp.float64)
    else:
        sflux_top_arr = jnp.asarray(sflux_top, dtype=jnp.float64).reshape(-1)
    photo_inputs = PhotoInputs(
        sflux_top=sflux_top_arr,
        def_bin_min=float(getattr(var, "def_bin_min", 0.0)),
        def_bin_max=float(getattr(var, "def_bin_max", 0.0)),
    )

    return RunState(atm=atm_inputs, rate=rate_inputs, photo=photo_inputs)


def apply_pytree_to_store(state: RunState, var, atm) -> None:
    """Reverse adapter: write the pytree fields back into the legacy
    `(var, atm)` containers in place.

    Phase 22d: writes the dense rate array to `var.k_arr` (the legacy
    `var.k` dict surface was retired). Atm arrays are stored as numpy
    ndarrays so the legacy containers remain ndarray-typed.
    """
    atm.pco        = np.asarray(state.atm.pco)
    atm.pico       = np.asarray(state.atm.pico)
    atm.Tco        = np.asarray(state.atm.Tco)
    atm.Kzz        = np.asarray(state.atm.Kzz)
    atm.vz         = np.asarray(state.atm.vz)
    atm.M          = np.asarray(state.atm.M)
    atm.n_0        = np.asarray(state.atm.n_0)
    atm.mu         = np.asarray(state.atm.mu)
    atm.g          = np.asarray(state.atm.g)
    atm.Hp         = np.asarray(state.atm.Hp)
    atm.dz         = np.asarray(state.atm.dz)
    atm.dzi        = np.asarray(state.atm.dzi)
    atm.zco        = np.asarray(state.atm.zco)
    atm.zmco       = np.asarray(state.atm.zmco)
    atm.ms         = np.asarray(state.atm.ms)
    atm.alpha      = np.asarray(state.atm.alpha)
    atm.Dzz        = np.asarray(state.atm.Dzz)
    atm.Dzz_cen    = np.asarray(state.atm.Dzz_cen)
    atm.vm         = np.asarray(state.atm.vm)
    atm.vs         = np.asarray(state.atm.vs)
    atm.top_flux   = np.asarray(state.atm.top_flux)
    atm.bot_flux   = np.asarray(state.atm.bot_flux)
    atm.bot_vdep   = np.asarray(state.atm.bot_vdep)
    atm.bot_fix_sp = np.asarray(state.atm.bot_fix_sp)

    var.k_arr = np.asarray(state.rate.k, dtype=np.float64)

    if int(state.photo.sflux_top.shape[0]) > 0:
        var.sflux_top = np.asarray(state.photo.sflux_top)
    var.def_bin_min = float(state.photo.def_bin_min)
    var.def_bin_max = float(state.photo.def_bin_max)
