"""Typed pre-loop input pytrees and runner state.

`RunState.with_pre_loop_setup(cfg)` runs the full pre-loop pipeline and
returns a populated pytree. The private `_Variables` / `_AtmData` /
`_Parameters` containers are scratch inside that constructor only.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax.numpy as jnp
import numpy as np

import chem_funs


class AtmInputs(NamedTuple):
    """Atmosphere structure. nz layers, ni species; (nz-1) arrays are interface-staggered."""
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
    Hpi:        jnp.ndarray   # (nz-1,)  — interface scale height
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
    """Rate constants. `k[0]` is unused — reactions are 1-based throughout VULCAN."""
    k:          jnp.ndarray   # (nr+1, nz) — forward rate constants


class IniAbunOutputs(NamedTuple):
    """Initial-abundance solver output. nz layers, ni species, n_atoms elements."""
    y:           jnp.ndarray         # (nz, ni)    — number density
    ymix:        jnp.ndarray         # (nz, ni)    — mixing ratio
    y_ini:       jnp.ndarray         # (nz, ni)    — initial-state snapshot
    atom_ini:    jnp.ndarray         # (n_atoms,)  — sum of stoich * y per atom
    atom_loss:   jnp.ndarray         # (n_atoms,)  — zeros at init
    atom_conden: jnp.ndarray         # (n_atoms,)  — zeros at init
    charge_list: tuple[str, ...]     # charged species (empty when use_ion=False)


class PhotoInputs(NamedTuple):
    """TOA stellar flux + wavelength extents. `sflux_top` shape (0,) when use_photo=False."""
    sflux_top:   jnp.ndarray
    def_bin_min: float
    def_bin_max: float


class PhotoStaticInputs(NamedTuple):
    """Dense photo cross-section pytree.

    `absp_cross` carries a row for every species in `absp_sp ∪ absp_T_sp` —
    T-dep species also keep their 1D room-T cross.
    `din12_indx` is `-1` until the post-photo `read_sflux` write fires; use
    `with_din12_indx(idx)` to attach it.
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
        if int(idx) < 0:
            raise ValueError(
                f"PhotoStaticInputs.din12_indx must be >= 0; got {idx}"
            )
        return self._replace(din12_indx=int(idx))


class StepInputs(NamedTuple):
    """Time-stepping state. `y_evo`/`t_evo` are zero-length when save_evolution=False."""
    y:        jnp.ndarray   # (nz, ni)  current number density
    y_prev:   jnp.ndarray   # (nz, ni)  last accepted state
    ymix:     jnp.ndarray   # (nz, ni)  mixing ratios
    t:        float         # integration time
    dt:       float         # current step size
    longdy:   float         # long-window dy (Tsai+17 eq 10)
    longdydt: float         # long-window dy/dt (eq 11)
    y_evo:    jnp.ndarray = jnp.zeros((0, 0, 0), dtype=jnp.float64)
    t_evo:    jnp.ndarray = jnp.zeros((0,), dtype=jnp.float64)


class ParamInputs(NamedTuple):
    """Convergence + retry counters that cross the runner boundary."""
    count:             int
    nega_count:        int
    loss_count:        int
    delta_count:       int
    delta:             float
    small_y:           float
    nega_y:            float
    fix_species_start: bool


class AtomInputs(NamedTuple):
    """Atom-conservation diagnostics. `atom_order` is the static ordering for the four arrays."""
    atom_order:     tuple        # static ordering, e.g. ('H', 'O', 'C', 'N', 'S', 'He')
    atom_ini:       jnp.ndarray  # (n_atoms,)
    atom_loss:      jnp.ndarray  # (n_atoms,)
    atom_loss_prev: jnp.ndarray  # (n_atoms,)
    atom_sum:       jnp.ndarray  # (n_atoms,)


class PhotoRuntimeInputs(NamedTuple):
    """Per-step photo state. `None` on RunState when use_photo=False."""
    tau:          jnp.ndarray   # (nz+1, nbin)
    aflux:        jnp.ndarray   # (nz, nbin)
    sflux:        jnp.ndarray   # (nz+1, nbin)
    dflux_d:      jnp.ndarray   # (nz+1, nbin)
    dflux_u:      jnp.ndarray   # (nz+1, nbin)
    prev_aflux:   jnp.ndarray   # (nz, nbin)
    aflux_change: float


class FixSpeciesInputs(NamedTuple):
    """Fixed-species snapshot. `conden_min_lev` is zero when fix_species_from_coldtrap_lev=False."""
    fix_species:    tuple        # static ordering of species names
    fix_y:          jnp.ndarray  # (n_fix_sp, nz)
    conden_min_lev: jnp.ndarray  # (n_fix_sp,) int


class RunMetadata(NamedTuple):
    """Host-side static metadata. Treat as immutable post-construction."""
    Rf:                dict           # 1-based reaction id -> equation string
    n_branch:          dict           # species -> int (#photodissoc branches)
    ion_branch:        dict           # species -> int (#photoionisation branches)
    photo_sp:          frozenset      # species with photodissociation
    ion_sp:            frozenset      # species with photoionisation
    pho_rate_index:    dict           # (sp, br) -> 1-based reaction index
    ion_rate_index:    dict           # (sp, br) -> 1-based reaction index
    ion_br_ratio:      dict           # (sp, br) -> branching ratio
    charge_list:       tuple          # ion species (with non-zero charge)
    conden_re_list:    tuple          # 1-based reaction ids that condense
    start_time:        float          # wall-clock start (for end-of-run print)
    Ti:                jnp.ndarray    # (nz,) initial T profile (atm refresh)
    gas_indx:          tuple          # gas-only species indices
    pref_indx:         int            # reference-layer index (height integ)
    gs:                float          # surface gravity (cm/s^2)
    sat_p:             dict           # species -> (nz,) saturation pressure
    sat_mix:           dict           # species -> (nz,) saturation mixing ratio
    r_p:               dict           # condensate -> particle radius (cm)
    rho_p:             dict           # condensate -> particle density (g/cm^3)
    fix_sp_indx:       dict           # species -> banded-Jacobian indices
    y_ini:             jnp.ndarray    # (nz, ni) initial-state snapshot from ini_y


class RunState(NamedTuple):
    """Umbrella pytree carrying every input the runner reads."""
    atm:           AtmInputs
    rate:          RateInputs
    photo:         PhotoInputs
    # Runner-state slots. Default `None` so legacy callers of
    # `pytree_from_store` (which only fills atm/rate/photo) keep working.
    step:          Optional[StepInputs] = None
    params:        Optional[ParamInputs] = None
    atoms:         Optional[AtomInputs] = None
    photo_runtime: Optional[PhotoRuntimeInputs] = None
    fix_species:   Optional[FixSpeciesInputs] = None
    # Host-side static metadata + photo cross-section pytree.
    # `metadata` defaults to None so legacy `runstate_from_store` callers
    # (which only fill the runner-state slots) keep working; the canonical
    # `with_pre_loop_setup(cfg)` builder populates both `metadata` and
    # `photo_static` (the dense cross-section pytree built once per run).
    metadata:      Optional[RunMetadata] = None
    photo_static:  Optional[PhotoStaticInputs] = None

    # ------------------------------------------------------------------
    # Constructors — canonical pre-loop entry points
    # ------------------------------------------------------------------
    @classmethod
    def with_pre_loop_setup(cls, cfg) -> "RunState":
        """Build a fully-initialised `RunState` from `vulcan_cfg`.

        Runs the same pre-loop pipeline `vulcan_jax.py` ran historically
        (atmosphere structure, rate constants + low-T caps + reverse +
        remove, initial abundance solver, photo cross-sections, photo
        runtime arrays + remove pass), then snapshots the result into
        the typed pytree.

        This is the canonical pre-loop entry point — driver / tests /
        examples / benchmarks all do
            ``rs = RunState.with_pre_loop_setup(cfg); rs = integ(rs)``.
        """
        return _build_pre_loop_runstate(cfg)

    @classmethod
    def fresh_from_cfg(cls, cfg) -> "RunState":
        """Build a `RunState` with zero-valued runtime slots.

        Returns a pytree with `step` / `params` / `atoms` /
        `photo_runtime` / `fix_species` filled with empty / zero
        arrays of the right shapes for `cfg`'s `nz` / `ni` / network.
        Used by tests that need the schema but not the full pipeline.

        The pre-loop slots `atm` / `rate` / `photo` come from the
        actual setup (we still need a real atmosphere structure for
        shapes); the runtime slots are zeroed via the `_fresh_*_inputs`
        helpers.
        """
        rs = _build_pre_loop_runstate(cfg)
        return rs._replace(
            step=_fresh_step_inputs(rs),
            params=_fresh_param_inputs(),
            atoms=_fresh_atom_inputs(rs),
            photo_runtime=_fresh_photo_runtime(rs),
        )


class StellarFlux(NamedTuple):
    """Stellar-flux read result. Empty arrays + zero extents when use_photo=False."""
    wavelength_nm: np.ndarray
    flux:          np.ndarray
    def_bin_min:   float
    def_bin_max:   float


def load_stellar_flux(cfg) -> StellarFlux:
    """Read the stellar flux file and compute the spectral-bin extents.

    Bin range is clamped to [2, 700] nm. Returns an empty payload when
    `cfg.use_photo` is False so callers can call unconditionally.
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
    def_bin_min = float(max(wavelength[0], 2.0))
    def_bin_max = float(min(wavelength[-1], 700.0))
    return StellarFlux(
        wavelength_nm=wavelength,
        flux=flux,
        def_bin_min=def_bin_min,
        def_bin_max=def_bin_max,
    )


def pytree_from_store(var, atm) -> RunState:
    """Snapshot atm + rate + photo fields off a legacy (var, atm) into a typed RunState."""
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
        Hpi=jnp.asarray(getattr(atm, "Hpi", np.zeros_like(np.asarray(atm.dzi)))),
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
    """Reverse adapter: write atm+rate+photo fields back into (var, atm)."""
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
    atm.Hpi        = np.asarray(state.atm.Hpi)
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


def _atom_order_for(cfg) -> tuple:
    return tuple(
        a for a in cfg.atom_list
        if a not in getattr(cfg, "loss_ex", [])
    )


def _atom_dict_to_arr(d, atom_order) -> np.ndarray:
    return np.asarray(
        [float(d.get(a, 0.0)) for a in atom_order],
        dtype=np.float64,
    )


def _atom_arr_to_dict(arr, atom_order) -> dict:
    return {a: float(arr[i]) for i, a in enumerate(atom_order)}


def runstate_from_store(var, atm, para) -> RunState:
    """Snapshot the full runner-input state of `(var, atm, para)`.

    Carries pre-loop fields plus runner-state slots (step/params/atoms/
    photo_runtime/fix_species). Anything the runner needs at entry comes
    from here.
    """
    import vulcan_cfg as _cfg

    base = pytree_from_store(var, atm)

    nz = int(atm.Tco.shape[0])
    ni = int(np.asarray(var.y).shape[1])

    step = StepInputs(
        y=jnp.asarray(var.y, dtype=jnp.float64),
        y_prev=jnp.asarray(
            getattr(var, "y_prev", var.y) if getattr(var, "y_prev", None) is not None else var.y,
            dtype=jnp.float64,
        ),
        ymix=jnp.asarray(var.ymix, dtype=jnp.float64),
        t=float(var.t),
        dt=float(var.dt),
        longdy=float(getattr(var, "longdy", 1.0)),
        longdydt=float(getattr(var, "longdydt", 1.0)),
    )

    params = ParamInputs(
        count=int(getattr(para, "count", 0)),
        nega_count=int(getattr(para, "nega_count", 0)),
        loss_count=int(getattr(para, "loss_count", 0)),
        delta_count=int(getattr(para, "delta_count", 0)),
        delta=float(getattr(para, "delta", 0.0)),
        small_y=float(getattr(para, "small_y", 0.0)),
        nega_y=float(getattr(para, "nega_y", 0.0)),
        fix_species_start=bool(getattr(para, "fix_species_start", False)),
    )

    atom_order = _atom_order_for(_cfg)
    atoms = AtomInputs(
        atom_order=atom_order,
        atom_ini=jnp.asarray(
            _atom_dict_to_arr(getattr(var, "atom_ini", {}), atom_order)
        ),
        atom_loss=jnp.asarray(
            _atom_dict_to_arr(getattr(var, "atom_loss", {}), atom_order)
        ),
        atom_loss_prev=jnp.asarray(
            _atom_dict_to_arr(getattr(var, "atom_loss_prev", {}), atom_order)
        ),
        atom_sum=jnp.asarray(
            _atom_dict_to_arr(getattr(var, "atom_sum", {}), atom_order)
        ),
    )

    if bool(getattr(_cfg, "use_photo", False)) and hasattr(var, "tau"):
        nbin = int(np.asarray(var.tau).shape[1])
        prev_aflux = (
            np.asarray(var.prev_aflux)
            if hasattr(var, "prev_aflux")
            else np.zeros((nz, nbin), dtype=np.float64)
        )
        photo_runtime = PhotoRuntimeInputs(
            tau=jnp.asarray(var.tau, dtype=jnp.float64),
            aflux=jnp.asarray(var.aflux, dtype=jnp.float64),
            sflux=jnp.asarray(var.sflux, dtype=jnp.float64),
            dflux_d=jnp.asarray(var.dflux_d, dtype=jnp.float64),
            dflux_u=jnp.asarray(var.dflux_u, dtype=jnp.float64),
            prev_aflux=jnp.asarray(prev_aflux, dtype=jnp.float64),
            aflux_change=float(getattr(var, "aflux_change", 0.0)),
        )
    else:
        photo_runtime = None

    fix_species_cfg = list(getattr(_cfg, "fix_species", []) or [])
    if fix_species_cfg:
        fix_y_arr = np.zeros((len(fix_species_cfg), nz), dtype=np.float64)
        coldtrap = np.zeros((len(fix_species_cfg),), dtype=np.int32)
        var_fix_y = getattr(var, "fix_y", {}) or {}
        var_coldtrap = getattr(atm, "conden_min_lev", {}) or {}
        for i, sp in enumerate(fix_species_cfg):
            row = var_fix_y.get(sp)
            if row is not None:
                fix_y_arr[i] = np.asarray(row, dtype=np.float64)
            coldtrap[i] = int(var_coldtrap.get(sp, 0))
        fix_species = FixSpeciesInputs(
            fix_species=tuple(fix_species_cfg),
            fix_y=jnp.asarray(fix_y_arr),
            conden_min_lev=jnp.asarray(coldtrap),
        )
    else:
        fix_species = FixSpeciesInputs(
            fix_species=(),
            fix_y=jnp.zeros((0, nz), dtype=jnp.float64),
            conden_min_lev=jnp.zeros((0,), dtype=jnp.int32),
        )
    del ni

    return base._replace(
        step=step,
        params=params,
        atoms=atoms,
        photo_runtime=photo_runtime,
        fix_species=fix_species,
        metadata=_runmetadata_from_legacy(var, atm, para),
    )


def runstate_to_store(state: RunState, var, atm, para) -> None:
    """Reverse adapter: write all RunState slots back to (var, atm, para)."""
    import vulcan_cfg as _cfg

    apply_pytree_to_store(state, var, atm)

    if state.step is not None:
        var.y = np.asarray(state.step.y, dtype=np.float64)
        var.y_prev = np.asarray(state.step.y_prev, dtype=np.float64)
        var.ymix = np.asarray(state.step.ymix, dtype=np.float64)
        var.t = float(state.step.t)
        var.dt = float(state.step.dt)
        var.longdy = float(state.step.longdy)
        var.longdydt = float(state.step.longdydt)

    if state.params is not None:
        para.count = int(state.params.count)
        para.nega_count = int(state.params.nega_count)
        para.loss_count = int(state.params.loss_count)
        para.delta_count = int(state.params.delta_count)
        para.delta = float(state.params.delta)
        para.small_y = float(state.params.small_y)
        para.nega_y = float(state.params.nega_y)
        para.fix_species_start = bool(state.params.fix_species_start)

    if state.atoms is not None:
        a = state.atoms
        var.atom_ini = _atom_arr_to_dict(np.asarray(a.atom_ini), a.atom_order)
        var.atom_loss = _atom_arr_to_dict(np.asarray(a.atom_loss), a.atom_order)
        var.atom_loss_prev = _atom_arr_to_dict(
            np.asarray(a.atom_loss_prev), a.atom_order
        )
        var.atom_sum = _atom_arr_to_dict(np.asarray(a.atom_sum), a.atom_order)

    if state.photo_runtime is not None:
        pr = state.photo_runtime
        var.tau = np.asarray(pr.tau, dtype=np.float64)
        var.aflux = np.asarray(pr.aflux, dtype=np.float64)
        var.sflux = np.asarray(pr.sflux, dtype=np.float64)
        var.dflux_d = np.asarray(pr.dflux_d, dtype=np.float64)
        var.dflux_u = np.asarray(pr.dflux_u, dtype=np.float64)
        var.prev_aflux = np.asarray(pr.prev_aflux, dtype=np.float64)
        var.aflux_change = float(pr.aflux_change)

    if state.fix_species is not None and len(state.fix_species.fix_species) > 0:
        fs = state.fix_species
        fix_y_np = np.asarray(fs.fix_y, dtype=np.float64)
        var.fix_y = {
            sp: fix_y_np[i].copy() for i, sp in enumerate(fs.fix_species)
        }
        if bool(getattr(_cfg, "fix_species_from_coldtrap_lev", False)):
            cm = np.asarray(fs.conden_min_lev, dtype=np.int32)
            for i, sp in enumerate(fs.fix_species):
                if not hasattr(atm, "conden_min_lev") or atm.conden_min_lev is None:
                    atm.conden_min_lev = {}
                atm.conden_min_lev[sp] = int(cm[i])


def _atm_metadata_from_atm(atm) -> dict:
    return dict(
        Ti=jnp.asarray(getattr(atm, "Ti", atm.Tco), dtype=jnp.float64),
        gas_indx=tuple(getattr(atm, "gas_indx", []) or []),
        pref_indx=int(getattr(atm, "pref_indx", 0)),
        gs=float(getattr(atm, "gs", 0.0)),
        sat_p=dict(getattr(atm, "sat_p", {}) or {}),
        sat_mix=dict(getattr(atm, "sat_mix", {}) or {}),
        r_p=dict(getattr(atm, "r_p", {}) or {}),
        rho_p=dict(getattr(atm, "rho_p", {}) or {}),
        fix_sp_indx=dict(getattr(atm, "fix_sp_indx", {}) or {}),
    )


def _runmetadata_from_legacy(var, atm, para) -> RunMetadata:
    """Snapshot host-side static metadata that doesn't fit the JAX pytrees."""
    atm_meta = _atm_metadata_from_atm(atm)
    y_ini_src = getattr(var, "y_ini", None)
    if y_ini_src is None:
        y_ini_src = getattr(var, "y", None)
    if y_ini_src is None:
        nz = int(np.asarray(atm.Tco).shape[0])
        y_ini_arr = jnp.zeros((nz, int(chem_funs.ni)), dtype=jnp.float64)
    else:
        y_ini_arr = jnp.asarray(np.asarray(y_ini_src), dtype=jnp.float64)
    return RunMetadata(
        Rf=dict(getattr(var, "Rf", {}) or {}),
        n_branch=dict(getattr(var, "n_branch", {}) or {}),
        ion_branch=dict(getattr(var, "ion_branch", {}) or {}),
        photo_sp=frozenset(getattr(var, "photo_sp", set()) or set()),
        ion_sp=frozenset(getattr(var, "ion_sp", set()) or set()),
        pho_rate_index=dict(getattr(var, "pho_rate_index", {}) or {}),
        ion_rate_index=dict(getattr(var, "ion_rate_index", {}) or {}),
        ion_br_ratio=dict(getattr(var, "ion_br_ratio", {}) or {}),
        charge_list=tuple(getattr(var, "charge_list", []) or []),
        conden_re_list=tuple(getattr(var, "conden_re_list", []) or []),
        start_time=float(
            getattr(para, "start_time", 0.0) if para is not None else 0.0
        ),
        y_ini=y_ini_arr,
        **atm_meta,
    )


def _build_pre_loop_runstate(cfg) -> RunState:
    """Run the full pre-loop pipeline and return a populated RunState."""
    import time

    import legacy_io as _io
    from atm_setup import Atm
    from ini_abun import InitialAbun
    import op_jax as _op_jax
    import rates as _rates_mod
    import photo_setup as _photo_setup

    stellar = load_stellar_flux(cfg)
    var = _Variables(stellar_flux=stellar)
    atm = _AtmData()
    para = _Parameters()
    para.start_time = time.time()

    make_atm = Atm()
    output = _io.Output()

    atm = make_atm.f_pico(atm)
    atm = make_atm.load_TPK(atm)
    if bool(getattr(cfg, "use_condense", False)):
        make_atm.sp_sat(atm)

    rate = _io.ReadRate()
    var = rate.read_rate(var, atm)
    network = _rates_mod.setup_var_k(cfg, var, atm)
    ini = InitialAbun()
    var = ini.ini_y(var, atm)
    var = ini.ele_sum(var)

    atm = make_atm.f_mu_dz(var, atm, output)
    make_atm.mol_diff(atm)
    make_atm.BC_flux(atm)

    photo_static_pytree = None
    if bool(getattr(cfg, "use_photo", False)):
        _photo_setup.populate_photo_arrays(var, atm)
        make_atm.read_sflux(var, atm)
        photo_static_pytree = _photo_setup._build_photo_static_dense(var, atm)
        photo_static_pytree = photo_static_pytree.with_din12_indx(
            int(var.sflux_din12_indx)
        )
        solver = _op_jax.Ros2JAX()
        solver._photo_static = photo_static_pytree
        solver.compute_tau(var, atm)
        solver.compute_flux(var, atm)
        solver.compute_J(var, atm)
        if bool(getattr(cfg, "use_ion", False)):
            solver.compute_Jion(var, atm)
        _rates_mod.apply_photo_remove(cfg, var, network, atm)

    rs = runstate_from_store(var, atm, para)
    return rs._replace(
        metadata=_runmetadata_from_legacy(var, atm, para),
        photo_static=photo_static_pytree,
    )


def _fresh_step_inputs(rs: RunState) -> StepInputs:
    nz = int(rs.atm.Tco.shape[0])
    ni = int(chem_funs.ni)
    return StepInputs(
        y=jnp.zeros((nz, ni), dtype=jnp.float64),
        y_prev=jnp.zeros((nz, ni), dtype=jnp.float64),
        ymix=jnp.zeros((nz, ni), dtype=jnp.float64),
        t=0.0,
        dt=0.0,
        longdy=1.0,
        longdydt=1.0,
    )


def _fresh_param_inputs() -> ParamInputs:
    return ParamInputs(
        count=0, nega_count=0, loss_count=0, delta_count=0,
        delta=0.0, small_y=0.0, nega_y=0.0, fix_species_start=False,
    )


def _fresh_atom_inputs(rs: RunState) -> AtomInputs:
    if rs.atoms is not None:
        atom_order = rs.atoms.atom_order
    else:
        import vulcan_cfg as _cfg
        atom_order = _atom_order_for(_cfg)
    n = len(atom_order)
    z = jnp.zeros((n,), dtype=jnp.float64)
    return AtomInputs(
        atom_order=atom_order,
        atom_ini=z, atom_loss=z, atom_loss_prev=z, atom_sum=z,
    )


def _fresh_photo_runtime(rs: RunState) -> Optional[PhotoRuntimeInputs]:
    nbin = int(rs.photo.sflux_top.shape[0])
    if nbin == 0:
        return None
    nz = int(rs.atm.Tco.shape[0])
    return PhotoRuntimeInputs(
        tau=jnp.zeros((nz + 1, nbin), dtype=jnp.float64),
        aflux=jnp.zeros((nz, nbin), dtype=jnp.float64),
        sflux=jnp.zeros((nz + 1, nbin), dtype=jnp.float64),
        dflux_d=jnp.zeros((nz + 1, nbin), dtype=jnp.float64),
        dflux_u=jnp.zeros((nz + 1, nbin), dtype=jnp.float64),
        prev_aflux=jnp.zeros((nz, nbin), dtype=jnp.float64),
        aflux_change=0.0,
    )


def legacy_view(rs: RunState):
    """Return a `(var, atm, para)` SimpleNamespace shim built from `rs`.

    Mutations to the shim do NOT round-trip back to `rs` — for that, use
    `runstate_to_store(rs, var, atm, para)` against real legacy containers.
    """
    import types
    md = rs.metadata
    var = types.SimpleNamespace()
    atm = types.SimpleNamespace()
    para = types.SimpleNamespace()

    a = rs.atm
    for f in a._fields:
        setattr(atm, f, np.asarray(getattr(a, f)))
    if md is not None:
        atm.Ti = np.asarray(md.Ti)
        atm.gas_indx = list(md.gas_indx)
        atm.pref_indx = int(md.pref_indx)
        atm.gs = float(md.gs)
        atm.sat_p = dict(md.sat_p)
        atm.sat_mix = dict(md.sat_mix)
        atm.r_p = dict(md.r_p)
        atm.rho_p = dict(md.rho_p)
        atm.fix_sp_indx = dict(md.fix_sp_indx)
    atm.conden_min_lev = {}

    var.k_arr = np.asarray(rs.rate.k, dtype=np.float64)
    var.k = {}
    var.def_bin_min = float(rs.photo.def_bin_min)
    var.def_bin_max = float(rs.photo.def_bin_max)
    import vulcan_cfg as _cfg
    var.var_save = [
        'k', 'y', 'ymix', 'y_ini', 't', 'dt', 'longdy', 'longdydt',
        'atom_ini', 'atom_sum', 'atom_loss', 'atom_conden', 'aflux_change', 'Rf',
    ]
    if bool(getattr(_cfg, "use_photo", False)):
        var.var_save.extend([
            'nbin', 'bins', 'dbin1', 'dbin2', 'tau', 'sflux', 'aflux',
            'cross', 'cross_scat', 'cross_J', 'J_sp', 'n_branch',
        ])
        if getattr(_cfg, "T_cross_sp", []):
            var.var_save.extend(['cross_J', 'cross_T'])
        if bool(getattr(_cfg, "use_ion", False)):
            var.var_save.extend([
                'charge_list', 'ion_sp', 'cross_Jion', 'Jion_sp',
                'ion_wavelen', 'ion_branch', 'ion_br_ratio',
            ])
    var.var_evol_save = ['y_time', 't_time']
    var.y_time = []
    var.t_time = []
    var.dy_time = []
    var.dydt_time = []
    var.atom_loss_time = []
    var.dt_time = []
    var.aflux_change = (
        float(rs.photo_runtime.aflux_change)
        if rs.photo_runtime is not None else 0.0
    )
    if int(rs.photo.sflux_top.shape[0]) > 0:
        var.sflux_top = np.asarray(rs.photo.sflux_top)
    if rs.step is not None:
        var.y = np.asarray(rs.step.y)
        var.y_prev = np.asarray(rs.step.y_prev)
        var.ymix = np.asarray(rs.step.ymix)
        var.t = float(rs.step.t)
        var.dt = float(rs.step.dt)
        var.longdy = float(rs.step.longdy)
        var.longdydt = float(rs.step.longdydt)
    if md is not None:
        var.y_ini = np.asarray(md.y_ini)
    if rs.atoms is not None:
        var.atom_ini = _atom_arr_to_dict(
            np.asarray(rs.atoms.atom_ini), rs.atoms.atom_order
        )
        var.atom_loss = _atom_arr_to_dict(
            np.asarray(rs.atoms.atom_loss), rs.atoms.atom_order
        )
        var.atom_loss_prev = _atom_arr_to_dict(
            np.asarray(rs.atoms.atom_loss_prev), rs.atoms.atom_order
        )
        var.atom_sum = _atom_arr_to_dict(
            np.asarray(rs.atoms.atom_sum), rs.atoms.atom_order
        )
        var.atom_conden = {a: 0.0 for a in rs.atoms.atom_order}
    if rs.photo_runtime is not None:
        var.tau = np.asarray(rs.photo_runtime.tau)
        var.aflux = np.asarray(rs.photo_runtime.aflux)
        var.sflux = np.asarray(rs.photo_runtime.sflux)
        var.dflux_d = np.asarray(rs.photo_runtime.dflux_d)
        var.dflux_u = np.asarray(rs.photo_runtime.dflux_u)
        var.prev_aflux = np.asarray(rs.photo_runtime.prev_aflux)
        var.aflux_change = float(rs.photo_runtime.aflux_change)
    if rs.photo_static is not None:
        var.nbin = int(rs.photo_static.nbin)
        var.bins = np.asarray(rs.photo_static.bins)
        var.dbin1 = float(rs.photo_static.dbin1)
        var.dbin2 = float(rs.photo_static.dbin2)
        var.J_sp = {}
        if int(getattr(rs.photo_static, "cross_Jion", np.zeros(0)).shape[0]) > 0:
            var.Jion_sp = {}
            var.ion_wavelen = {}
    if md is not None:
        var.Rf = dict(md.Rf)
        var.n_branch = dict(md.n_branch)
        var.ion_branch = dict(md.ion_branch)
        var.photo_sp = set(md.photo_sp)
        var.ion_sp = set(md.ion_sp)
        var.pho_rate_index = dict(md.pho_rate_index)
        var.ion_rate_index = dict(md.ion_rate_index)
        var.ion_br_ratio = dict(md.ion_br_ratio)
        var.charge_list = list(md.charge_list)
        var.conden_re_list = list(md.conden_re_list)

    if rs.params is not None:
        para.count = int(rs.params.count)
        para.nega_count = int(rs.params.nega_count)
        para.loss_count = int(rs.params.loss_count)
        para.delta_count = int(rs.params.delta_count)
        para.delta = float(rs.params.delta)
        para.small_y = float(rs.params.small_y)
        para.nega_y = float(rs.params.nega_y)
        para.fix_species_start = bool(rs.params.fix_species_start)
    if md is not None:
        para.start_time = float(md.start_time)

    return var, atm, para


class _Variables(object):
    """Private mutable scratch container used by the pre-loop pipeline."""
    def __init__(self, stellar_flux=None):
        from chem_funs import ni as _ni, spec_list as _spec_list  # noqa: F401
        from vulcan_cfg import nz as _nz
        import vulcan_cfg as _vcfg

        self.k = {}
        self.k_arr = None
        self.y = np.zeros((_nz, _ni))
        self.y_prev = np.zeros((_nz, _ni))
        self.ymix = np.zeros((_nz, _ni))
        self.y_ini = np.zeros((_nz, _ni))
        self.t = 0
        self.dt = _vcfg.dttry
        self.dy = 1.
        self.dy_prev = 1.
        self.dydt = 1.
        self.longdy = 1.
        self.longdydt = 1.

        self.dy_time = []
        self.dydt_time = []
        self.atim_loss_time = []
        self.ymix_time = []
        self.y_time = []
        self.t_time = []
        self.dt_time = []
        self.atom_loss_time = []

        self.atom_ini = {}
        self.atom_sum = {}
        self.atom_loss = {}
        self.atom_loss_prev = {}
        self.atom_conden = {}

        self.Rf = {}
        self.Rindx = {}
        self.a, self.n, self.E, self.a_inf, self.n_inf, self.E_inf, = [
            {} for _ in range(6)
        ]
        self.k_fun, self.k_inf = [{} for _ in range(2)]
        self.kinf_fun = {}
        self.k_fun_new = {}
        self.photo_sp = set()
        self.pho_rate_index, self.n_branch, self.wavelen = {}, {}, {}
        self.ion_rate_index, self.ion_branch, self.ion_wavelen, self.ion_br_ratio = (
            {}, {}, {}, {}
        )
        self.charge_list, self.ion_sp = [], set()

        self.aflux_change = 0.

        if stellar_flux is None:
            stellar_flux = load_stellar_flux(_vcfg)
        self.def_bin_min = stellar_flux.def_bin_min
        self.def_bin_max = stellar_flux.def_bin_max

        self.var_save = ['k', 'y', 'ymix', 'y_ini', 't', 'dt', 'longdy', 'longdydt',
                         'atom_ini', 'atom_sum', 'atom_loss', 'atom_conden',
                         'aflux_change', 'Rf']
        if _vcfg.use_photo:
            self.var_save.extend(['nbin', 'bins', 'dbin1', 'dbin2', 'tau', 'sflux',
                                  'aflux', 'cross', 'cross_scat', 'cross_J',
                                  'J_sp', 'n_branch'])
            if _vcfg.T_cross_sp:
                self.var_save.extend(['cross_J', 'cross_T'])
            if _vcfg.use_ion:
                self.var_save.extend(['charge_list', 'ion_sp', 'cross_Jion',
                                      'Jion_sp', 'ion_wavelen', 'ion_branch',
                                      'ion_br_ratio'])
        self.var_evol_save = ['y_time', 't_time']
        self.conden_re_list = []

        self.v_ratio = np.ones(_nz)


class _AtmData(object):
    """Private mutable scratch container for atmosphere-structure arrays."""
    def __init__(self):
        from chem_funs import ni as _ni, spec_list as _spec_list
        from vulcan_cfg import nz as _nz
        import vulcan_cfg as _vcfg

        self.pco = np.logspace(np.log10(_vcfg.P_b), np.log10(_vcfg.P_t), _nz)
        self.pico = np.empty(_nz + 1)
        self.dz = np.zeros(_nz)
        self.dzi = np.zeros(_nz - 1)
        self.zco = np.zeros(_nz + 1)
        self.zmco = np.empty(_nz)
        self.Tco = np.empty(_nz)
        self.Kzz = np.zeros(_nz - 1)
        self.vz = np.zeros(_nz - 1)
        self.M = np.empty(_nz)
        self.n_0 = np.empty(_nz)
        self.Hp = np.empty(_nz)
        self.mu = np.empty(_nz)
        self.ms = np.empty(_ni)
        self.Dzz = np.zeros((_nz - 1, _ni))
        self.Dzz_cen = np.zeros((_nz, _ni))
        self.vm = np.zeros((_nz, _ni))
        self.vs = np.zeros((_nz - 1, _ni))
        self.alpha = np.zeros(_ni)
        self.gs = _vcfg.gs
        self.g = np.zeros(_nz)

        self.top_flux = np.zeros(_ni)
        self.bot_flux = np.zeros(_ni)
        self.bot_vdep = np.zeros(_ni)
        self.bot_fix_sp = np.zeros(_ni)

        self.sat_p = {}
        self.sat_mix = {}
        self.conden_min_lev = {}

        self.gas_indx = [
            _ for _ in range(_ni) if _spec_list[_] not in _vcfg.non_gas_sp
        ]

        self.fix_sp_indx = {}
        if hasattr(_vcfg, "fix_species"):
            for sp in _vcfg.fix_species:
                self.fix_sp_indx[sp] = np.arange(
                    _spec_list.index(sp),
                    _spec_list.index(sp) + _ni * _nz,
                    _ni,
                )

        if _vcfg.use_ion:
            self.fix_e_indx = np.arange(
                _spec_list.index('e'),
                _spec_list.index('e') + _ni * _nz,
                _ni,
            )

        self.r_p, self.rho_p = {}, {}
        if _vcfg.use_condense:
            for sp in _vcfg.r_p.keys():
                self.r_p[sp] = _vcfg.r_p[sp]
            for sp in _vcfg.rho_p.keys():
                self.rho_p[sp] = _vcfg.rho_p[sp]

            self.conden_status = np.zeros(_nz, dtype=bool)


class _Parameters(object):
    """Private mutable scratch container for numerical-method counters and flags."""
    def __init__(self):
        from vulcan_cfg import nz as _nz
        from chem_funs import ni as _ni

        self.nega_y = 0
        self.small_y = 0
        self.delta = 0
        self.count = 0
        self.nega_count = 0
        self.loss_count = 0
        self.delta_count = 0
        self.end_case = 0
        self.solver_str = ''
        self.switch_final_photo_frq = False
        self.where_varies_most = np.zeros((_nz, _ni))
        self.fix_species_start = False
