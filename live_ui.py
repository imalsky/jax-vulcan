"""Host-side live mixing-ratio + flux plotter, fired between JIT'd step chunks."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import vulcan_cfg

_TEX_LABELS = {
    "H": "H", "H2": "H$_2$", "O": "O", "OH": "OH", "H2O": "H$_2$O",
    "CH": "CH", "C": "C", "CH2": "CH$_2$", "CH3": "CH$_3$", "CH4": "CH$_4$",
    "HCO": "HCO", "H2CO": "H$_2$CO", "C4H2": "C$_4$H$_2$",
    "C2": "C$_2$", "C2H2": "C$_2$H$_2$", "C2H3": "C$_2$H$_3$", "C2H": "C$_2$H",
    "CO": "CO", "CO2": "CO$_2$", "He": "He", "O2": "O$_2$",
    "CH3OH": "CH$_3$OH", "C2H4": "C$_2$H$_4$", "C2H5": "C$_2$H$_5$",
    "C2H6": "C$_2$H$_6$", "CH3O": "CH$_3$O", "CH2OH": "CH$_2$OH", "NH3": "NH$_3$",
}

_TABLEAU20 = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
    (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
    (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
]
_TABLEAU20 = [(r / 255.0, g / 255.0, b / 255.0) for (r, g, b) in _TABLEAU20]


def any_live_flag_on(cfg=vulcan_cfg) -> bool:
    return any(
        bool(getattr(cfg, name, False))
        for name in ("use_live_plot", "use_live_flux",
                     "use_save_movie", "use_flux_movie")
    )


class LiveUI:
    def __init__(self) -> None:
        self.pic_count = 0
        self.flux_pic_count = 0
        self._species_index: dict[str, int] | None = None
        self._plt = None

    def _ensure_mpl(self):
        if self._plt is None:
            import matplotlib
            if (not os.environ.get("DISPLAY")
                    and not os.environ.get("MPLBACKEND")
                    and "darwin" not in os.uname().sysname.lower()):
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            self._plt = plt
        return self._plt

    def _ensure_species_index(self):
        if self._species_index is None:
            import chem_funs
            self._species_index = {sp: i for i, sp in enumerate(chem_funs.spec_list)}
        return self._species_index

    def dispatch(self, var, atm, para) -> None:
        live_plot = bool(getattr(vulcan_cfg, "use_live_plot", False))
        live_flux = bool(getattr(vulcan_cfg, "use_live_flux", False))
        save_movie = bool(getattr(vulcan_cfg, "use_save_movie", False))
        flux_movie = bool(getattr(vulcan_cfg, "use_flux_movie", False))
        if not (live_plot or live_flux or save_movie or flux_movie):
            return

        if live_plot or save_movie:
            self.update_mix(var, atm, para,
                            save_movie=save_movie, show=live_plot)

        if (live_flux or flux_movie) and bool(getattr(vulcan_cfg, "use_photo", False)):
            self.update_flux(var, atm, para,
                             save_movie=flux_movie, show=live_flux)

    def update_mix(self, var, atm, para,
                   save_movie: bool = False, show: bool = True) -> None:
        plt = self._ensure_mpl()
        species_idx = self._ensure_species_index()

        plt.figure("live mixing ratios")
        if show:
            plt.ion()

        palette = list(_TABLEAU20)
        plot_spec = list(getattr(vulcan_cfg, "plot_spec", []))
        for color_index, sp in enumerate(plot_spec):
            if sp not in species_idx:
                continue
            sp_lab = _TEX_LABELS.get(sp, sp)
            if color_index >= len(palette):
                palette.append(tuple(np.random.rand(3)))
            color = palette[color_index]
            sp_i = species_idx[sp]

            if not getattr(vulcan_cfg, "plot_height", False):
                plt.plot(var.ymix[:, sp_i], atm.pco / 1.0e6,
                         color=color, label=sp_lab)
                if (getattr(vulcan_cfg, "use_condense", False)
                        and sp in getattr(vulcan_cfg, "condense_sp", [])
                        and hasattr(atm, "sat_mix")
                        and sp in atm.sat_mix):
                    plt.plot(atm.sat_mix[sp], atm.pco / 1.0e6,
                             color=color, label=f"{sp_lab} sat", ls="--")
                plt.gca().set_yscale("log")
                if not getattr(self, "_mix_axis_inverted", False):
                    plt.gca().invert_yaxis()
                    self._mix_axis_inverted = True
                plt.ylabel("Pressure (bar)")
                plt.ylim((vulcan_cfg.P_b / 1.0e6, vulcan_cfg.P_t / 1.0e6))
            else:
                plt.plot(var.ymix[:, sp_i], atm.zmco / 1.0e5,
                         color=color, label=sp_lab)
                if (getattr(vulcan_cfg, "use_condense", False)
                        and sp in getattr(vulcan_cfg, "condense_sp", [])
                        and hasattr(atm, "sat_mix")
                        and sp in atm.sat_mix):
                    plt.plot(atm.sat_mix[sp], atm.zco[1:] / 1.0e5,
                             color=color, label=f"{sp_lab} sat", ls="--")
                plt.ylim((atm.zco[0] / 1e5, atm.zco[-1] / 1e5))
                plt.ylabel("Height (km)")

        plt.title(f"{int(para.count)} steps and {float(var.t):.2e} s")
        plt.gca().set_xscale("log")
        plt.xlim(1.0e-20, 1.0)
        plt.legend(frameon=0, prop={"size": 14}, loc=3)
        plt.xlabel("Mixing Ratios")

        if show:
            plt.show(block=False)
            plt.pause(0.001)

        if save_movie:
            movie_dir = Path(getattr(vulcan_cfg, "movie_dir", "plot/movie/"))
            movie_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(movie_dir / f"{self.pic_count}.png"), dpi=200)
            self.pic_count += 1

        plt.clf()
        self._mix_axis_inverted = False

    def update_flux(self, var, atm, para,
                    save_movie: bool = False, show: bool = True) -> None:
        plt = self._ensure_mpl()

        plt.figure("live flux")
        if show:
            plt.ion()

        plt.plot(np.sum(var.dflux_u, axis=1), atm.pico / 1.0e6, label="up flux")
        plt.plot(np.sum(var.dflux_d, axis=1), atm.pico / 1.0e6,
                 label="down flux", ls="--", lw=1.2)
        plt.plot(np.sum(var.sflux, axis=1), atm.pico / 1.0e6,
                 label="stellar flux", ls=":", lw=1.5)

        plt.title(f"{int(para.count)} steps and {float(var.t):.2e} s")
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.gca().invert_yaxis()
        plt.xlim(xmin=1.0e-8)
        plt.ylim((atm.pico[0] / 1.0e6, atm.pico[-1] / 1.0e6))
        plt.legend(frameon=0, prop={"size": 14}, loc=3)
        plt.xlabel("Diffusive flux")
        plt.ylabel("Pressure (bar)")

        if show:
            plt.show(block=False)
            plt.pause(0.1)

        if save_movie:
            flux_dir = Path("plot/movie")
            flux_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(flux_dir / f"flux-{int(para.count)}.jpg"))
            self.flux_pic_count += 1

        plt.clf()
