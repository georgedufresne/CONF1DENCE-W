#!/usr/bin/env python
"""Make a W isochrone -- the exact recipe behind the cluster panels, self-contained.

An isochrone is "all stars of one age". Building a good-looking one from an EEP-conditioned emulator is TWO steps, and
the quality comes from doing each in its well-conditioned direction:

  STEP 1  BACKBONE (age inversion).  For a target age, find at each EEP the initial MASS whose age equals it.
          Do it at FIXED EEP, where age falls monotonically with mass -- the well-conditioned direction. Solving the
          other way ("fix mass, find EEP") is nearly flat near the RGB tip and zig-zags. We bracket a sign change and
          BISECT, and keep only roots where age DECREASES through the target (real stars; ascending roots are spurious
          extrapolation), with the exact physical constraint that mass is strictly increasing with EEP along an
          isochrone. This is what stops the curve from spiking.  -> arrays (mass, eep)

  STEP 2  POSITION (the flow's MODE, not a sample).  W is a conditional normalising flow: for a context it is a whole
          probability density over (logTeff, logg, logL, logR, cHe, Mhe), not a point. We take the MODE -- push the
          centre of the base Gaussian (z = 0) back through the flow's inverse. It is deterministic, so there is NO
          sampling scatter; that is why the curve is smooth. (Drawing random samples and averaging is noisier and
          slightly biased.)  -> logTeff, logL at every backbone point

EEP conditioning is why none of this blows up: every (physics, EEP, mass) triple is a valid point by construction, so
there is no "dead star" extrapolation the way an age-conditioned emulator has past the tip.

  python isochrone_W.py --age 12.5 --plot iso.png                 # default physics, 12.5 Gyr
  python isochrone_W.py --age 12.5 --set FeH=-2.27 --out iso.npz  # metal-poor (M15-like)

  from isochrone_W import w_isochrone, WEmulator, AgeHead
  W, ah = WEmulator(), AgeHead()
  iso = w_isochrone(W, ah, W.fmt["label_default"], age_gyr=12.5)   # dict: eep, mass, log_Teff, log_L, ...
"""
from __future__ import annotations
import argparse, numpy as np, torch
from predict_W import WEmulator, AgeHead, _inv_col

EEP_LO, EEP_HI = 202.0, 605.0        # trained EEP range (202 = ZAMS ... 491 = RGB base ... 605 = RGB tip)


@torch.no_grad()
def central_mode(W, eep, mass, phys):
    """W's MODE (z = 0 through the inverse flow) at each (eep, mass); deterministic -> smooth. Returns a dict of the 6
    outputs in physical units."""
    ctx = W._context(phys, np.asarray(eep, float), np.asarray(mass, float))      # normalised, already sliced to label_take
    z = W.flow(ctx).transform.inv(torch.zeros(len(ctx), W.cfg["n_features"], device=W.dev)).cpu().numpy()
    out = {}
    for j, (nm, sc, sp) in enumerate(zip(W.dh, W.fmt["data_scaler"], W.fmt["data_scale"])):
        out[nm] = _inv_col(np.clip(z[:, j], -0.5, 1.5), sc, sp)
    return out


def age_inversion(ah, phys22, age_gyr, mlo=0.10, mhi=2.5, nm=1200, neep=280, nbis=44):
    """At each EEP solve age(EEP, mass) = age_gyr for the mass. Returns (mass, eep), mass strictly increasing with EEP.
    Faithful port of the production routine (bracket a DESCENDING root, bisect, enforce monotonic mass)."""
    phys22 = np.asarray(phys22, float)[:len(ah.theta)]
    tgt = np.log10(age_gyr * 1e9)
    eg = np.linspace(EEP_LO, EEP_HI, neep); mg = np.linspace(mlo, mhi, nm)
    E = np.repeat(eg, nm); M = np.tile(mg, neep)
    F = ah.log_age(E, M, phys22).reshape(neep, nm) - tgt          # + = too old, - = too young
    desc = (F[:, :-1] > 0) & (F[:, 1:] <= 0)                      # age crosses the target going + -> - (a real star)
    sel = np.full(neep, -1, int); cap = nm - 2                    # walk EEP down from the tip; mass must not increase
    for r in range(neep - 1, -1, -1):
        c = np.nonzero(desc[r])[0]; c = c[c <= cap]
        if len(c) == 0: continue
        sel[r] = c[-1]; cap = c[-1]
    rows = np.nonzero(sel >= 0)[0]
    if len(rows) == 0:
        return np.array([]), np.array([])
    j = sel[rows]; e = eg[rows]; lo = mg[j].copy(); hi = mg[j + 1].copy(); flo = F[rows, j]
    for _ in range(nbis):                                         # ~44 halvings -> ~1e-13 in mass
        mid = 0.5 * (lo + hi); fm = ah.log_age(e, mid, phys22) - tgt
        same = np.sign(fm) == np.sign(flo)
        lo = np.where(same, mid, lo); hi = np.where(same, hi, mid); flo = np.where(same, fm, flo)
    m = np.maximum.accumulate(0.5 * (lo + hi))                   # enforce the known monotonicity
    ok = (m > mlo + 1e-6) & (m < mhi - 1e-6)
    return m[ok], e[ok]


def w_isochrone(W, ah, phys, age_gyr, **kw):
    """The full isochrone: backbone (age inversion) + W's mode at each point. phys = 24-value label vector
    (label_header order); only the 22 physics entries are used, EEP/mass are filled per point."""
    mass, eep = age_inversion(ah, phys[:22], age_gyr, **kw)
    iso = {"age_Gyr": float(age_gyr), "eep": eep, "mass": mass}
    if len(eep):
        iso.update(central_mode(W, eep, mass, phys))
    return iso


def plot_isochrone(iso, path, label=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(iso["log_Teff"], iso["log_L"], "-", lw=1.8, label=label or f"W, {iso['age_Gyr']:.1f} Gyr")
    for mv in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):                    # mass ticks (pile up on the RGB -- mass degeneracy)
        if iso["mass"].min() <= mv <= iso["mass"].max():
            o = np.argsort(iso["mass"])
            x = np.interp(mv, iso["mass"][o], iso["log_Teff"][o]); y = np.interp(mv, iso["mass"][o], iso["log_L"][o])
            ax.scatter([x], [y], s=22, facecolor="white", edgecolor="C0", lw=1.3, zorder=5)
            ax.annotate(f"{mv:.1f}", (x, y), textcoords="offset points", xytext=(5, -9), fontsize=7)
    ax.invert_xaxis()                                            # HR convention: hot (high logTeff) on the LEFT
    ax.set_xlabel(r"$\log T_{\rm eff}$"); ax.set_ylabel(r"$\log L / L_\odot$"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--age", type=float, required=True, help="isochrone age in Gyr")
    ap.add_argument("--set", nargs="*", default=[], help="physics overrides NAME=VALUE (names from the format label_header)")
    ap.add_argument("--neep", type=int, default=280); ap.add_argument("--nmass", type=int, default=1200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write the isochrone to an .npz")
    ap.add_argument("--plot", default=None, help="write a PNG of the isochrone")
    a = ap.parse_args()
    W = WEmulator(device=a.device); ah = AgeHead(device=a.device)
    phys = list(W.fmt.get("label_default", [0.0] * len(W.lh)))
    for kv in a.set:
        nm, val = kv.split("="); phys[W.lh.index(nm.strip())] = float(val)
    iso = w_isochrone(W, ah, phys, a.age, neep=a.neep, nm=a.nmass)
    if not len(iso["eep"]):
        print("# no isochrone: the age head found no valid mass at this age/physics"); return
    print(f"# W isochrone, {a.age:.2f} Gyr, [Fe/H]={phys[0]:+.3f};  {len(iso['eep'])} points (EEP {iso['eep'][0]:.0f}..{iso['eep'][-1]:.0f})")
    print("# EEP  mass  log_Teff  log_L  log_g  log_R")
    for i in range(0, len(iso["eep"]), max(1, len(iso["eep"]) // 25)):
        print(f"  {iso['eep'][i]:5.0f}  {iso['mass'][i]:.4f}  {iso['log_Teff'][i]:.4f}  {iso['log_L'][i]:+.4f}  {iso['log_g'][i]:+.3f}  {iso['log_R'][i]:+.3f}")
    if a.out:
        np.savez(a.out, **iso); print(f"# wrote {a.out}")
    if a.plot:
        plot_isochrone(iso, a.plot); print(f"# wrote {a.plot}")


if __name__ == "__main__":
    main()
