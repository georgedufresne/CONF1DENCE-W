#!/usr/bin/env python
"""Stand-alone loader and predictor for the W emulator (EEP-conditioned DSEE flow).

W is a conditional neural spline flow (zuko NSF, 10 transforms x [256]*10) trained on the pyeep-1.2 EEP tags. It maps
  22 stellar physics parameters + EEP + initial mass  ->  log_Teff, log_g, log_L, log_R, central He4, He-core mass
and was the best per-point emulator in our tests (see model/DSEE_W_eep/README.md).

This file depends only on torch, zuko, numpy (NOT on the CONF1DENCE package internals), so it runs anywhere those three
are installed. The normalisation below is copied verbatim from CONF1DENCE.utils.helper.data_transform, and the model is
built exactly as dsee1/iso_panels.py builds it (including the --label-drop column selection W trained with).

  # central prediction along a mass's track at a grid of EEPs, default physics:
  python predict_W.py --eep 202:605:1 --mass 0.85
  # a whole mass grid:
  python predict_W.py --eep 300,400,491,550 --mass 0.6,0.8,1.0 --out pred.npz
  # your own physics (named overrides; everything else from the format's label_default):
  python predict_W.py --eep 400 --mass 1.0 --set FeH=-1.0 "Mixing Length=1.9"

As a library:
  from predict_W import WEmulator
  W = WEmulator()                                  # loads model/DSEE_W_eep/best_rmse.pt + the format
  out = W.predict(eep=[400,491,550], mass=[0.8,0.8,0.8])     # dict: log_Teff, log_L, ...  (central/medoid values)
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch, zuko

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_CKPT = os.path.join(HERE, "data", "model", "DSEE_W_eep", "best_rmse.pt")
DEF_FMT = os.path.join(HERE, "data", "format", "DSEE_mcdsep_eep_format.json")


# --- normalisation (verbatim from CONF1DENCE.utils.helper.data_transform; only the scalers this format uses) --------
def _norm_col(col, scaler, p):
    if scaler in ("None", None):        return col
    if scaler == "linear_rescale":      return (col - p[0]) / (p[1] - p[0])
    if scaler == "normal":              return (col - p[0]) / p[1]
    if scaler == "log_rescale":         return (np.log10(np.clip(col, 1e-10, None)) - p[0]) / (p[1] - p[0])
    raise ValueError(f"input scaler {scaler!r} not supported by this stand-alone file")


def _inv_col(col, scaler, p):
    if scaler == "linear_rescale":      return col * (p[1] - p[0]) + p[0]
    if scaler == "log_rescale":         return np.power(10.0, col * (p[1] - p[0]) + p[0])
    if scaler == "normal":              return col * p[1] + p[0]
    if scaler in ("None", None):        return col
    raise ValueError(f"output scaler {scaler!r} not supported by this stand-alone file")


class WEmulator:
    def __init__(self, ckpt=DEF_CKPT, fmt=DEF_FMT, device="cpu"):
        self.dev = torch.device(device)
        self.fmt = json.load(open(fmt))
        ck = torch.load(ckpt, map_location=self.dev, weights_only=False)
        c = ck["config"]
        assert c.get("arch", "nsf") == "nsf", "this loader is for the NSF emulator (W); fm models load differently"
        self.cfg = c
        flow = zuko.flows.NSF(features=c["n_features"], context=c["n_context"],
                              transforms=c["transforms"], hidden_features=[c["hidden"]] * c["hidden_layers"])
        flow.load_state_dict({k[5:] if k.startswith("flow.") else k: v for k, v in ck["model"].items()})
        flow.to(self.dev).eval()
        self.flow = flow
        self.take = c.get("label_take")                 # the 18 of 24 context columns W trained on (--label-drop)
        self.step = ck.get("global_step", 0)
        self.lh = self.fmt["label_header"]; self.dh = self.fmt["data_header"]

    def _context(self, phys, eeps, masses):
        """Full 24-column normalised context (then sliced to label_take), exactly as iso_panels.encode()."""
        eeps = np.asarray(eeps, float); masses = np.asarray(masses, float); n = len(eeps)
        cols = []
        for k, (nm, sc, sp) in enumerate(zip(self.lh, self.fmt["label_scaler"], self.fmt["label_scale"])):
            if nm in ("Age", "EEP"):   v = eeps
            elif nm == "mass":         v = masses
            else:                      v = np.full(n, phys[k], float)
            cols.append(_norm_col(v, sc, sp)[:, None])
        ctx = np.hstack(cols).astype(np.float32)
        if self.take:
            ctx = ctx[:, self.take]
        return torch.from_numpy(ctx).to(self.dev)

    @torch.no_grad()
    def predict(self, eep, mass, phys=None, n_samples=101, raw=False):
        """Central (medoid) prediction at each (eep, mass). phys: 24-vector in label_header order (physics columns used;
        EEP/mass positions ignored). Default = the format's label_default. Returns {output_name: array}."""
        eep = np.atleast_1d(eep); mass = np.atleast_1d(mass)
        if mass.size == 1 and eep.size > 1: mass = np.full(eep.shape, mass[0])
        if phys is None: phys = list(self.fmt.get("label_default", [0.0] * len(self.lh)))
        ctx = self._context(phys, eep, mass)
        s = self.flow(ctx).sample((n_samples,)).permute(1, 0, 2)          # (N, n_samples, 6) normalised
        if raw:
            samp = s.cpu().numpy()
            out = {}
            for j, (nm, sc, sp) in enumerate(zip(self.dh, self.fmt["data_scaler"], self.fmt["data_scale"])):
                out[nm] = _inv_col(samp[:, :, j].reshape(-1), sc, sp).reshape(samp.shape[0], samp.shape[1])
            return out
        pick = torch.cdist(s, s).sum(-1).argmin(-1)                       # medoid index per point
        med = s[torch.arange(s.shape[0]), pick].cpu().numpy()            # (N, 6) normalised central value
        out = {}
        for j, (nm, sc, sp) in enumerate(zip(self.dh, self.fmt["data_scaler"], self.fmt["data_scale"])):
            out[nm] = _inv_col(med[:, j], sc, sp)
        out["EEP"] = eep.astype(float); out["mass"] = mass.astype(float)
        return out


def _parse_grid(s):
    if ":" in s:
        a, b, st = (s.split(":") + ["1"])[:3]; return np.arange(float(a), float(b) + 1e-9, float(st))
    return np.array([float(x) for x in s.split(",")])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DEF_CKPT); ap.add_argument("--fmt", default=DEF_FMT)
    ap.add_argument("--eep", required=True, help="EEP grid: 'a:b:step' or comma list (202..605 is the trained range)")
    ap.add_argument("--mass", required=True, help="mass(es) in Msun: one value, or a comma list matching --eep")
    ap.add_argument("--set", nargs="*", default=[], help="physics overrides NAME=VALUE (names from the format label_header)")
    ap.add_argument("--n-samples", type=int, default=101); ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write an .npz of the predictions")
    a = ap.parse_args()
    W = WEmulator(a.ckpt, a.fmt, a.device)
    phys = list(W.fmt.get("label_default", [0.0] * len(W.lh)))
    for kv in a.set:
        nm, val = kv.split("="); phys[W.lh.index(nm.strip())] = float(val)
    eeps = _parse_grid(a.eep); masses = _parse_grid(a.mass)
    if masses.size == 1: masses = np.full(eeps.shape, masses[0])
    if eeps.size == 1 and masses.size > 1: eeps = np.full(masses.shape, eeps[0])
    out = W.predict(eeps, masses, phys=phys, n_samples=a.n_samples)
    print(f"# W emulator (step {W.step:,}); {len(eeps)} points; central (medoid of {a.n_samples}) predictions")
    print("# " + "  ".join(["EEP", "mass"] + W.dh))
    for i in range(len(eeps)):
        print("  ".join(f"{out[k][i]:+.4f}" for k in ["EEP", "mass"] + W.dh))
    if a.out:
        np.savez(a.out, **out); print(f"# wrote {a.out}")


if __name__ == "__main__":
    main()
