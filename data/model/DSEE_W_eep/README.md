# W emulator

**W** = the EEP-conditioned DSEE emulator, a conditional neural spline flow (zuko NSF).

| | |
|---|---|
| architecture | zuko NSF, 10 transforms, hidden `[256] x 10` (~6.4 M params) |
| inputs | 22 physics parameters + EEP + initial mass; 6 dead physics columns dropped -> 18 context values |
| outputs | `log_Teff, log_g, log_L, log_R, center_he4, he_core_mass` |
| EEP tags | pyeep-1.2 |
| recipe | sensitivity-weighted EEP sampling, batch 16384, **cosine learning rate** 1e-4 -> 1e-6 |
| checkpoint | `best_rmse.pt`, step 1,642,400, chosen by lowest per-phase RMSE on held-out validation shards |
| format | `../../format/DSEE_mcdsep_eep_format.json` (normalisation; required) |

## Why W

Against the original age-conditioned emulator (OG), W is comparable on the main sequence and far better past the
turnoff — on held-out stars, ~16–20 % lower median error in log Teff / log L overall and ~30–80 % lower on the
subgiant and giant branches. The cosine learning rate was the change that put it ahead of the constant-LR baseline.

## Files

- `best_rmse.pt` — weights + `config` (architecture, `label_take` = the 18 context columns used). Load with
  `../../../predict_W.py` or `dsee1/iso_panels.load`.
- `age_head_v4.pt` — optional small MLP for stellar age (physics + EEP + mass -> log age). W predicts position along a
  track, not age; use this head when an age is needed.
- `config.json` — machine-readable summary.

## Caveat

The age head (v4) is the weakest piece and is worth retraining; W's structural predictions (Teff, L, g, R, core) are
the strong part.

## Age head calling convention (age_head_v4.pt)

W predicts a star's POSITION along a track; `age_head_v4.pt` gives the matching age. It is an MLP (`layers`=5 x
`hidden`=512, SiLU) with a strict input/output layout (all constants are IN the checkpoint):

    input x (length 24), in THIS order:
      [ (theta[i] - mu[i]) / sd[i]   for the 22 physics params, in label_header[:22] order ]   <- physics FIRST,
      + [ (mass - 0.1) / 3.9 ]                                                                   standardised by the
      + [ (EEP  - 202.0) / 403.0 ]                                                               head's OWN mu/sd
    output = net(x) * ys + ym   =   log10(age / years)                                           then mass, then EEP

Checkpoint keys: `state_dict, theta` (the 22 names, == label_header[:22]), `mu, sd` (per-physics standardisation),
`ym`(=8.8938), `ys`(=0.9241), `hidden, layers`. Use `predict_W.AgeHead`:

    from predict_W import WEmulator, AgeHead
    ah = AgeHead()
    ah.age_gyr(eep=[454, 605], mass=1.0, phys22=WEmulator().fmt["label_default"][:22])   # -> ~[10.6, 12.0] Gyr

or `python predict_W.py --eep 202:605:1 --mass 1.0 --age`.

Common mistake (the ~10x-too-young symptom): feeding raw or format-normalised physics instead of `(theta-mu)/sd`. The
net then sees ~its training mean and returns ~`10**ym` = 0.78 Gyr for every star -- monotonic but ~10x low. Also note
EEP and mass go at the END, not the front. To build a fixed-age isochrone you invert age(EEP, mass) for the mass at
each EEP (age falls monotonically with mass at fixed EEP -- the well-conditioned direction).

Caveat: the v4 head is the weakest piece of the pipeline and is worth retraining; W's structural outputs are the strong
part.
