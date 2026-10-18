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
