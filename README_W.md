# CONF1DENCE + W emulator

A self-contained copy of the CONF1DENCE package **with the W emulator added**, meant to be dropped onto another
machine. It does **not** modify the original CONF1DENCE working copy — this is a separate bundle.

W is the EEP-conditioned DSEE emulator (a conditional neural spline flow) that was the best per-point model in our
tests: it maps 22 stellar physics parameters + an EEP (equivalent evolutionary point) + the initial mass to
`log_Teff, log_g, log_L, log_R, central He4, He-core mass`.

## What's here

```
predict_W.py                         stand-alone loader + predictor (only needs torch, zuko, numpy)
data/model/DSEE_W_eep/
    best_rmse.pt                     the W weights (zuko NSF, 10 transforms x [256]*10)
    age_head_v4.pt                   small MLP: physics + EEP + mass -> log age (optional, for ages)
    config.json                      architecture + which context columns W uses
    README.md                        details and caveats
data/format/DSEE_mcdsep_eep_format.json   the input/output normalisation W was trained with (required)
CONF1DENCE/ setup.py README.md LICENSE    the package source (pip-installable), for the rest of the pipeline
scripts/dsee_train/                  the trainer used for W (train_dsee.py, flow_matching.py, shard_dataset.py),
                                     for reproducibility and for training new variants
```
The large original models (OG, etc.) and the raw shards are **not** included — only W.

## Use it (no install needed for prediction)

```bash
pip install torch zuko numpy          # the only requirements for predict_W.py
python predict_W.py --eep 202:605:1 --mass 0.85          # a 0.85 Msun track, default physics
python predict_W.py --eep 300,400,491,550 --mass 0.8 --out pred.npz
python predict_W.py --eep 400 --mass 1.0 --set FeH=-1.0 "Mixing Length=1.9"   # your own physics
```

As a library:
```python
from predict_W import WEmulator
W = WEmulator()                                   # loads data/model/DSEE_W_eep/best_rmse.pt + the format
out = W.predict(eep=[400, 491, 550], mass=0.8)    # dict of central predictions: out["log_Teff"], out["log_L"], ...
```

`predict()` returns the **central (medoid)** prediction by default; pass `raw=True` for the full sample cloud (the
flow is a full conditional density, not a point estimate).


## Make an isochrone

`isochrone_W.py` builds a fixed-age isochrone with the exact recipe behind our cluster figures (age-inversion backbone +
the flow's mode -- see the file's docstring for why each step is done the way it is):

```bash
python isochrone_W.py --age 12.5 --set FeH=-2.27 --plot iso.png     # metal-poor, 12.5 Gyr
```
```python
from isochrone_W import w_isochrone, WEmulator, AgeHead
iso = w_isochrone(WEmulator(), AgeHead(), WEmulator().fmt["label_default"], age_gyr=12.5)   # dict: eep, mass, log_Teff, log_L, ...
```

## Install the full package (optional)

To use the rest of the CONF1DENCE pipeline on the new machine:
```bash
pip install -e .          # from this directory (reads setup.py)
```

## Notes

- EEPs run 202–605 (the trained range); 491 is the RGB base, 580+ is the upper giant branch.
- Physics inputs are named in `data/format/.../label_header`; `label_default` gives sensible defaults, and
  `--set NAME=VALUE` overrides individual ones.
- W drops 6 physics columns that don't affect the tracks; `predict_W.py` handles that automatically (it builds the
  full 24-column context and selects the 18 W uses, from `config.json`'s `label_take`).
