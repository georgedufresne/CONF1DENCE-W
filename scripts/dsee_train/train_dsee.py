#!/usr/bin/env python
"""Multi-GPU (DDP) trainer for the DSEE zuko NSF flow with robust checkpointing.

Architecture is unchanged from ``CONF1DENCE.train.train.TrainMod``:
    zuko.flows.NSF(features=6, context=25, transforms=10, hidden_features=[256]*10)
loss = -flow(context).log_prob(x).mean().

What this adds over the original loop:
  * DistributedDataParallel across all GPUs on the node (launch with torchrun).
  * A fast shard-based data pipeline (see shard_dataset.py) instead of per-row
    reads from a 15B-row virtual dataset.
  * Time-based *and* step-based checkpointing to DartFS (durable), written
    atomically, with automatic resume on (re)start -> preemption costs minutes.
  * Signal-triggered checkpoint (SLURM ``--signal=USR1@...``) so a preemption
    loses essentially no work.
  * DDP-safe fixed steps/epoch (no hang on uneven shard sizes) and DDP-safe
    NaN handling (bad samples masked, every rank still runs backward).
  * Periodic validation on held-out shards; keeps a ``best.pt``.

Launch (single node, N GPUs):
    torchrun --standalone --nproc_per_node=N train_dsee.py --shard-dir ... [opts]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import zuko

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shard_dataset import ShardIterableDataset, dims_from_format, list_shards


# --------------------------------------------------------------------------- #
# Model wrapper: DDP needs forward() to return tensors (not a distribution),
# so we return the per-sample negative log-likelihood.
# --------------------------------------------------------------------------- #
class FlowModule(nn.Module):
    def __init__(self, n_features, n_context, transforms, hidden_features):
        super().__init__()
        self.flow = zuko.flows.NSF(
            features=n_features,
            context=n_context,
            transforms=transforms,
            hidden_features=hidden_features,
        )

    def forward(self, x, c):
        return -self.flow(c).log_prob(x)  # (B,)


# --------------------------------------------------------------------------- #
# Distributed helpers
# --------------------------------------------------------------------------- #
def ddp_setup():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def is_dist():
    return dist.is_available() and dist.is_initialized()


def barrier():
    if is_dist():
        dist.barrier()


def all_reduce_min_int(value, device):
    if not is_dist():
        return int(value)
    t = torch.tensor([int(value)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return int(t.item())


def all_reduce_max_flags(flags, device):
    """flags: list of ints -> element-wise MAX across ranks (returns list)."""
    if not is_dist():
        return list(flags)
    t = torch.tensor(flags, device=device, dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.tolist()


# --------------------------------------------------------------------------- #
# Signal handling (preemption): set a flag; the training loop checkpoints and
# stops at the next safe boundary.
# --------------------------------------------------------------------------- #
_SIGNAL_STOP = {"flag": False}


def _install_signal_handlers():
    def handler(signum, frame):
        _SIGNAL_STOP["flag"] = True

    for sig in (signal.SIGUSR1, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


# --------------------------------------------------------------------------- #
# Checkpoint I/O (rank 0 only), atomic via tmp + os.replace.
# --------------------------------------------------------------------------- #
def save_checkpoint(path, model, optimizer, scaler, scheduler, epoch, global_step,
                    best_val, config):
    tmp = path + ".tmp"
    payload = {
        "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_val": best_val,
        "config": config,
        "torch_rng": torch.get_rng_state(),
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer, scaler, scheduler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    (model.module if isinstance(model, DDP) else model).load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if ckpt.get("torch_rng") is not None:
        try:
            torch.set_rng_state(ckpt["torch_rng"].cpu())
        except Exception:
            pass
    return ckpt.get("epoch", 0), ckpt.get("global_step", 0), ckpt.get("best_val", float("inf"))


# --------------------------------------------------------------------------- #
# Validation (rank 0 only; other ranks wait at a barrier)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(model, val_shards, args, device, n_features, n_context, data_take=None, label_take=None):
    module = model.module if isinstance(model, DDP) else model
    module.eval()
    ds = ShardIterableDataset(
        val_shards, batch_size=args.batch_size,
        data_cols=n_features, label_cols=n_context, data_take=data_take, label_take=label_take,
        block_size=args.block_size, shuffle=False, drop_last=True,
        seed=args.seed, rank=0, world_size=1,
    )
    loader = DataLoader(ds, batch_size=None, num_workers=min(4, args.num_workers),
                        pin_memory=True)
    total = 0.0
    count = 0
    for i, (x, c) in enumerate(loader):
        if i >= args.val_steps:
            break
        x = x.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True)
        nll = module(x, c)
        nll = nll[torch.isfinite(nll)]
        if nll.numel():
            total += nll.sum().item()
            count += nll.numel()
    module.train()
    return total / max(count, 1)


# --------------------------------------------------------------------------- #
# Interpretable progress metric and checkpoint snapshots
# --------------------------------------------------------------------------- #
PROBE_PHASES = [("MS", 202, 353), ("turnoff", 353, 454), ("SGB", 454, 491), ("RGB", 491, 580), ("tip", 580, 606)]


def build_probe(val_shards, n_rows, fmt, data_take, label_take, device, seed=0):
    """Fixed probe set: rows of COMPLETE tracks (status 0) from the validation shards, the same number in each phase
    (by EEP). Returns None if the shards carry no per-row status."""
    import h5py
    rng = np.random.default_rng(seed)
    hdr = list(fmt["label_header"]); ie = hdr.index("EEP"); lo, hi = fmt["label_scale"][ie]
    per = max(1, n_rows // len(PROBE_PHASES))
    got = [[] for _ in PROBE_PHASES]
    shards = list(val_shards); rng.shuffle(shards)
    for p in shards:
        with h5py.File(p, "r") as f:
            if "status" not in f:
                return None
            n = f["label"].shape[0]
            for s0 in rng.integers(0, max(1, n - 4000), 24):
                sl = slice(int(s0), int(s0) + 4000)
                st = f["status"][sl]; lab = f["label"][sl]; dat = f["data"][sl]
                e = lab[:, ie] * (hi - lo) + lo
                for k, (_, a, b) in enumerate(PROBE_PHASES):
                    j = np.nonzero((st == 0) & (e >= a) & (e < b))[0]
                    if len(j) and sum(len(x[0]) for x in got[k]) < per:
                        j = rng.choice(j, min(len(j), 200), replace=False)
                        got[k].append((dat[j], lab[j]))
        if all(sum(len(x[0]) for x in g) >= per for g in got):
            break
    D, L, P = [], [], []
    for k, g in enumerate(got):
        if not g:
            continue
        d = np.concatenate([x[0] for x in g])[:per]; l = np.concatenate([x[1] for x in g])[:per]
        D.append(d); L.append(l); P.append(np.full(len(d), k))
    D = np.concatenate(D); L = np.concatenate(L); P = np.concatenate(P)
    if data_take is not None:
        D = D[:, data_take]
    if label_take is not None:
        L = L[:, label_take]
    dh = list(fmt["data_header"]); cols = data_take if data_take is not None else list(range(len(dh)))
    names = [dh[i] for i in cols]; span = np.array([fmt["data_scale"][i][1] - fmt["data_scale"][i][0] for i in cols])
    q = {nm: names.index(nm) for nm in ("log_Teff", "log_L") if nm in names}
    T = lambda a: torch.tensor(np.ascontiguousarray(a), device=device, dtype=torch.float32)
    return {"x": T(D), "c": T(L), "phase": P, "span": span, "cols": q}


@torch.no_grad()
def probe_eval(core, probe, arch):
    """Central prediction (NSF: inverse of the base-space origin; FM: ODE from x = 0) on the probe rows.
    Returns {quantity: [median |error| in mdex per phase]} and the mean over phases per quantity."""
    was = core.training
    core.eval()
    nf = probe["x"].shape[1]; out = []
    for s in range(0, len(probe["c"]), 8192):
        c = probe["c"][s:s + 8192]
        if arch == "fm":
            x = torch.zeros(len(c), nf, device=c.device); steps = getattr(core, "ode_steps", 32); dt = 1.0 / steps
            for i in range(steps):
                t = torch.full((len(c),), i * dt, device=c.device)
                k1 = core.velocity(x, t, c); k2 = core.velocity(x + 0.5 * dt * k1, t + 0.5 * dt, c)
                x = x + dt * k2
        else:
            x = core.flow(c).transform.inv(torch.zeros(len(c), nf, device=c.device))
        out.append(x.float())
    core.train(was)
    err = ((torch.cat(out) - probe["x"]).abs().cpu().numpy()) * probe["span"] * 1e3
    res = {}
    for nm, j in probe["cols"].items():
        res[nm] = [float(np.median(err[probe["phase"] == k, j])) if (probe["phase"] == k).any() else float("nan")
                   for k in range(len(PROBE_PHASES))]
    return res


def save_snapshot(ckpt_dir, core, epoch, global_step, config, probe_res):
    d = os.path.join(ckpt_dir, "snapshots"); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"step{global_step:09d}.pt")
    torch.save({"model": core.state_dict(), "config": config, "epoch": epoch, "global_step": global_step,
                "probe": probe_res, "time": time.time()}, p + ".tmp")
    os.replace(p + ".tmp", p)
    return p


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--format", required=True, help="path to *_format.json")
    ap.add_argument("--arch", default="nsf", choices=["nsf","fm"],
                    help="nsf = zuko neural spline flow (A-G); fm = conditional "
                         "flow matching (variant H). NOTE: 'fm' optimises an MSE, "
                         "not a likelihood, so its loss and best_val are on a "
                         "different scale and must not be compared with nsf runs.")
    ap.add_argument("--ode-steps", type=int, default=32,
                    help="midpoint-rule steps used when sampling an fm model")
    ap.add_argument("--fm-ot", type=int, default=0,
                    help="arch=fm: minibatch optimal-transport noise<->data coupling, solved on sub-batches of this "
                         "size (0 = the old independent/random pairing, variant H). Straightens the paths: fewer ODE "
                         "steps and better main-sequence log L. 1024 is a good default.")
    ap.add_argument("--fm-ot-eps", type=float, default=0.05, help="arch=fm: Sinkhorn entropic regularisation for --fm-ot")
    ap.add_argument("--fm-ot-iters", type=int, default=50, help="arch=fm: Sinkhorn iterations for --fm-ot")
    ap.add_argument("--data-take", default=None,
                    help="comma-separated indices into the 6 stored output columns "
                         "to train on, e.g. '0,2' for log_Teff and log_L only. "
                         "The format file must list exactly those columns.")
    ap.add_argument("--ckpt-dir", required=True, help="DURABLE dir (DartFS/home)")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--block-size", type=int, default=262144)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resume-new-epoch", action="store_true",
                    help="on resume, start the NEXT epoch number (fresh shuffle) instead of replaying the saved epoch "
                         "from its first block. Timed checkpoints store the CURRENT epoch, and the data order is seeded by "
                         "seed + 1009*epoch, so without this every mid-epoch restart re-reads the same leading blocks.")
    ap.add_argument("--resume-lr", type=float, default=None,
                    help="override optimizer LR after loading latest.pt")
    ap.add_argument("--lr-scheduler", choices=["none", "plateau"], default="plateau",
                    help="LR schedule. 'plateau' = ReduceLROnPlateau on val_nll, "
                         "mirroring the paper's proven recipe (default).")
    ap.add_argument("--lr-factor", type=float, default=0.5,
                    help="ReduceLROnPlateau: multiply LR by this on plateau.")
    ap.add_argument("--lr-patience", type=int, default=10,
                    help="ReduceLROnPlateau: epochs with no val improvement "
                         "before decaying LR.")
    ap.add_argument("--lr-threshold", type=float, default=1e-4,
                    help="ReduceLROnPlateau: min val_nll improvement to count.")
    ap.add_argument("--min-lr", type=float, default=1e-6,
                    help="ReduceLROnPlateau: lower bound on LR.")
    ap.add_argument("--compile", action="store_true",
                    help="wrap the model in torch.compile for operator fusion "
                         "(biggest remaining speedup on L40S).")
    ap.add_argument("--transforms", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--hidden-layers", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = unlimited")
    ap.add_argument("--num-workers", type=int, default=-1, help="-1 = auto")
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--eep-uniform-table", default=None,
                    help="JSON with {edges, prob, col}: per-row acceptance "
                         "probabilities that flatten the EEP distribution. "
                         "Omit for the native (every stored model) sampling.")
    ap.add_argument("--eep-uniform-table2", default=None,
                    help="second acceptance table, multiplied with the first "
                         "(e.g. a metallicity weighting on label column 0)")
    ap.add_argument("--val-shards", type=int, default=8, help="held-out shards")
    ap.add_argument("--test-shards", type=int, default=0,
                    help="the LAST this-many shards are a test set the trainer never reads (not even for best.pt selection); the --val-shards validation shards are the ones just before them. 85/10/5 of 420 shards: --val-shards 42 --test-shards 21")
    ap.add_argument("--label-drop", default=None,
                    help="comma-separated context column NAMES (from the format's label_header) to leave out of the model's context, e.g. physics inputs that do not change the tracks. The checkpoint config stores the kept disk indices as label_take; loaders must slice a full-width context with it.")
    ap.add_argument("--val-steps", type=int, default=400)
    ap.add_argument("--ckpt-minutes", type=float, default=30.0)
    ap.add_argument("--max-walltime-min", type=float, default=0.0,
                    help="stop cleanly after this many minutes (0 = no limit)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true",
                    help="[deprecated] alias for --precision fp16")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32",
                    help="autocast compute precision. bf16 is fastest on "
                         "L40S/Ampere+ and needs no gradient scaler (default: fp32)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--compile-mode", default="default", choices=["default", "reduce-overhead"],
                    help="torch.compile mode with --compile. reduce-overhead = CUDA graphs: one launch per captured "
                         "step instead of thousands of small kernels (5.5x the steps/s of default for the 10x256 NSF "
                         "at batch 16384 on a GH200, where default is CPU-launch bound with the GPU ~85%% idle)")
    ap.add_argument("--lr-schedule-steps", type=int, default=0,
                    help="if > 0: cosine learning-rate decay from --lr to --lr-min over this many OPTIMIZER STEPS "
                         "(global_step); takes precedence over --lr-schedule-minutes")
    ap.add_argument("--lr-schedule-minutes", type=float, default=0.0,
                    help="if > 0: cosine learning-rate decay from --lr to --lr-min over this many TRAINING MINUTES, "
                         "counted across pieces (the budget is in minutes); replaces the plateau scheduler")
    ap.add_argument("--lr-done-minutes", type=float, default=0.0,
                    help="training minutes already done in earlier pieces (the run's ledger), for the cosine position")
    ap.add_argument("--lr-min", type=float, default=1e-6, help="floor of the cosine schedule")
    ap.add_argument("--lr-table", default="",
                    help="JSON {\"step\": [...], \"lr\": [...]}: learning rate interpolated in global_step from this table "
                         "(e.g. the lr another run actually had at each step, so a seed replica follows the same schedule "
                         "however its pieces fall); takes precedence over --lr-schedule-steps/-minutes")
    ap.add_argument("--probe-rows", type=int, default=0,
                    help="if > 0: a fixed set of this many validation rows (complete tracks, equal per phase); every "
                         "log line then also prints the central prediction's median |d log Teff| and |d log L| in "
                         "mdex (mean over phases), and every checkpoint prints them per phase")
    ap.add_argument("--snapshots", action="store_true",
                    help="also keep a model-only copy of every checkpoint in <ckpt-dir>/snapshots/, so the final "
                         "model can be chosen afterwards on per-phase error (dsee1/score_snapshots.py)")
    ap.add_argument("--sync-every", type=int, default=25,
                    help="steps between DDP control-flag syncs")
    args = ap.parse_args()
    # Per-run overrides: <ckpt-dir>/run_overrides.json, read at start-up, so an already-queued job can be retargeted
    # (e.g. CUDA graphs, a step budget) without resubmitting it and losing its place in the queue.
    _ov_path = os.path.join(args.ckpt_dir, "run_overrides.json")
    if os.path.exists(_ov_path):
        with open(_ov_path) as _fh:
            _ov = json.load(_fh)
        for _k, _v in _ov.items():
            if _k.startswith("_"):
                continue
            if not hasattr(args, _k):
                raise SystemExit(f"{_ov_path}: unknown option {_k}")
            setattr(args, _k, _v)
        print(f"[cfg] overrides from {_ov_path}: " + ", ".join(f"{k}={v}" for k, v in _ov.items() if not k.startswith("_")), flush=True)

    # Resolve compute precision. --amp is the old fp16 switch; honour it only
    # if --precision was left at its default.
    if args.amp and args.precision == "fp32":
        args.precision = "fp16"
    amp_enabled = args.precision in ("fp16", "bf16")
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    t_start = time.time()
    rank, world_size, local_rank = ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_master = rank == 0
    _install_signal_handlers()
    torch.manual_seed(args.seed + rank)

    if args.num_workers < 0:
        try:
            avail = len(os.sched_getaffinity(0))
        except AttributeError:
            avail = os.cpu_count() or 2
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
        args.num_workers = max(2, avail // max(local_world, 1) - 1)

    with open(args.format) as fh:
        fmt = json.load(fh)
    n_features, n_context = dims_from_format(fmt)
    data_take = None
    if args.data_take:
        data_take = [int(x) for x in args.data_take.split(",")]
        if len(data_take) != n_features:
            raise SystemExit(f"--data-take lists {len(data_take)} columns but "
                             f"{args.format} declares {n_features}")

    label_take = None
    n_context_disk = n_context
    if args.label_drop:
        hdr = list(fmt["label_header"])
        drop = [x.strip() for x in args.label_drop.split(",") if x.strip()]
        unknown = [x for x in drop if x not in hdr]
        if unknown:
            raise SystemExit(f"--label-drop: {unknown} not in label_header {hdr}")
        if any(fmt["label_scaler"][hdr.index(x)] == "mantissa_exponent" for x in drop):
            raise SystemExit("--label-drop does not support mantissa_exponent columns")
        label_take = [i for i, x in enumerate(hdr) if x not in drop]
        n_context = len(label_take)

    all_shards = list_shards(args.shard_dir)
    test_shards = []
    if args.test_shards > 0:
        test_shards = all_shards[-args.test_shards:]
        all_shards = all_shards[: -args.test_shards]
    if args.val_shards > 0:
        train_shards = all_shards[: -args.val_shards]
        val_shards = all_shards[-args.val_shards:]
    else:
        train_shards = all_shards
        val_shards = []

    if is_master:
        print(f"[cfg] world_size={world_size} device={device} "
              f"n_features={n_features} n_context={n_context}", flush=True)
        if label_take is not None:
            print(f"[cfg] context: {n_context} of {n_context_disk} columns (dropped {args.label_drop})", flush=True)
        if test_shards:
            print(f"[cfg] test shards (never read): {os.path.basename(test_shards[0])} .. "
                  f"{os.path.basename(test_shards[-1])}", flush=True)
        print(f"[cfg] shards: {len(train_shards)} train / {len(val_shards)} val / {len(test_shards)} test | "
              f"batch={args.batch_size} block={args.block_size} "
              f"workers={args.num_workers} precision={args.precision}", flush=True)

    # --- data ---
    acc = {}
    if args.eep_uniform_table:
        with open(args.eep_uniform_table) as fh:
            t = json.load(fh)
        acc = dict(accept_edges=t["edges"], accept_prob=t["prob"],
                   accept_col=t.get("col", 22))
        if args.eep_uniform_table2:
            with open(args.eep_uniform_table2) as fh2:
                t2 = json.load(fh2)
            acc.update(accept2_edges=t2["edges"], accept2_prob=t2["prob"],
                       accept2_col=t2.get("col", 0))
            if is_master:
                print("[cfg] second acceptance axis from %s (col %d)"
                      % (args.eep_uniform_table2, t2.get("col", 0)), flush=True)
        if is_master:
            print("[cfg] EEP-uniform sampling from %s (%d bins)"
                  % (args.eep_uniform_table, len(t["prob"])), flush=True)
    train_ds = ShardIterableDataset(
        train_shards, batch_size=args.batch_size,
        data_cols=n_features, label_cols=n_context_disk, data_take=data_take, label_take=label_take,
        block_size=args.block_size, shuffle=True, drop_last=True,
        seed=args.seed, rank=rank, world_size=world_size, **acc,
    )
    if args.eep_uniform_table and is_master:
        print("[cfg] acceptance keeps %.1f%% of rows" % (100 * train_ds.accept_rate),
              flush=True)
    loader = DataLoader(
        train_ds, batch_size=None, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
    )
    steps_per_epoch = all_reduce_min_int(train_ds.steps_for_rank(args.num_workers), device)
    if is_master:
        print(f"[cfg] steps/epoch (DDP-synced min)={steps_per_epoch:,}", flush=True)

    # --- model / optim ---
    # `core` is the canonical, uncompiled, un-DDP'd module: it owns the
    # parameters and is the ONLY thing we ever state_dict() / load into, so
    # checkpoints stay portable regardless of --compile or world_size.
    if args.arch == "fm":
        from flow_matching import FlowMatchModule
        core = FlowMatchModule(n_features, n_context, args.transforms,
                               [args.hidden] * args.hidden_layers,
                               ode_steps=args.ode_steps, ot_chunk=args.fm_ot,
                               ot_eps=args.fm_ot_eps, ot_iters=args.fm_ot_iters).to(device)
        if is_master:
            print(f"[cfg] flow matching: MLP {args.hidden}x{args.hidden_layers}, ode_steps={args.ode_steps}, "
                  f"OT coupling {'chunk ' + str(args.fm_ot) + f' (eps {args.fm_ot_eps}, {args.fm_ot_iters} Sinkhorn iters)' if args.fm_ot else 'OFF (random pairing)'}", flush=True)
    else:
        core = FlowModule(n_features, n_context, args.transforms,
                      [args.hidden] * args.hidden_layers).to(device)
    if world_size > 1:
        ddp_kwargs = {"device_ids": [local_rank]} if torch.cuda.is_available() else {}
        model = DDP(core, **ddp_kwargs)
    else:
        model = core
    # Compiled handle used only for the hot forward/backward. torch.compile
    # wraps the DDP module (supported; DDPOptimizer handles graph breaks). If
    # zuko trips something, fall back to eager by dropping --compile.
    if args.compile:
        fwd_model = torch.compile(model) if args.compile_mode == "default" else torch.compile(model, mode=args.compile_mode)
    else:
        fwd_model = model
    optimizer = torch.optim.Adam(core.parameters(), lr=args.lr)
    # GradScaler is only meaningful for fp16; bf16 has fp32 dynamic range and
    # needs no loss scaling, fp32 needs none either.
    scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16"))
    # LR scheduler. ReduceLROnPlateau on val_nll mirrors the paper recipe; the
    # original DDP rewrite dropped this, which let the constant LR wander out of
    # the basin. Stepped once per epoch on the (broadcast) val metric.
    scheduler = None
    if args.lr_scheduler == "plateau" and args.lr_schedule_minutes <= 0 and args.lr_schedule_steps <= 0 and not args.lr_table:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor,
            patience=args.lr_patience, threshold=args.lr_threshold,
            min_lr=args.min_lr)

    config = vars(args) | {"n_features": n_features, "n_context": n_context,
                           "n_context_disk": n_context_disk, "label_take": label_take,
                           "data_take": data_take, "arch": args.arch,
                           "test_shards_list": [os.path.basename(p) for p in test_shards]}

    # --- resume ---
    os.makedirs(args.ckpt_dir, exist_ok=True)
    latest_path = os.path.join(args.ckpt_dir, "latest.pt")
    best_path = os.path.join(args.ckpt_dir, "best.pt")
    done_marker = os.path.join(args.ckpt_dir, "COMPLETED")
    start_epoch, global_step, best_val = 0, 0, float("inf")
    if os.path.exists(latest_path):
        start_epoch, global_step, best_val = load_checkpoint(
            latest_path, model, optimizer, scaler, scheduler, device)
        if args.resume_new_epoch:
            start_epoch += 1
        if args.resume_lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = args.resume_lr
            if scheduler is not None:
                scheduler._last_lr = [args.resume_lr] * len(optimizer.param_groups)
        if is_master:
            print(f"[resume] from {latest_path}: epoch={start_epoch} "
                  f"step={global_step} best_val={best_val:.4f} "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)
    barrier()

    probe = None
    if is_master and args.probe_rows > 0 and val_shards:
        cache = os.path.join(args.ckpt_dir, f"probe_cache_{args.probe_rows}.pt")    # same rows every piece; built once
        if os.path.exists(cache):
            probe = torch.load(cache, map_location=device, weights_only=False)
        else:
            probe = build_probe(val_shards, args.probe_rows, fmt, data_take, label_take, device)
            if probe is not None:
                torch.save(probe, cache + ".tmp"); os.replace(cache + ".tmp", cache)
        if probe is not None:
            print(f"[probe] {len(probe['c']):,} validation rows of complete tracks, "
                  f"{len(probe['c']) // len(PROBE_PHASES):,} per phase ({', '.join(p[0] for p in PROBE_PHASES)})", flush=True)
    probe_res = None

    def probe_line(tag):
        r = probe_eval(core, probe, args.arch)
        print(f"[probe] {tag} median |error| mdex by phase (" + "/".join(p[0] for p in PROBE_PHASES) + "): "
              + "  ".join(f"{nm} " + "/".join(f"{v:.2f}" for v in vals) for nm, vals in r.items()), flush=True)
        return r

    def cosine_lr():
        """lr at the current point of the run. Step clock (--lr-schedule-steps) if given, else minutes done in earlier
        pieces + minutes since this process started (the clock the piece ledger and --max-walltime-min use)."""
        if lr_tab is not None:
            return float(np.interp(global_step, lr_tab[0], lr_tab[1]))
        if args.lr_schedule_steps > 0:
            f = min(max(global_step / args.lr_schedule_steps, 0.0), 1.0)
        else:
            t = args.lr_done_minutes + (time.time() - t_start) / 60.0
            f = min(max(t / args.lr_schedule_minutes, 0.0), 1.0)
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1.0 + math.cos(math.pi * f))

    lr_tab = None
    if args.lr_table:
        with open(args.lr_table) as fh:
            _t = json.load(fh)
        lr_tab = (np.asarray(_t["step"], np.float64), np.asarray(_t["lr"], np.float64))
    use_cosine = lr_tab is not None or args.lr_schedule_steps > 0 or args.lr_schedule_minutes > 0
    if use_cosine:
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr()
        if is_master:
            over = (f"the lr table {args.lr_table} ({len(lr_tab[0])} points to step {int(lr_tab[0][-1]):,}; "
                    f"step {global_step:,} now)" if lr_tab is not None
                    else f"{args.lr_schedule_steps:,} steps (step {global_step:,} now)" if args.lr_schedule_steps > 0
                    else f"{args.lr_schedule_minutes:.0f} training min ({args.lr_done_minutes:.0f} done)")
            print(f"[cfg] cosine lr {args.lr:.1e} -> {args.lr_min:.1e} over {over}; "
                  f"lr now {optimizer.param_groups[0]['lr']:.2e}", flush=True)

    def write_progress():
        with open(os.path.join(args.ckpt_dir, "progress.json.tmp"), "w") as fh:
            json.dump({"global_step": global_step, "rows_seen": global_step * args.batch_size * world_size,
                       "batch_size": args.batch_size, "epoch": epoch, "time": time.time()}, fh)
        os.replace(os.path.join(args.ckpt_dir, "progress.json.tmp"), os.path.join(args.ckpt_dir, "progress.json"))

    walltime_limit = args.max_walltime_min * 60.0 if args.max_walltime_min > 0 else None
    last_ckpt = time.time()
    stop_all = False

    if args.max_steps and global_step >= args.max_steps:
        if is_master:
            print(f"[done] step budget already reached: step {global_step:,} >= --max-steps {args.max_steps:,}", flush=True)
            epoch = start_epoch
            write_progress()
        if is_dist():
            dist.destroy_process_group()
        sys.exit(0)

    for epoch in range(start_epoch, args.max_epochs):
        train_ds.set_epoch(epoch)
        core.train()
        # Kept as GPU tensors and only pulled to host (.item()) at the
        # log_every cadence -- avoids a CPU<->GPU sync every single step
        # (torch.isfinite(...).any()/.item() calls used to block the CPU
        # from queuing the next step's kernels while the GPU caught up).
        running = torch.zeros((), device=device)
        running_n = torch.zeros((), device=device, dtype=torch.long)
        bad_samples = torch.zeros((), device=device, dtype=torch.long)
        ep_t = time.time()
        last_log_t = time.time()

        for step, (x, c) in enumerate(loader):
            if step >= steps_per_epoch:
                break
            x = x.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)

            # Scrub any non-finite *inputs* before the forward. Masking the
            # loss after the fact does NOT protect the shared flow parameters:
            # a NaN/Inf that reaches the graph makes backward compute 0*NaN=NaN
            # for the shared weights, which the next optimizer step writes into
            # the model permanently. nan_to_num keeps the batch shape static
            # (no GPU->CPU sync) so the hot loop stays fast.
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            c = torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)

            optimizer.zero_grad(set_to_none=True)
            if use_cosine and global_step % 25 == 0:
                lr_now = cosine_lr()
                for group in optimizer.param_groups:
                    group["lr"] = lr_now
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                nll = fwd_model(x, c)                   # (B,)
                finite = torch.isfinite(nll)
                n_finite = finite.sum()
                # Branch-free: zero out non-finite entries but still route a
                # (zero) gradient through every element so all ranks' graphs
                # match and DDP stays in sync, even if a rank has 0 finite.
                safe_nll = torch.where(finite, nll, torch.zeros_like(nll))
                loss_sum = safe_nll.sum()
                loss = loss_sum / n_finite.clamp(min=1)

            scaler.scale(loss).backward()
            model_ref = core
            # DDP-safe gradient guard. After backward the grads are already
            # all-reduced, so they are identical on every rank and this
            # decision stays consistent across the group. Replacing any
            # non-finite grad with 0 (rather than clipping, whose norm would
            # itself be NaN and scale every grad to NaN) guarantees a transient
            # NaN from one batch can never poison the weights -- the offending
            # step is simply skipped for those parameters and training survives.
            scaler.unscale_(optimizer)
            for p in model_ref.parameters():
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model_ref.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            bad_samples += nll.numel() - n_finite
            running += loss_sum.detach()
            running_n += n_finite.detach()

            if is_master and args.log_every and (global_step % args.log_every == 0):
                avg = (running / running_n.clamp(min=1)).item()
                bad = int(bad_samples.item())
                now = time.time()
                sps = args.log_every / max(now - last_log_t, 1e-9)
                last_log_t = now
                samp_s = sps * args.batch_size * world_size
                extra = ""
                if probe is not None:
                    r = probe_eval(core, probe, args.arch)
                    extra = " | probe " + " ".join(f"{nm.replace('log_', '')} {np.nanmean(v):.2f}" for nm, v in r.items()) + " mdex"
                    last_log_t = time.time()           # keep the probe's own cost out of steps/s
                print(f"[e{epoch} s{step}/{steps_per_epoch} g{global_step}] "
                      f"loss={avg:.4f} bad={bad} lr={optimizer.param_groups[0]['lr']:.2e} "
                      f"{sps:.1f} steps/s {samp_s/1e6:.2f}M samp/s{extra}", flush=True)

            # -- DDP-synced control decisions (checkpoint / stop) --
            if global_step % args.sync_every == 0 or _SIGNAL_STOP["flag"]:
                now = time.time()
                sig = 1 if _SIGNAL_STOP["flag"] else 0
                time_ckpt = 1 if (now - last_ckpt) >= args.ckpt_minutes * 60 else 0
                wall_stop = 1 if (walltime_limit and (now - t_start) >= walltime_limit) else 0
                step_stop = 1 if (args.max_steps and global_step >= args.max_steps) else 0
                # rank 0 drives time/wall; signal can arrive on any rank.
                flags = [
                    (time_ckpt if is_master else 0) or sig,
                    sig or (wall_stop if is_master else 0) or (step_stop if is_master else 0),
                ]
                do_ckpt, do_stop = all_reduce_max_flags(flags, device)
                if do_ckpt or do_stop:
                    barrier()
                    if is_master:
                        save_checkpoint(latest_path, core, optimizer, scaler,
                                        scheduler, epoch, global_step, best_val, config)
                        print(f"[ckpt] latest @ epoch={epoch} step={global_step} "
                              f"({'signal' if sig else 'timed'})", flush=True)
                        write_progress()
                        if probe is not None:
                            probe_res = probe_line(f"step {global_step}")
                        if args.snapshots:
                            save_snapshot(args.ckpt_dir, core, epoch, global_step, config, probe_res)
                    last_ckpt = time.time()
                    barrier()
                if do_stop:
                    stop_all = True
                    break

        if stop_all:
            if is_master:
                print("[stop] preemption/walltime/max-steps -> exiting for requeue",
                      flush=True)
            break

        # -- end-of-epoch: validation + checkpoints --
        barrier()
        val = float("nan")
        if val_shards:
            if is_master:
                val = validate(core, val_shards, args, device, n_features, n_context_disk, data_take, label_take)
                print(f"[val] epoch={epoch} val_nll={val:.4f} "
                      f"(epoch time {time.time()-ep_t:.0f}s)", flush=True)
                if val < best_val:
                    best_val = val
                    save_checkpoint(best_path, core, optimizer, scaler,
                                    scheduler, epoch + 1, global_step, best_val, config)
                    print(f"[val] new best -> {best_path}", flush=True)
            barrier()

        # -- LR schedule: step on the val metric, broadcast so every rank keeps
        # an identical LR (val is computed on rank 0 only). Done before the
        # end-of-epoch checkpoint so the scheduler state is persisted in it.
        if scheduler is not None and val_shards:
            vt = torch.tensor([val if is_master else 0.0], device=device,
                              dtype=torch.float32)
            if is_dist():
                dist.broadcast(vt, src=0)
            val_b = float(vt.item())
            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_b)
            new_lr = optimizer.param_groups[0]["lr"]
            if is_master and new_lr != prev_lr:
                print(f"[sched] plateau: val_nll={val_b:.4f} "
                      f"lr {prev_lr:.2e} -> {new_lr:.2e}", flush=True)

        if is_master:
            save_checkpoint(latest_path, core, optimizer, scaler,
                            scheduler, epoch + 1, global_step, best_val, config)
            print(f"[ckpt] end of epoch {epoch} -> latest", flush=True)
            write_progress()
            if probe is not None:
                probe_res = probe_line(f"end of epoch {epoch}, step {global_step}")
            if args.snapshots:
                save_snapshot(args.ckpt_dir, core, epoch + 1, global_step, config, probe_res)
        last_ckpt = time.time()
        barrier()

    if is_master and not stop_all:
        with open(done_marker, "w") as fh:
            fh.write(f"epoch={min(epoch+1, args.max_epochs)} step={global_step}\n")
        print("[done] training complete (COMPLETED marker written)", flush=True)

    if is_dist():
        dist.destroy_process_group()
    # Non-zero exit on preemption stop makes SLURM requeue treat it as needing rerun.
    sys.exit(0 if (not stop_all) else 0)


if __name__ == "__main__":
    main()
