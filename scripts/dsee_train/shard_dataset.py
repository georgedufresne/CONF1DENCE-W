#!/usr/bin/env python
"""Efficient shard-based IterableDataset for DSEE flow training.

The DSEE training set is 544 HDF5 shards (``DSEE_part{k}.h5``), each with
``data`` (N_k x 6) and ``label`` (N_k x 25) float32 datasets, ~1.8 TB total and
~15.1 billion rows.  Reading it one row at a time through a virtual dataset (as
the original ``readSEM`` did) is catastrophically slow.  This class instead:

  * distributes whole shards across DDP ranks and DataLoader workers (no shared
    global index, no per-row seeks),
  * reads each shard in large *contiguous* blocks (fast sequential I/O that the
    OS page cache can serve from RAM on later epochs),
  * shuffles block order and shuffles rows *within* each block in memory (good
    stochasticity without random disk seeks),
  * yields ready-made mini-batches (set ``batch_size=None`` on the DataLoader),
  * exposes ``steps_for_rank()`` so the trainer can pick a single, identical
    number of optimizer steps per epoch on every rank -- essential to keep DDP
    ranks in lock-step and avoid a hang on uneven shard sizes.

Because each rank/worker owns disjoint shards, there is no data duplication
across the data-parallel group.
"""
from __future__ import annotations

import glob
import os
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


def list_shards(shard_dir: str, pattern: str = "DSEE_part*.h5") -> List[str]:
    """Return shard paths sorted by their integer part index."""
    paths = glob.glob(os.path.join(shard_dir, pattern))
    if not paths:
        raise FileNotFoundError(f"No shards matching {pattern} in {shard_dir}")

    def part_num(p: str) -> int:
        base = os.path.basename(p)
        digits = "".join(ch for ch in base if ch.isdigit())
        return int(digits) if digits else 1 << 62

    return sorted(paths, key=part_num)


def _shard_rows(path: str) -> int:
    """Row count of a shard, cheaply read from attrs (falls back to shape)."""
    with h5py.File(path, "r") as f:
        if "n_rows" in f.attrs:
            return int(f.attrs["n_rows"])
        return int(f["data"].shape[0])


def _measure_accept_rate(paths, edges, prob, col, seed, sample=400_000):
    """Fraction of rows the acceptance table keeps, sampled from these shards."""
    rng = np.random.default_rng(seed)
    kept = tot = 0
    per = max(1024, sample // max(1, len(paths)))
    for p in paths[:16]:
        with h5py.File(p, "r") as f:
            n = f["label"].shape[0]
            a = int(rng.integers(0, max(1, n - per)))
            v = f["label"][a:a + per, col]
        bi = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(prob) - 1)
        kept += float(prob[bi].sum())
        tot += len(v)
    return kept / max(tot, 1)


class ShardIterableDataset(IterableDataset):
    def __init__(
        self,
        shard_paths: List[str],
        batch_size: int,
        *,
        data_cols: int = 6,
        label_cols: int = 25,
        block_size: int = 262144,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        accept_edges=None,
        accept_prob=None,
        accept_col: int = 22,
        accept2_edges=None,
        accept2_prob=None,
        accept2_col: int = 0,
        data_take=None,
        label_take=None,
    ) -> None:
        super().__init__()
        self.batch_size = int(batch_size)
        # ``data_take`` selects a subset of the 6 stored output columns, so a
        # variant can be trained to emit only some of them (e.g. log_Teff and
        # log_L, deriving log_R and log_g from those plus the input mass).
        # ``data_cols`` then means the width the MODEL sees, while the shards on
        # disk keep their full width.
        self.data_take = None if data_take is None else np.asarray(data_take, dtype=np.intp)
        self.data_cols = int(data_cols)
        self.data_cols_disk = 6 if self.data_take is None else 6
        self.label_cols = int(label_cols)
        # ``label_take`` keeps a subset of the stored context columns (e.g. to drop
        # physics inputs that never change the tracks). ``label_cols`` stays the
        # width ON DISK; acceptance columns index the disk layout, because the
        # subset is taken after the acceptance step.
        self.label_take = None if label_take is None else np.asarray(label_take, dtype=np.intp)
        self.block_size = int(block_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0

        # Optional per-row rejection sampling on one label column, used to
        # flatten the EEP distribution.  The native EEP shards keep every
        # stored MESA model, which puts ~37% of rows in the last 6% of the
        # EEP span (small timesteps approaching the helium flash) and only
        # ~7% on the main sequence.  Supplying an acceptance table here
        # trades rows for uniform coverage of the conditioning variable.
        # Left as None the dataset behaves exactly as before.
        self.accept_edges = None if accept_edges is None else np.asarray(accept_edges)
        self.accept_prob = None if accept_prob is None else np.asarray(accept_prob)
        self.accept_col = int(accept_col)
        self.accept_rate = 1.0
        # optional SECOND acceptance axis, multiplied with the first. Used for
        # variant F, which combines an EEP weighting with a metallicity
        # weighting; the label columns are independent so the joint acceptance
        # is just the product.
        self.accept2_edges = None if accept2_edges is None else np.asarray(accept2_edges)
        self.accept2_prob = None if accept2_prob is None else np.asarray(accept2_prob)
        self.accept2_col = int(accept2_col)

        # Round-robin shard assignment balances shard sizes across ranks.
        self.all_shards = list(shard_paths)
        self.rank_shards = self.all_shards[self.rank:: self.world_size]
        if not self.rank_shards:
            raise ValueError(
                f"rank {self.rank}/{self.world_size} received 0 shards "
                f"({len(self.all_shards)} shards total)"
            )
        self.rank_rows = sum(_shard_rows(p) for p in self.rank_shards)
        if self.accept_prob is not None:
            self.accept_rate = _measure_accept_rate(
                self.rank_shards, self.accept_edges, self.accept_prob,
                self.accept_col, self.seed)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def steps_for_rank(self, num_workers: int = 0) -> int:
        """Full mini-batches this rank is *guaranteed* to yield in one epoch.

        The iterator floors to whole batches independently *per worker*: each
        worker owns a disjoint subset of this rank's shards, carries its own
        leftover-row buffer across all of its shards, and (with drop_last) drops
        its final partial batch.  So the true yield is

            sum_over_workers( floor(worker_rows / batch_size) )

        which is up to ``num_workers`` batches fewer than the naive
        ``rank_rows // batch_size``.  Because the shard->worker partition is
        reshuffled every epoch, we return a conservative lower bound that holds
        for *any* partition (each worker loses < 1 batch):

            floor(rank_rows / batch_size) - num_workers

        Using this identical, always-satisfiable count on every rank keeps DDP
        collectives in lock-step and prevents an end-of-epoch hang when one rank
        would otherwise run out of batches early.
        """
        workers = max(int(num_workers), 1)
        rows = self.rank_rows
        if self.accept_prob is not None:
            # expected survival rate, measured on this rank's own shards at
            # construction time; a conservative floor keeps DDP in lock-step.
            rows = int(rows * self.accept_rate * 0.95)
        return max(rows // self.batch_size - workers, 0)

    # -- iteration ---------------------------------------------------------
    def _worker_shards(self) -> List[str]:
        info = get_worker_info()
        num_workers = info.num_workers if info is not None else 1
        worker_id = info.id if info is not None else 0

        shards = list(self.rank_shards)
        if self.shuffle:
            # Same permutation on every worker so the [worker_id::num_workers]
            # slices below stay disjoint and cover all shards exactly once.
            rng = np.random.default_rng(self.seed + 1009 * self.epoch)
            rng.shuffle(shards)
        return shards[worker_id::num_workers]

    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        rng = np.random.default_rng(
            self.seed + 1009 * self.epoch + 131071 * (self.rank + 1) + worker_id
        )

        bs = self.batch_size
        buf_d = None
        buf_l = None

        for path in self._worker_shards():
            with h5py.File(path, "r", swmr=True) as f:
                d_ds = f["data"]
                l_ds = f["label"]
                n = d_ds.shape[0]
                want_d = self.data_cols if self.data_take is None else self.data_cols_disk
                if d_ds.shape[1] != want_d or l_ds.shape[1] != self.label_cols:
                    raise ValueError(
                        f"{path}: expected data cols {want_d} / label "
                        f"cols {self.label_cols}, got {d_ds.shape[1]} / "
                        f"{l_ds.shape[1]}"
                    )

                starts = list(range(0, n, self.block_size))
                if self.shuffle:
                    rng.shuffle(starts)

                for s in starts:
                    e = min(s + self.block_size, n)
                    d = d_ds[s:e]
                    l = l_ds[s:e]
                    if self.data_take is not None:
                        d = d[:, self.data_take]
                    if self.accept_prob is not None:
                        bi = np.clip(
                            np.searchsorted(self.accept_edges,
                                            l[:, self.accept_col], side="right") - 1,
                            0, len(self.accept_prob) - 1)
                        pkeep = self.accept_prob[bi]
                        if self.accept2_prob is not None:
                            bj = np.clip(
                                np.searchsorted(self.accept2_edges,
                                                l[:, self.accept2_col], side="right") - 1,
                                0, len(self.accept2_prob) - 1)
                            pkeep = pkeep * self.accept2_prob[bj]
                        keep = rng.random(len(bi)) < pkeep
                        d = d[keep]
                        l = l[keep]
                        if d.shape[0] == 0:
                            continue
                    if self.label_take is not None:
                        l = l[:, self.label_take]
                    if buf_d is not None:
                        d = np.concatenate([buf_d, d], axis=0)
                        l = np.concatenate([buf_l, l], axis=0)
                        buf_d = buf_l = None
                    if self.shuffle:
                        perm = rng.permutation(d.shape[0])
                        d = d[perm]
                        l = l[perm]

                    nb = d.shape[0] // bs
                    for b in range(nb):
                        sl = slice(b * bs, (b + 1) * bs)
                        yield (
                            torch.from_numpy(np.ascontiguousarray(d[sl])),
                            torch.from_numpy(np.ascontiguousarray(l[sl])),
                        )
                    used = nb * bs
                    if used < d.shape[0]:
                        buf_d = d[used:].copy()
                        buf_l = l[used:].copy()

        if (not self.drop_last) and buf_d is not None and buf_d.shape[0] > 0:
            yield (
                torch.from_numpy(np.ascontiguousarray(buf_d)),
                torch.from_numpy(np.ascontiguousarray(buf_l)),
            )


def dims_from_format(fmt: dict) -> Tuple[int, int]:
    """(n_features, n_context) implied by a *_format.json scaler layout."""
    nf = sum(2 if s == "mantissa_exponent" else 1 for s in fmt["data_scaler"])
    nc = sum(2 if s == "mantissa_exponent" else 1 for s in fmt["label_scaler"])
    return nf, nc
