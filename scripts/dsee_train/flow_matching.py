#!/usr/bin/env python
"""Conditional flow matching, as a drop-in alternative to the NSF.

Variant H. Everything else is held at variant C's settings -- the same
sensitivity-weighted EEP acceptance table, the same batch size, learning rate
and optimiser, one GPU -- so the only thing that changes is how the conditional
density is modelled.

METHOD
------
Instead of learning an invertible map and maximising exact likelihood, we learn
a velocity field v(x_t, t, c) that transports noise to data along the straight
(optimal-transport) path

    x0 ~ N(0, I),  x1 = data,  x_t = (1 - t) x0 + t x1,  t ~ U(0, 1)

and regress the constant velocity of that path:

    loss = || v(x_t, t, c) - (x1 - x0) ||^2

Sampling integrates dx/dt = v(x, t, c) from t = 0 to t = 1.

OT COUPLING (--fm-ot, 2026-10-17)
---------------------------------
The pairing of a noise sample x0 to a data sample x1 is arbitrary above: each
data point is matched to an INDEPENDENT Gaussian draw. The straight segments
then cross, so the marginal velocity the network must fit is multi-valued and
curved, which (a) needs many ODE steps to integrate accurately -- the step-count
instability -- and (b) hurts exactly where the conditional density is tight, the
main-sequence log L. Minibatch OT (Tong et al. 2023) instead pairs x0 to x1
within the batch to minimise total transport cost, so the paths cross far less,
the learned field is closer to a straight constant-velocity map, and both
problems ease. ``ot_chunk`` sets the sub-batch the coupling is solved on (exact
OT is O(m^3); chunks keep it cheap and are a standard, unbiased-enough
approximation); the coupling is entropic (Sinkhorn) and sampled from, with the
conditioning c kept attached to the data point.

WHY IT MIGHT HELP HERE
----------------------
The NSF must be invertible, which forces monotonic rational-quadratic splines
and an autoregressive factorisation over the 6 outputs. Flow matching has no
such constraint -- the velocity field is a plain MLP -- so it can allocate
capacity freely. The region that is currently failing (the metal-rich subgiant
to lower-RGB corner, EEP 500-550 at [Fe/H] > 0) is one where log L varies
extremely fast along the coordinate, which is exactly where spline flows tend
to need many knots.

WHAT IT COSTS
-------------
The training loss is an MSE, NOT a negative log-likelihood, so ``best_val`` for
this variant is on a completely different scale and must never be compared with
the NSF variants' values. Compare on held-out log_Teff / log_L accuracy
instead. Sampling is also an ODE solve rather than one pass, so it is slower at
inference by roughly the number of integration steps.
"""
from __future__ import annotations
import math
import torch
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of t in [0,1]; a raw scalar t trains noticeably worse."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * freqs[None, :] * 1000.0
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)


def ot_reorder(x0, x1, chunk, eps=0.05, iters=50):
    """Reorder the noise batch x0 so that x0[j] is paired with data x1[j] (the
    conditioning stays attached to x1 at index j), via minibatch entropic OT.

    Within each chunk of `chunk` rows: squared-euclidean cost between the `chunk`
    data points and the `chunk` noise points, normalised; `iters` Sinkhorn steps
    give a doubly-stochastic coupling P (rows = data, cols = noise); one noise
    index is sampled per data point from its row of P (with replacement -- an
    approximate coupling, as in minibatch OT-CFM). All on-device, static shapes."""
    n = x0.shape[0]
    outs = []
    for s in range(0, n, chunk):
        a = x0[s:s + chunk]; b = x1[s:s + chunk]              # a = noise, b = data
        C = torch.cdist(b.float(), a.float()) ** 2            # (m_data, m_noise)
        C = C / (C.mean() + 1e-8)
        K = torch.exp(-C / eps)
        u = torch.ones(b.shape[0], 1, device=a.device)
        v = torch.ones(a.shape[0], 1, device=a.device)
        for _ in range(iters):
            u = 1.0 / (K @ v + 1e-8)
            v = 1.0 / (K.t() @ u + 1e-8)
        P = u * K * v.t()                                     # (data, noise) coupling
        idx = torch.multinomial(P + 1e-12, 1).squeeze(1)      # one noise per data row
        outs.append(a[idx])
    return torch.cat(outs, 0).to(x0.dtype)


class _Sampler:
    """Mimics the ``flow(context)`` object the rest of the codebase expects, so
    eval and plotting code that calls ``.sample((n,))`` works unchanged."""

    def __init__(self, module: "FlowMatchModule", c: torch.Tensor, steps: int):
        self.m, self.c, self.steps = module, c, steps

    @torch.no_grad()
    def sample(self, shape):
        n = shape[0] if isinstance(shape, (tuple, list)) else int(shape)
        B, d = self.c.shape[0], self.m.n_features
        c = self.c.repeat_interleave(n, dim=0)
        x = torch.randn(B * n, d, device=self.c.device, dtype=self.c.dtype)
        dt = 1.0 / self.steps
        for i in range(self.steps):                      # midpoint (RK2)
            t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
            k1 = self.m.velocity(x, t, c)
            k2 = self.m.velocity(x + 0.5 * dt * k1, t + 0.5 * dt, c)
            x = x + dt * k2
        return x.view(B, n, d).transpose(0, 1).contiguous()   # (n, B, d)


class FlowMatchModule(nn.Module):
    def __init__(self, n_features, n_context, transforms, hidden_features,
                 t_embed=64, ode_steps=32, ot_chunk=0, ot_eps=0.05, ot_iters=50):
        super().__init__()
        self.n_features, self.n_context = n_features, n_context
        self.t_embed, self.ode_steps = t_embed, ode_steps
        self.ot_chunk, self.ot_eps, self.ot_iters = ot_chunk, ot_eps, ot_iters
        width = hidden_features[0] if hidden_features else 256
        depth = len(hidden_features) if hidden_features else 8
        layers, d = [], n_features + n_context + t_embed
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers.append(nn.Linear(d, n_features))
        self.net = nn.Sequential(*layers)

    def velocity(self, x, t, c):
        return self.net(torch.cat([x, c, timestep_embedding(t, self.t_embed)], dim=-1))

    def forward(self, x, c):
        """Per-sample conditional flow-matching loss, shape (B,)."""
        x0 = torch.randn_like(x)
        if self.ot_chunk > 0:
            with torch.no_grad():
                x0 = ot_reorder(x0, x, self.ot_chunk, self.ot_eps, self.ot_iters)
        t = torch.rand(x.shape[0], device=x.device, dtype=x.dtype)
        xt = (1.0 - t)[:, None] * x0 + t[:, None] * x
        target = x - x0
        return ((self.velocity(xt, t, c) - target) ** 2).mean(dim=-1)

    def flow(self, c):
        return _Sampler(self, c, self.ode_steps)
