"""Proximal Policy Optimisation (asymmetric actor-critic) in JAX.

The actor is a plain MLP on normalised on-board observations; it is exported
as numpy arrays (see ``export_policy``) so a robot computer can run it with
nothing but numpy (``malecns_body.deploy``).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, asdict

import jax
import jax.numpy as jnp
import numpy as np
import optax


# ----------------------------------------------------------------------------
# networks
# ----------------------------------------------------------------------------

def init_mlp(key, sizes, out_scale=1.0):
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for i, (k, n_in, n_out) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
        last = i == len(sizes) - 2
        gain = out_scale if last else np.sqrt(2.0)
        W = jax.nn.initializers.orthogonal(gain)(k, (n_in, n_out), jnp.float32)
        params.append((W, jnp.zeros(n_out, jnp.float32)))
    return params


def mlp(params, x):
    for W, b in params[:-1]:
        x = jax.nn.elu(x @ W + b)
    W, b = params[-1]
    return x @ W + b


class RunningNorm:
    """Running mean / variance (parallel Welford), clipped normalisation."""

    def __init__(self, dim, clip=5.0):
        self.mean = np.zeros(dim, np.float64)
        self.var = np.ones(dim, np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip).astype(np.float32)

    def state(self):
        return dict(mean=self.mean, var=self.var, count=self.count, clip=self.clip)

    def load(self, s):
        self.mean, self.var, self.count, self.clip = s["mean"], s["var"], s["count"], s["clip"]


@dataclass
class PPOConfig:
    hidden: tuple = (256, 256, 128)
    init_std: float = 0.5
    lr: float = 5e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    value_coef: float = 1.0
    entropy_coef: float = 0.004
    epochs: int = 5
    minibatches: int = 4
    horizon: int = 24
    max_grad_norm: float = 1.0
    desired_kl: float = 0.01


def init_params(key, obs_dim, cobs_dim, act_dim, cfg: PPOConfig):
    ka, kc = jax.random.split(key)
    return dict(
        actor=init_mlp(ka, [obs_dim, *cfg.hidden, act_dim], out_scale=0.01),
        critic=init_mlp(kc, [cobs_dim, *cfg.hidden, 1], out_scale=1.0),
        log_std=jnp.full(act_dim, np.log(cfg.init_std), jnp.float32),
    )


def _gauss_logp(a, mean, log_std):
    return (-0.5 * ((a - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)


@jax.jit
def policy_step(params, obs, cobs, key):
    mean = mlp(params["actor"], obs)
    log_std = params["log_std"]
    a = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
    return a, _gauss_logp(a, mean, log_std), mlp(params["critic"], cobs)[:, 0], mean


@jax.jit
def policy_mean(params, obs):
    return mlp(params["actor"], obs)


@jax.jit
def value_fn(params, cobs):
    return mlp(params["critic"], cobs)[:, 0]


def make_update(cfg: PPOConfig):
    opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                      optax.inject_hyperparams(optax.adam)(learning_rate=cfg.lr))

    def loss_fn(params, b):
        mean = mlp(params["actor"], b["obs"])
        log_std = params["log_std"]
        logp = _gauss_logp(b["act"], mean, log_std)
        ratio = jnp.exp(logp - b["logp"])
        adv = b["adv"]
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
        v = mlp(params["critic"], b["cobs"])[:, 0]
        v_clip = b["val"] + jnp.clip(v - b["val"], -cfg.clip, cfg.clip)
        vl = jnp.maximum((v - b["ret"]) ** 2, (v_clip - b["ret"]) ** 2).mean()
        ent = (log_std + 0.5 * np.log(2 * np.pi * np.e)).sum()
        # KL(old || new) for diagonal Gaussians with shared std
        old_mean, old_std = b["mean"], jnp.exp(b["log_std"])
        new_std = jnp.exp(log_std)
        kl = (jnp.log(new_std / old_std) + (old_std ** 2 + (old_mean - mean) ** 2) / (2 * new_std ** 2) - 0.5).sum(-1).mean()
        loss = pg + cfg.value_coef * vl - cfg.entropy_coef * ent
        return loss, dict(pg=pg, vl=vl, ent=ent, kl=kl)

    @jax.jit
    def update(params, opt_state, batch, lr):
        opt_state[1].hyperparams["learning_rate"] = lr
        (loss, stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params["log_std"] = jnp.clip(params["log_std"], np.log(0.05), np.log(1.5))
        return params, opt_state, stats

    return opt, update


def gae(rew, val, done, last_val, gamma, lam):
    T = rew.shape[0]
    adv = np.zeros_like(rew)
    last = np.zeros_like(last_val)
    for t in reversed(range(T)):
        nv = last_val if t == T - 1 else val[t + 1]
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nv * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + val


def save_checkpoint(path, params, opt_state, obs_norm, cobs_norm, extra):
    blob = dict(params=jax.tree_util.tree_map(np.asarray, params),
                obs_norm=obs_norm.state(), cobs_norm=cobs_norm.state(), extra=extra)
    with open(path, "wb") as f:
        pickle.dump(blob, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def export_policy(path, params, obs_norm, meta: dict):
    """Write a framework-free policy file (numpy arrays only)."""
    arrays = {}
    for i, (W, b) in enumerate(params["actor"]):
        arrays[f"W{i}"] = np.asarray(W, np.float32)
        arrays[f"b{i}"] = np.asarray(b, np.float32)
    arrays["obs_mean"] = obs_norm.mean.astype(np.float32)
    arrays["obs_std"] = np.sqrt(obs_norm.var + 1e-8).astype(np.float32)
    arrays["obs_clip"] = np.float32(obs_norm.clip)
    arrays["n_layers"] = np.int32(len(params["actor"]))
    import json
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez(path, **arrays)
