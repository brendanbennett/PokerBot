"""PPO actor-critic and update step for Ponk self-play training.

Action space matches Ponk.step's expected (int, float) tuple:
  action_type in {0=call, 1=raise, 2=fold}, raise_frac in [0, 1].
The raise fraction is sampled from a Beta head and only contributes to the
log-prob/entropy when action_type == raise. The env may clip the sampled
fraction to honor min_raise / all-in, so `act` exposes the Beta params and
the categorical log-prob separately; the caller computes the combined logp
against the *executed* fraction (see train_selfplay.collect_rollout).
"""
from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical

CALL, RAISE, FOLD = 0, 1, 2


class Transition(TypedDict):
    obs: np.ndarray
    a_type: int
    r_frac: float
    logp: float
    value: float
    reward: float
    done: bool
    adv: float
    ret: float


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.action_logits = nn.Linear(hidden, 3)
        self.raise_alpha = nn.Linear(hidden, 1)
        self.raise_beta = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def _heads(self, obs: torch.Tensor):
        z = self.trunk(obs)
        logits = self.action_logits(z)
        # +1.0 keeps alpha,beta > 1 so the Beta is unimodal — easier to optimize.
        alpha = F.softplus(self.raise_alpha(z)).squeeze(-1) + 1.0
        beta = F.softplus(self.raise_beta(z)).squeeze(-1) + 1.0
        value = self.value_head(z).squeeze(-1)
        return logits, alpha, beta, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        logits, alpha, beta, value = self._heads(obs)
        cat = Categorical(logits=logits)
        a_type = cat.sample()
        cat_lp = cat.log_prob(a_type)
        beta_dist = Beta(alpha, beta)
        r_frac = beta_dist.sample().clamp(1e-4, 1 - 1e-4)
        # Beta params returned so the caller can recompute log_prob against
        # whatever fraction the env actually executed (env may clip to
        # min_raise / all-in).
        return a_type, r_frac, value, alpha, beta, cat_lp

    def evaluate(self, obs: torch.Tensor, a_type: torch.Tensor, r_frac: torch.Tensor):
        logits, alpha, beta, value = self._heads(obs)
        cat = Categorical(logits=logits)
        cat_lp = cat.log_prob(a_type)
        cat_ent = cat.entropy()
        beta_dist = Beta(alpha, beta)
        r_frac_safe = r_frac.clamp(1e-4, 1 - 1e-4)
        beta_lp = beta_dist.log_prob(r_frac_safe)
        beta_ent = beta_dist.entropy()
        is_raise = (a_type == RAISE).float()
        logp = cat_lp + is_raise * beta_lp
        entropy = cat_ent + is_raise * beta_ent
        return logp, entropy, value


def compute_gae(traj: list[dict], gamma: float, lam: float):
    """Per-player GAE. `traj` is one player's transitions across a single hand,
    in turn order. The last entry has done=True and the terminal reward."""
    n = len(traj)
    advantages = [0.0] * n
    returns = [0.0] * n
    gae = 0.0
    for t in reversed(range(n)):
        next_value = 0.0 if t == n - 1 else traj[t + 1]["value"]
        next_nonterminal = 0.0 if traj[t]["done"] else 1.0
        delta = traj[t]["reward"] + gamma * next_value * next_nonterminal - traj[t]["value"]
        gae = delta + gamma * lam * next_nonterminal * gae
        advantages[t] = gae
        returns[t] = gae + traj[t]["value"]
    return advantages, returns


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: list[dict],
    *,
    epochs: int,
    batch_size: int,
    clip: float,
    entropy_coef: float,
    value_coef: float,
    device: torch.device,
    max_grad_norm: float = 0.5,
) -> dict:
    obs = torch.as_tensor(np.stack([t["obs"] for t in transitions]), dtype=torch.float32, device=device)
    a_types = torch.as_tensor([t["a_type"] for t in transitions], dtype=torch.long, device=device)
    r_fracs = torch.as_tensor([t["r_frac"] for t in transitions], dtype=torch.float32, device=device)
    old_logps = torch.as_tensor([t["logp"] for t in transitions], dtype=torch.float32, device=device)
    advs = torch.as_tensor([t["adv"] for t in transitions], dtype=torch.float32, device=device)
    rets = torch.as_tensor([t["ret"] for t in transitions], dtype=torch.float32, device=device)

    n = len(transitions)
    idx = np.arange(n)
    tot_pol = tot_val = tot_ent = 0.0
    n_batches = 0
    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            b_t = torch.as_tensor(b, dtype=torch.long, device=device)
            new_logp, ent, value = model.evaluate(obs[b_t], a_types[b_t], r_fracs[b_t])
            ratio = (new_logp - old_logps[b_t]).exp()
            adv_b = advs[b_t]
            s1 = ratio * adv_b
            s2 = ratio.clamp(1 - clip, 1 + clip) * adv_b
            pol_loss = -torch.min(s1, s2).mean()
            val_loss = F.mse_loss(value, rets[b_t])
            ent_mean = ent.mean()
            loss = pol_loss + value_coef * val_loss - entropy_coef * ent_mean
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            tot_pol += float(pol_loss.item())
            tot_val += float(val_loss.item())
            tot_ent += float(ent_mean.item())
            n_batches += 1
    return {
        "policy": tot_pol / max(n_batches, 1),
        "value": tot_val / max(n_batches, 1),
        "entropy": tot_ent / max(n_batches, 1),
    }
