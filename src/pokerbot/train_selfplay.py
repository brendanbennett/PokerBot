"""Self-play PPO training for Ponk.

Run:
    python -m pokerbot.train_selfplay

Every seat in the env is driven by the latest policy. Each hand is played in a
fresh env (dealer randomized) so a busted player never blocks training. The
sole reward signal is each player's normalized money_diff at hand end, credited
to that player's last action — value/advantage propagate it back via GAE.
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from pokerbot.env import Ponk, PonkConfig
from pokerbot.ppo import ActorCritic, compute_gae, ppo_update


def make_hand_env(config: PonkConfig) -> Ponk:
    env = Ponk(config)
    env.dealer = random.randint(0, config.num_players - 1)
    return env


def collect_rollout(model: ActorCritic, config: PonkConfig, hands: int, device: torch.device):
    model.eval()
    transitions: list[dict] = []
    hand_returns: list[float] = []
    n_players = config.num_players

    for _ in range(hands):
        env = make_hand_env(config)
        obs = env.reset()
        per_player: dict[int, list[dict]] = {i: [] for i in range(n_players)}

        while True:
            cur = env.turn
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a_type, r_frac, logp, value = model.act(obs_t)
            action = (int(a_type.item()), float(r_frac.item()))
            next_obs, _r, winner = env.step(action)
            per_player[cur].append({
                "obs": obs.astype(np.float32),
                "a_type": int(a_type.item()),
                "r_frac": float(r_frac.item()),
                "logp": float(logp.item()),
                "value": float(value.item()),
                "reward": 0.0,
                "done": False,
            })
            obs = next_obs
            if winner != -1:
                for p_idx in range(n_players):
                    if not per_player[p_idx]:
                        continue
                    money_diff = env.players[p_idx].get_money_diff()
                    norm = money_diff / config.starting_money
                    per_player[p_idx][-1]["reward"] = norm
                    per_player[p_idx][-1]["done"] = True
                    hand_returns.append(norm)
                break

        for p_idx in range(n_players):
            traj = per_player[p_idx]
            if not traj:
                continue
            advs, rets = compute_gae(traj, gamma=0.99, lam=0.95)
            for t, a, r in zip(traj, advs, rets):
                t["adv"] = a
                t["ret"] = r
                transitions.append(t)

    return transitions, hand_returns


def normalize_advantages(transitions: list[dict]) -> None:
    advs = np.array([t["adv"] for t in transitions], dtype=np.float32)
    mean, std = advs.mean(), advs.std()
    advs = (advs - mean) / (std + 1e-8)
    for i, t in enumerate(transitions):
        t["adv"] = float(advs[i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--starting-money", type=int, default=100)
    parser.add_argument("--small-blind", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--hands-per-rollout", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt-every", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = PonkConfig(
        num_players=args.num_players,
        small_blind=args.small_blind,
        starting_money=args.starting_money,
    )
    obs_dim = 104 + 3 * args.num_players
    model = ActorCritic(obs_dim, hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}  obs_dim={obs_dim}  params={sum(p.numel() for p in model.parameters())}")

    for it in range(1, args.iterations + 1):
        t0 = time.time()
        transitions, hand_returns = collect_rollout(model, config, args.hands_per_rollout, device)
        if not transitions:
            print(f"iter={it:04d}  no transitions collected — skipping")
            continue
        normalize_advantages(transitions)

        model.train()
        losses = ppo_update(
            model, optimizer, transitions,
            epochs=args.ppo_epochs, batch_size=args.batch_size,
            clip=args.clip, entropy_coef=args.entropy_coef, value_coef=args.value_coef,
            device=device,
        )
        elapsed = time.time() - t0
        mean_ret = float(np.mean(hand_returns)) if hand_returns else 0.0
        std_ret = float(np.std(hand_returns)) if hand_returns else 0.0
        print(
            f"iter={it:04d}  n={len(transitions):5d}  "
            f"ret={mean_ret:+.3f}±{std_ret:.3f}  "
            f"pol={losses['policy']:+.4f}  val={losses['value']:.4f}  ent={losses['entropy']:.3f}  "
            f"({elapsed:.1f}s)"
        )

        if it % args.ckpt_every == 0:
            path = ckpt_dir / f"ppo_iter{it:04d}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": it,
                "args": vars(args),
            }, path)
            print(f"  saved {path}")


if __name__ == "__main__":
    main()
