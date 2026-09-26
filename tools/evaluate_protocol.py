"""Standalone evaluation protocol specification (no model dependency).

Documents the exact evaluation loop used in the paper: conditions, seeds,
episode counts, and the background compositor contract. Intended as a
readable reference; the full evaluation with model loading requires the
training codebase (see paper Algorithm 1 for the model interface).
"""
import argparse

CONDITIONS = {
    "clean": {"video": None, "description": "no video background"},
    "seen":  {"video": "train pool (video0-79)", "background_split": "train"},
    "unseen": {"video": "test pool (video90-99)", "background_split": "test"},
}

EPISODES = {"default": 20, "cup-catch": 50}
ENV_SEED_BASE = 424243
BACKGROUND_SEED_BASE = 1618034
PLANNER_SEED_BASE = 8675300

def parse_args():
    p = argparse.ArgumentParser(description="ISI-WM evaluation protocol")
    p.add_argument("--task", required=True,
                   choices=["acrobot-swingup","cartpole-swingup","cup-catch",
                            "finger-spin","reacher-easy","walker-walk"])
    p.add_argument("--condition", required=True, choices=list(CONDITIONS.keys()))
    p.add_argument("--episodes", type=int, default=None,
                   help="default 20; cup-catch uses 50")
    p.add_argument("--training-seed", type=int, required=True, help="6, 7, or 8")
    return p.parse_args()

def main():
    args = parse_args()
    n = args.episodes or EPISODES.get(args.task, EPISODES["default"])
    cond = CONDITIONS[args.condition]
    print(f"Task: {args.task}")
    print(f"Condition: {args.condition} ({cond['description']})")
    print(f"Episodes: {n}")
    print(f"Training seed: {args.training_seed}")
    print(f"Env seed base: {ENV_SEED_BASE}")
    print(f"Background seed base: {BACKGROUND_SEED_BASE}")
    print(f"Planner seed base: {PLANNER_SEED_BASE}")
    print()
    print("Protocol: for each episode i in range(n):")
    print("  1. env = make_env(task, seed=ENV_SEED_BASE + i)")
    print("  2. if condition != clean: attach video background")
    print(f"     source from {cond.get('video', 'N/A')}, seed=BACKGROUND_SEED_BASE + i")
    print("  3. obs = env.reset()")
    print("  4. for t in range(500):")
    print("       action = agent.act(obs)  # requires trained checkpoint")
    print("       obs, reward, done, _ = env.step(action)")
    print("  5. record episode return")
    print()
    print("Note: agent.act() requires the trained ISI-WM checkpoint;")
    print("the training implementation is described in the paper (Algorithm 1).")

if __name__ == "__main__":
    main()
