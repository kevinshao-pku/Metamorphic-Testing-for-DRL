import gymnasium as gym
import numpy as np
import torch
import random
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import argparse
import os

# Define paths relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DQN_MODEL_PATH = os.path.join(SCRIPT_DIR, "dqn-MsPacmanNoFrameskip-v4.zip")
PPO_MODEL_PATH = os.path.join(SCRIPT_DIR, "ppo-MsPacmanNoFrameskip-v4.zip")


def black_out_score(obs, score_rows=10):
    """Blacks out the score area at the bottom of the screen."""
    obs_copy = np.copy(obs)
    # The score is at the bottom, so we black out the last few rows.
    obs_copy[..., -score_rows:, :, :] = 0
    return obs_copy


def test_score_invariance(model, env_id="MsPacmanNoFrameskip-v4", seed=42, threshold=15.0, score_rows=10):
    """
    Runs a single game episode to test if the agent's action is invariant
    to the score area being blacked out.
    """
    # Set all relevant random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    vec_env = make_atari_env(env_id, n_envs=1, seed=seed)
    env = VecFrameStack(vec_env, n_stack=4)

    obs = env.reset()
    terminated = [False]
    total_steps = 0
    divergent_steps = 0

    while not terminated[0]:
        action_original, _ = model.predict(obs, deterministic=True)

        mutated_obs = black_out_score(obs, score_rows)
        action_mutated, _ = model.predict(mutated_obs, deterministic=True)

        if action_original[0] != action_mutated[0]:
            divergent_steps += 1

        obs, _, terminated, _ = env.step(action_original)
        total_steps += 1

    env.close()

    divergence_pct = (divergent_steps / total_steps) * 100 if total_steps > 0 else 0
    result = "PASSED" if divergence_pct <= threshold else "FAILED"
    divergence_rate_str = f"{divergence_pct:.2f}%"
    threshold_str = f"{threshold:.2f}%"

    # --- USE CLEAR, ACTION-BASED LABELS ---
    print("\n" + "=" * 40)
    print(" SCORE BLACKOUT TEST RESULTS (ACTION)")
    print(f"Seed: {seed}")
    print(f"Model:                 {model.__class__.__name__:>8}")
    print(f"Score Rows Blacked Out:{score_rows:>8}")
    print(f"Total Steps:           {total_steps:>8}")
    print(f"Divergent Steps:       {divergent_steps:>8}")
    print(f"Divergence Rate:       {divergence_rate_str:>8}")
    print(f"Threshold:             {threshold_str:>8}")
    print(f"Result:                {result:>8}")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(description="Test model's invariance to score blackout in Ms. Pac-Man.")
    parser.add_argument("--model", type=str, required=True, choices=["dqn", "ppo"], help="Model to test (dqn or ppo)")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to run (should be 1).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the environment.")
    parser.add_argument("--threshold", type=float, default=15.0, help="Threshold for passing the test.")
    parser.add_argument("--score_rows", type=int, default=10, help="Number of rows to black out from the bottom.")
    args = parser.parse_args()

    if args.model == "dqn":
        model_path = DQN_MODEL_PATH
        model_class = DQN
    else: # ppo
        model_path = PPO_MODEL_PATH
        model_class = PPO

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    model = model_class.load(model_path, device="cpu")

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)

    test_score_invariance(model, seed=seed, threshold=args.threshold, score_rows=args.score_rows)


if __name__ == '__main__':
    main()