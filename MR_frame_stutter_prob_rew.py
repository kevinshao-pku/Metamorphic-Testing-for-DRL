import numpy as np
import argparse
import os
import random
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

# Define paths relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DQN_MODEL_PATH = os.path.join(SCRIPT_DIR, "dqn-MsPacmanNoFrameskip-v4.zip")
PPO_MODEL_PATH = os.path.join(SCRIPT_DIR, "../ppo-MsPacmanNoFrameskip-v4.zip")


def play_game_session(model, num_episodes, seed, stutter_probability=0.0):
    """
    Plays game sessions with probabilistic frame stutter.
    A "stutter" is a persistent error where a missed frame (by duplicating the previous one)
    is propagated into all subsequent observations.
    """
    vec_env = make_atari_env("MsPacmanNoFrameskip-v4", n_envs=1, seed=seed)
    env = VecFrameStack(vec_env, n_stack=4)

    episode_rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        current_episode_reward = 0
        last_obs = None

        while not done:
            final_obs = np.array(obs)

            if last_obs is not None:
                final_obs[..., :-1] = last_obs[..., 1:]

            if stutter_probability > 0 and np.random.rand() < stutter_probability:
                final_obs[..., -1] = final_obs[..., -2]

            last_obs = final_obs.copy()

            action, _states = model.predict(final_obs, deterministic=True)
            obs, reward, terminated, info = env.step(action)
            done = terminated[0]
            current_episode_reward += reward[0]
        
        episode_rewards.append(current_episode_reward)

    env.close()
    return episode_rewards


def main():
    parser = argparse.ArgumentParser(
        description="Metamorphic test for performance change with probabilistic frame stutter.")
    parser.add_argument("--model", type=str, required=True, choices=["dqn", "ppo"], help="Model to test (dqn or ppo)")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes for each run.")
    parser.add_argument("--threshold", type=float, default=15.0,
                        help="Allowed percentage performance drop before the test fails.")
    parser.add_argument("--stutter_probability", type=float, default=0.1,
                        help="Probability of a frame stutter.")
    parser.add_argument("--seed", type=int, required=True, help='Random seed for the environment.')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.model == "dqn":
        model_path = DQN_MODEL_PATH
        model_class = DQN
    else:
        model_path = PPO_MODEL_PATH
        model_class = PPO

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    model = model_class.load(model_path, device="cpu")

    print(f"\nRunning original case (no stutter)...")
    original_rewards = play_game_session(model, num_episodes=args.num_episodes, seed=args.seed, stutter_probability=0.0)
    original_reward_total = sum(original_rewards)

    print(f"\nRunning mutated case (stutter probability: {args.stutter_probability})...")
    mutated_rewards = play_game_session(model, num_episodes=args.num_episodes, seed=args.seed,
                                        stutter_probability=args.stutter_probability)
    mutated_reward_total = sum(mutated_rewards)

    if original_reward_total == 0:
        performance_change_percent = 0 if mutated_reward_total == 0 else float('-inf')
    else:
        performance_change_percent = 100 * (mutated_reward_total - original_reward_total) / abs(original_reward_total)

    performance_drop_abs = abs(performance_change_percent)
    result = "PASSED" if performance_drop_abs <= args.threshold else "FAILED"

    reward_change_str = f"{performance_change_percent:+0.2f}%"
    threshold_str = f"{args.threshold:.2f}%"

    print("\n" + "-" * 50)
    print("        FRAME STUTTER REWARD-BASED RESULTS")
    print(f"Seed: {args.seed}")
    print(f"Model:                 {args.model.upper():>8}")
    print(f"Stutter Probability (p):{args.stutter_probability:>8.2f}")
    print(f"Original Reward Total: {original_reward_total:>8.0f}")
    print(f"Mutated Reward Total:  {mutated_reward_total:>8.0f}")
    print(f"Performance Change:    {reward_change_str:>8}")
    print(f"Threshold:             {threshold_str:>8}")
    print(f"Result:                {result:>8}")
    print("-" * 50)


if __name__ == "__main__":
    main()