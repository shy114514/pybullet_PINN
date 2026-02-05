#!/usr/bin/env python3
"""
Unified evaluation script for push task RL.

Supports both PyBullet and Isaac Lab backends with Stable Baselines 3.

Usage:
    # Evaluate with PyBullet
    python evaluate.py --backend pybullet --model-path ./models/xxx/

    # Evaluate with video recording
    python evaluate.py --backend pybullet --model-path ./models/xxx/ --save-video

    # Evaluate multiple checkpoints
    python evaluate.py --model-path ./models/xxx/checkpoints/
"""

import os
import sys
import argparse
import glob
import re

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate RL agent for Push Task")

    # Backend selection
    parser.add_argument("--backend", type=str, default="pybullet",
                        choices=["pybullet", "isaac_lab"],
                        help="Environment backend")

    # Model path
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model file or directory")

    # Evaluation settings
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes to evaluate")
    parser.add_argument("--obs-type", type=str, default="state",
                        choices=["state", "image"],
                        help="Observation type")

    # Output options
    parser.add_argument("--save-video", action="store_true",
                        help="Save video recordings")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file")

    return parser.parse_args()


def get_steps(filename: str) -> int:
    """Extract step count from checkpoint filename."""
    match = re.search(r'_(\d+)_steps', filename)
    return int(match.group(1)) if match else 0


def find_vecnormalize_file(model_path: str) -> str:
    """Find VecNormalize stats file."""
    base_path = model_path.replace('.zip', '')

    possible_paths = [
        base_path + '.pkl',
        base_path + '_vecnormalize.pkl',
        os.path.join(os.path.dirname(model_path), 'vecnormalize.pkl'),
    ]

    model_dir = os.path.dirname(model_path)
    if model_dir:
        pkl_files = glob.glob(os.path.join(model_dir, '*.pkl'))
        model_steps = get_steps(model_path)
        for pkl in pkl_files:
            if str(model_steps) in pkl:
                possible_paths.insert(0, pkl)

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None


def get_model_path(args) -> str:
    """Determine model path."""
    if args.model_path:
        return args.model_path

    models_dir = "./models"
    if not os.path.exists(models_dir):
        print(f"Models directory {models_dir} not found.")
        return None

    suffix = f"_{args.backend}_{args.obs_type}"
    model_dirs = [d for d in os.listdir(models_dir)
                  if os.path.isdir(os.path.join(models_dir, d)) and suffix in d]

    if not model_dirs:
        print(f"No model directories found with suffix '{suffix}'")
        return None

    model_dirs.sort(reverse=True)
    latest_dir = os.path.join(models_dir, model_dirs[0])

    zip_files = glob.glob(os.path.join(latest_dir, "*.zip"))

    if zip_files:
        final_models = [f for f in zip_files if "_steps" not in os.path.basename(f)]
        if final_models:
            print(f"Auto-detected model: {final_models[0]}")
            return final_models[0]

    print(f"No model files found in {latest_dir}")
    return None


def get_models_to_evaluate(model_path: str) -> list:
    """Get list of models to evaluate."""
    if os.path.isdir(model_path):
        checkpoint_files = glob.glob(os.path.join(model_path, "*.zip"))
        checkpoint_files.sort(key=get_steps)

        if not checkpoint_files:
            print(f"No checkpoints found in {model_path}")
            return []

        return checkpoint_files

    if not model_path.endswith('.zip'):
        model_path += ".zip"

    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return []

    return [model_path]


def create_eval_env(args, cfg, vecnorm_path=None):
    """Create evaluation environment."""
    from envs import make_env
    from gymnasium.wrappers import TimeLimit

    env = make_env(
        backend=args.backend,
        cfg=cfg,
        render_mode="rgb_array",
        obs_type=args.obs_type,
    )

    # Wrap in DummyVecEnv for VecNormalize compatibility
    if not hasattr(env, 'num_envs'):
        env = DummyVecEnv([lambda: env])

    # Load VecNormalize for state mode
    if args.obs_type == "state" and vecnorm_path:
        print(f"Loading VecNormalize: {vecnorm_path}")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    return env


def plot_trajectory(ee_traj, obj_traj, target_pos, episode_num, output_dir, success):
    """Plot trajectory visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(ee_traj[:, 0], ee_traj[:, 1], 'b-', linewidth=1.5,
            label='EE Trajectory', alpha=0.7)
    ax.scatter(ee_traj[0, 0], ee_traj[0, 1], c='blue', marker='o',
               s=100, zorder=5, label='EE Start')
    ax.scatter(ee_traj[-1, 0], ee_traj[-1, 1], c='blue', marker='s',
               s=100, zorder=5, label='EE End')

    ax.plot(obj_traj[:, 0], obj_traj[:, 1], 'r-', linewidth=2,
            label='Object Trajectory', alpha=0.7)
    ax.scatter(obj_traj[0, 0], obj_traj[0, 1], c='red', marker='o',
               s=150, zorder=5, label='Object Start')
    ax.scatter(obj_traj[-1, 0], obj_traj[-1, 1], c='red', marker='s',
               s=150, zorder=5, label='Object End')

    ax.scatter(target_pos[0], target_pos[1], c='green', marker='*',
               s=300, zorder=5, label='Target')
    circle = plt.Circle((target_pos[0], target_pos[1]), 0.05,
                         color='green', fill=False, linestyle='--',
                         linewidth=2, label='Target Zone')
    ax.add_patch(circle)

    final_dist = np.linalg.norm(obj_traj[-1] - np.array(target_pos))

    status = "SUCCESS" if success else "FAILED"
    status_color = "green" if success else "red"
    ax.set_title(f'Episode {episode_num + 1} - {status}\n'
                 f'Final Distance: {final_dist:.3f}m',
                 fontsize=14, color=status_color)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig_path = os.path.join(plots_dir, f"episode_{episode_num + 1}_trajectory.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return fig_path


def save_video(frames, episode_num, output_dir, success, fps=10):
    """Save video recording."""
    try:
        import imageio
    except ImportError:
        print("Warning: imageio not installed, skipping video save")
        return None

    videos_dir = os.path.join(output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    video_path = os.path.join(videos_dir, f"episode_{episode_num + 1}_video.mp4")

    writer = imageio.get_writer(video_path, fps=fps, codec='libx264',
                                pixelformat='yuv420p', quality=7)

    for frame in frames:
        writer.append_data(frame)

    writer.close()

    return video_path


def run_episode(env, model, episode_num, output_dir, save_video_flag, cfg):
    """Run a single evaluation episode."""
    seed = 42 + episode_num
    np.random.seed(seed)

    obs = env.reset()

    done = False
    step = 0
    episode_reward = 0
    max_steps = cfg.max_episode_steps

    ee_trajectory = []
    obj_trajectory = []
    target_pos_fixed = None
    frames = [] if save_video_flag else None

    # Get base environment for coordinate extraction
    if hasattr(env, 'envs'):
        base_env = env.envs[0]
        while hasattr(base_env, 'env'):
            base_env = base_env.env
    else:
        base_env = env

    while not done and step < max_steps:
        # Collect positions
        try:
            ee_pos = base_env.get_ee_position()[:2]
            obj_pos, _ = base_env.get_object_pose()
            target_pos, _ = base_env.get_target_pose()

            ee_trajectory.append(ee_pos)
            obj_trajectory.append(obj_pos[:2])
            if target_pos_fixed is None:
                target_pos_fixed = target_pos[:2]
        except (AttributeError, NotImplementedError):
            pass

        # Capture frame
        if save_video_flag:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        # Take action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        episode_reward += reward[0] if hasattr(reward, '__len__') else reward
        done = dones[0] if hasattr(dones, '__len__') else dones

        step += 1

    # Convert trajectories
    ee_trajectory = np.array(ee_trajectory) if ee_trajectory else None
    obj_trajectory = np.array(obj_trajectory) if obj_trajectory else None

    success = done and step < max_steps

    # Save outputs
    if ee_trajectory is not None and len(ee_trajectory) > 0:
        plot_path = plot_trajectory(
            ee_trajectory, obj_trajectory, target_pos_fixed,
            episode_num, output_dir, success
        )
        print(f"  Saved plot: {plot_path}")

    if save_video_flag and frames:
        video_path = save_video(frames, episode_num, output_dir, success)
        if video_path:
            print(f"  Saved video: {video_path}")

    return success, episode_reward, step


def evaluate_model(model_file, args, cfg, output_dir):
    """Evaluate a single model."""
    model_name = os.path.splitext(os.path.basename(model_file))[0]
    step_count = get_steps(model_file)

    print(f"\nEvaluating: {model_file}")

    vecnorm_path = find_vecnormalize_file(model_file)
    env = create_eval_env(args, cfg, vecnorm_path)

    try:
        model = PPO.load(model_file, device="cpu")
    except Exception as e:
        print(f"Failed to load model: {e}")
        env.close()
        return None

    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    success_count = 0
    total_rewards = []
    total_steps = []

    for episode in range(args.episodes):
        print(f"  Episode {episode + 1}/{args.episodes}")

        success, reward, steps = run_episode(
            env, model, episode, model_output_dir, args.save_video, cfg
        )

        total_rewards.append(reward)
        total_steps.append(steps)

        if success:
            success_count += 1
            print(f"    SUCCESS - Reward: {reward:.2f}, Steps: {steps}")
        else:
            print(f"    FAILED - Reward: {reward:.2f}, Steps: {steps}")

    env.close()

    success_rate = (success_count / args.episodes) * 100
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)

    print(f"\n  Summary: Success Rate: {success_rate:.1f}%, "
          f"Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.1f}")

    return {
        "name": model_name,
        "steps": step_count,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
    }


def print_summary(results):
    """Print evaluation summary."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    sorted_keys = sorted(results.keys(),
                         key=lambda k: results[k]["steps"] if results[k]["steps"] > 0 else 0)

    print(f"{'Model':<40} | {'Steps':<10} | {'Success':<10} | {'Reward':<10}")
    print("-" * 80)

    for name in sorted_keys:
        r = results[name]
        print(f"{name:<40} | {r['steps']:<10} | {r['success_rate']:>6.1f}%   | {r['avg_reward']:>8.2f}")


def evaluate(args):
    """Main evaluation function."""
    from envs import get_available_backends
    from envs.factory import load_config_from_yaml
    from envs.base import BasePushEnvConfig

    available_backends = get_available_backends()
    if args.backend not in available_backends:
        print(f"Backend '{args.backend}' not available.")
        print(f"Available: {available_backends}")
        return

    model_path = get_model_path(args)
    if not model_path:
        return

    models = get_models_to_evaluate(model_path)
    if not models:
        return

    # Load config
    if args.config:
        cfg = load_config_from_yaml(args.config)
        print(f"Loaded config from {args.config}")
    else:
        # Try to load from model directory
        config_dir = os.path.dirname(model_path) if os.path.isfile(model_path) else model_path
        config_path = os.path.join(config_dir, "config_used.yaml")
        flag = False
        for _ in range(4):
            if os.path.exists(config_path):
                cfg = load_config_from_yaml(config_path)
                print(f"Loaded config from {config_path}")
                flag = True
                break
            config_dir = os.path.dirname(config_dir)
            config_path = os.path.join(config_dir, "config_used.yaml")
        if not flag:
            cfg = BasePushEnvConfig()
            print("Using default environment config.")

    # Output directory
    model_dir = os.path.dirname(model_path) if os.path.isfile(model_path) else model_path
    output_dir = os.path.join(model_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Backend: {args.backend}")
    print(f"Results: {output_dir}")

    results = {}
    for model_file in models:
        result = evaluate_model(model_file, args, cfg, output_dir)
        if result:
            results[result["name"]] = result

    if results:
        print_summary(results)


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
