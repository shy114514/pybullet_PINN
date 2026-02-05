#!/usr/bin/env python3
"""
Unified training script for push task RL.

Supports both PyBullet and Isaac Lab backends with Stable Baselines 3.

Usage:
    # Train with PyBullet (CPU)
    python train.py --backend pybullet --timesteps 1000000

    # Train with Isaac Lab (GPU)
    python train.py --backend isaac_lab --timesteps 1000000 --n-envs 1024

    # Resume from checkpoint
    python train.py --backend pybullet --load-checkpoint ./models/xxx/

    # Use custom config
    python train.py --config config-s2.yaml --timesteps 3000000
"""

import os
import sys
import argparse
import shutil
from datetime import datetime
import time
import glob
import re

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RL agent for Push Task")

    # Backend selection
    parser.add_argument("--backend", type=str, default="pybullet",
                        choices=["pybullet", "isaac_lab"],
                        help="Environment backend: 'pybullet' or 'isaac_lab'")

    # Observation type
    parser.add_argument("--obs-type", type=str, default="state",
                        choices=["state", "image"],
                        help="Observation type: 'state' or 'image'")

    # Training parameters
    parser.add_argument("--timesteps", type=int, default=1000000,
                        help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=16,
                        help="Number of parallel environments")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cpu', 'cuda', or 'auto'")

    # Checkpoint
    parser.add_argument("--save-freq", type=int, default=None,
                        help="Checkpoint save frequency in steps")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Path to checkpoint file or directory")

    # Config
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (default: config.yaml)")

    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: auto-generated)")
    
    # Test mode
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode (no training)")

    return parser.parse_args()


class ProgressBarCallback(BaseCallback):
    """Progress bar callback with training metrics."""

    def __init__(self, total_timesteps: int, update_freq: int = 32768, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.update_freq = update_freq
        self.start_time = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_update_step = 0

    def _on_training_start(self):
        self.start_time = time.time()
        self.initial_timesteps = self.model.num_timesteps
        self.target_timesteps = self.initial_timesteps + self.total_timesteps
        self.last_update_step = self.initial_timesteps
        print("\n" * 6)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

        if self.num_timesteps - self.last_update_step >= self.update_freq:
            self._update_progress_bar()
            self.last_update_step = self.num_timesteps

        return True

    def _update_progress_bar(self):
        current_steps = self.num_timesteps
        elapsed_time = time.time() - self.start_time
        steps_done = current_steps - self.initial_timesteps

        progress = min(steps_done / self.total_timesteps, 1.0)
        progress_percent = progress * 100

        if steps_done > 0:
            steps_per_sec = steps_done / elapsed_time
            remaining_steps = self.total_timesteps - steps_done
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            eta_str = self._format_time(eta_seconds)
            fps_str = f"{steps_per_sec:.0f}"
        else:
            eta_str = "N/A"
            fps_str = "N/A"

        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-100:]
            recent_lengths = self.episode_lengths[-100:]
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(recent_lengths)
            reward_str = f"{avg_reward:.1f}"
            length_str = f"{avg_length:.0f}"
        else:
            reward_str = "N/A"
            length_str = "N/A"

        bar_length = 40
        filled_length = int(bar_length * progress)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)

        elapsed_str = self._format_time(elapsed_time)

        sys.stdout.write("\033[6A\033[J")

        status_lines = [
            f"╔{'═' * 58}╗",
            f"║  [{bar}] {progress_percent:5.1f}%  ║",
            f"╠{'═' * 58}╣",
            f"║  Steps: {steps_done:>10,} / {self.total_timesteps:<10,} (Total: {current_steps:,}) ║",
            f"║  FPS: {fps_str:>8}  │  Elapsed: {elapsed_str}  │  ETA: {eta_str}  ║",
            f"║  Reward: {reward_str:>8}  │  EpLen: {length_str:>8}  │  Episodes: {len(self.episode_rewards):<6} ║",
            f"╚{'═' * 58}╝",
        ]

        sys.stdout.write("\n".join(status_lines) + "\n")
        sys.stdout.flush()

    def _format_time(self, seconds: float) -> str:
        if seconds < 0 or seconds > 86400 * 30:
            return "N/A"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _on_training_end(self):
        print()
        total_time = time.time() - self.start_time
        print(f"Training completed in {self._format_time(total_time)}")
        if len(self.episode_rewards) > 0:
            print(f"Final avg reward (last 100 ep): {np.mean(self.episode_rewards[-100:]):.2f}")


class SaveVecNormalizeCallback(BaseCallback):
    """Callback to save VecNormalize statistics."""

    def __init__(self, save_freq: int, save_path: str, name_prefix: str, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            if hasattr(self.training_env, 'save'):
                self.training_env.save(path)
        return True


def resolve_checkpoint_path(checkpoint_path: str):
    """Resolve checkpoint path to (model_path, vecnorm_path)."""
    if checkpoint_path is None:
        return None, None

    if os.path.isdir(checkpoint_path):
        zip_files = glob.glob(os.path.join(checkpoint_path, "*.zip"))
        if not zip_files:
            raise ValueError(f"No .zip model files found in {checkpoint_path}")

        final_models = [f for f in zip_files if "_steps" not in os.path.basename(f)]
        if final_models:
            model_path = final_models[0]
        else:
            def get_steps(f):
                match = re.search(r'_(\d+)_steps', f)
                return int(match.group(1)) if match else 0
            zip_files.sort(key=get_steps, reverse=True)
            model_path = zip_files[0]

        pkl_files = glob.glob(os.path.join(checkpoint_path, "*.pkl"))
        if pkl_files:
            model_base = os.path.splitext(os.path.basename(model_path))[0]
            matching_pkl = [p for p in pkl_files if model_base in p or "vecnormalize" in p.lower()]
            vecnorm_path = matching_pkl[0] if matching_pkl else pkl_files[0]
        else:
            vecnorm_path = None

        return model_path, vecnorm_path

    if not checkpoint_path.endswith('.zip'):
        checkpoint_path += '.zip'

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Model file not found: {checkpoint_path}")

    base_path = checkpoint_path.replace('.zip', '')
    possible_pkl_paths = [
        base_path + "_vecnormalize.pkl",
        base_path + ".pkl",
        os.path.join(os.path.dirname(checkpoint_path), "vecnormalize.pkl"),
    ]

    vecnorm_path = None
    for pkl_path in possible_pkl_paths:
        if os.path.exists(pkl_path):
            vecnorm_path = pkl_path
            break

    return checkpoint_path, vecnorm_path


def get_device(args):
    """Determine device to use."""
    if args.device == "auto":
        if args.backend == "isaac_lab":
            return "cuda"
        else:
            return "cpu"
    return args.device


def create_env(args, cfg, vecnorm_path=None):
    """Create training environment."""
    from envs import make_vec_env

    device = get_device(args)

    env = make_vec_env(
        backend=args.backend,
        cfg=cfg,
        n_envs=args.n_envs,
        obs_type=args.obs_type,
        device=device,
        render_mode="human" if args.test else None,
    )

    # For PyBullet with state obs, load VecNormalize if available
    if args.backend == "pybullet" and args.obs_type == "state":
        if vecnorm_path and os.path.exists(vecnorm_path):
            print(f"Loading VecNormalize stats from {vecnorm_path}")
            env = VecNormalize.load(vecnorm_path, env.venv if hasattr(env, 'venv') else env)
            env.training = True
            env.norm_reward = False

    return env


def create_callbacks(args, config, env, n_envs):
    """Create training callbacks."""
    callbacks = []

    progress_callback = ProgressBarCallback(
        total_timesteps=args.timesteps,
        update_freq=32768,
        verbose=1
    )
    callbacks.append(progress_callback)

    if args.save_freq is not None:
        checkpoint_path = config["checkpoint_path"]
        os.makedirs(checkpoint_path, exist_ok=True)

        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.save_freq // n_envs, 1),
            save_path=checkpoint_path,
            name_prefix=config["model_name"],
        )
        callbacks.append(checkpoint_callback)

        if args.obs_type == "state" and isinstance(env, VecNormalize):
            norm_callback = SaveVecNormalizeCallback(
                save_freq=max(args.save_freq // n_envs, 1),
                save_path=checkpoint_path,
                name_prefix=config["model_name"],
            )
            callbacks.append(norm_callback)

    return callbacks


def get_training_config(args, cfg):
    """Get training configuration."""
    if args.output_dir:
        base_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = f"./models/{timestamp}_{args.backend}_{args.obs_type}"

    config = {
        "base_dir": base_dir,
        "checkpoint_path": f"{base_dir}/checkpoints/",
        "tensorboard_log": f"{base_dir}/tensorboard/",
        "model_save_path": f"{base_dir}/ppo_push_robot",
        "model_name": "ppo_push_robot",
    }

    if args.obs_type == "image":
        config["policy_type"] = "CnnPolicy"
        config["ppo_kwargs"] = {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
        }
    else:
        config["policy_type"] = "MlpPolicy"
        config["ppo_kwargs"] = {
            "learning_rate": 0.0001,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "policy_kwargs": dict(net_arch=dict(
                pi=[256, 256, 128],
                vf=[256, 256, 128]
            )),
        }

    return config


def train(args):
    """Main training function."""
    from envs import get_available_backends
    from envs.factory import load_config_from_yaml
    from envs.base import BasePushEnvConfig

    # Check backend availability
    available_backends = get_available_backends()
    if args.backend not in available_backends:
        print(f"Error: Backend '{args.backend}' is not available.")
        print(f"Available backends: {available_backends}")
        return

    print(f"Training with backend: {args.backend}")
    print(f"Observation type: {args.obs_type}")
    print(f"Number of environments: {args.n_envs}")

    # Load configuration
    if args.config:
        cfg = load_config_from_yaml(args.config)
        print(f"Using config file: {args.config}")
    else:
        cfg = BasePushEnvConfig()
        print("Using default configuration")

    # Get training config
    config = get_training_config(args, cfg)

    # Create output directory and save config before training
    os.makedirs(config["base_dir"], exist_ok=True)
    if args.config:
        config_dst = os.path.join(config["base_dir"], "config_used.yaml")
        shutil.copy(args.config, config_dst)
        print(f"Config saved: {config_dst}")

    # Resolve checkpoint
    model_path, vecnorm_path = resolve_checkpoint_path(args.load_checkpoint)

    # Create environment
    env = create_env(args, cfg, vecnorm_path)

    # Create callbacks
    callbacks = create_callbacks(args, config, env, args.n_envs)

    # Determine device
    device = get_device(args)

    # Create or load model
    if model_path:
        print(f"Loading model from checkpoint: {model_path}")
        model = PPO.load(model_path, env=env, tensorboard_log=config["tensorboard_log"])
        reset_num_timesteps = False
    else:
        print(f"Creating new PPO model with {config['policy_type']}...")
        model = PPO(
            config["policy_type"],
            env,
            verbose=0,
            tensorboard_log=config["tensorboard_log"],
            device=device,
            **config["ppo_kwargs"],
        )
        reset_num_timesteps = True

    # Train
    print(f"Training for {args.timesteps} timesteps...")
    print(f"TensorBoard logs: tensorboard --logdir {config['tensorboard_log']}")
    print("-" * 80)

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps
    )

    # Save final model
    os.makedirs(config["base_dir"], exist_ok=True)
    model.save(config["model_save_path"])
    print(f"Model saved: {config['model_save_path']}")

    # Save VecNormalize stats
    if args.obs_type == "state" and isinstance(env, VecNormalize):
        env.save(config["model_save_path"] + "_vecnormalize.pkl")
        print(f"VecNormalize saved: {config['model_save_path']}_vecnormalize.pkl")

    env.close()
    print("Training complete!")

def test_env(args):
    """Main training function."""
    from envs import get_available_backends
    from envs.factory import load_config_from_yaml
    from envs.base import BasePushEnvConfig

    # Check backend availability
    available_backends = get_available_backends()
    if args.backend not in available_backends:
        print(f"Error: Backend '{args.backend}' is not available.")
        print(f"Available backends: {available_backends}")
        return

    print(f"Training with backend: {args.backend}")
    print(f"Observation type: {args.obs_type}")
    print(f"Number of environments: {args.n_envs}")

    # Load configuration
    if args.config:
        cfg = load_config_from_yaml(args.config)
        print(f"Using config file: {args.config}")
    else:
        cfg = BasePushEnvConfig()
        print("Using default configuration")

    # Get training config
    config = get_training_config(args, cfg)

    # Create output directory and save config before training
    os.makedirs(config["base_dir"], exist_ok=True)
    if args.config:
        config_dst = os.path.join(config["base_dir"], "config_used.yaml")
        shutil.copy(args.config, config_dst)
        print(f"Config saved: {config_dst}")

    # Resolve checkpoint
    model_path, vecnorm_path = resolve_checkpoint_path(args.load_checkpoint)

    # Create environment
    env = create_env(args, cfg, vecnorm_path)

    while True:
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            action = action.reshape((env.num_envs, -1))
            obs, reward, done, info = env.step(action)


if __name__ == "__main__":
    args = parse_args()
    if args.test:
        test_env(args)
    else:
        train(args)

