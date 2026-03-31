"""
Environment factory for unified environment creation.

This module provides a unified interface for creating push task environments
regardless of the backend (PyBullet or Isaac Lab).
"""

from typing import Optional, List, Callable, Dict, Any, Union
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from .base import BasePushEnvConfig, BasePushEnv
from .pybullet import PyBulletPushEnv, WMPyBulletPushEnv


def get_available_backends() -> List[str]:
    """Get list of available environment backends.

    Returns:
        List of available backend names.
    """
    backends = ["pybullet", "pybullet_wm"]  # PyBullet backends are always available

    # Check Isaac Lab availability
    try:
        from .isaac_lab import check_isaac_lab_available
        if check_isaac_lab_available():
            backends.append("isaac_lab")
    except ImportError:
        pass

    return backends


def make_env(
    backend: str = "pybullet",
    cfg: Optional[BasePushEnvConfig] = None,
    render_mode: Optional[str] = None,
    obs_type: str = "state",
    num_envs: int = 1,
    device: str = "cpu",
    max_episode_steps: Optional[int] = None,
) -> gym.Env:
    """Create a push task environment.

    Args:
        backend: Backend to use ("pybullet" or "isaac_lab").
        cfg: Environment configuration. If None, uses default config.
        render_mode: Render mode ("human", "rgb_array", or None).
        obs_type: Observation type ("state" or "image").
        num_envs: Number of parallel environments (only for Isaac Lab).
        device: Device to use ("cpu" or "cuda").
        max_episode_steps: Override max episode steps from config.

    Returns:
        Gymnasium-compatible environment.

    Raises:
        ValueError: If backend is not available.
    """
    available = get_available_backends()
    if backend not in available:
        raise ValueError(
            f"Backend '{backend}' is not available. "
            f"Available backends: {available}"
        )

    # Use default config if not provided
    if cfg is None:
        cfg = BasePushEnvConfig()

    # Override max_episode_steps if provided
    if max_episode_steps is not None:
        cfg.max_episode_steps = max_episode_steps

    # Create environment based on backend
    if backend == "pybullet":
        env = PyBulletPushEnv(
            cfg=cfg,
            render_mode=render_mode,
            obs_type=obs_type,
            num_envs=1,  # PyBullet always uses 1 env
            device=device,  # PyBullet always uses CPU
        )
        # Wrap with TimeLimit
        env = TimeLimit(env, max_episode_steps=cfg.max_episode_steps)

    elif backend == "pybullet_wm":
        env = WMPyBulletPushEnv(
            cfg=cfg,
            render_mode=render_mode,
            obs_type=obs_type,
            num_envs=1,
            device=device,
        )
        env = TimeLimit(env, max_episode_steps=cfg.max_episode_steps)

    elif backend == "isaac_lab":
        from .isaac_lab import IsaacLabPushEnv

        if num_envs > 1:
            # For vectorized environments, use the VecEnv wrapper
            from .isaac_lab import IsaacLabVecEnvWrapper
            env = IsaacLabVecEnvWrapper(
                cfg=cfg,
                num_envs=num_envs,
                device=device,
                render_mode=render_mode,
            )
        else:
            env = IsaacLabPushEnv(
                cfg=cfg,
                render_mode=render_mode,
                obs_type=obs_type,
                num_envs=1,
                device=device,
            )
            env = TimeLimit(env, max_episode_steps=cfg.max_episode_steps)

    return env


def make_vec_env(
    backend: str = "pybullet",
    cfg: Optional[BasePushEnvConfig] = None,
    n_envs: int = 12,
    obs_type: str = "state",
    device: str = "cpu",
    vec_env_cls: Optional[type] = None,
    max_episode_steps: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> Any:
    """Create vectorized push task environments.

    For PyBullet: Uses SubprocVecEnv for parallel CPU environments.
    For Isaac Lab: Uses native GPU parallelization.

    Args:
        backend: Backend to use ("pybullet" or "isaac_lab").
        cfg: Environment configuration. If None, uses default config.
        n_envs: Number of parallel environments.
        obs_type: Observation type ("state" or "image").
        device: Device to use ("cpu" or "cuda").
        vec_env_cls: VecEnv class to use for PyBullet (default: SubprocVecEnv).
        max_episode_steps: Override max episode steps from config.

    Returns:
        Vectorized environment compatible with Stable Baselines 3.
    """
    from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    # Use default config if not provided
    if cfg is None:
        cfg = BasePushEnvConfig()

    if max_episode_steps is not None:
        cfg.max_episode_steps = max_episode_steps

    if backend == "pybullet":
        # Use SB3's make_vec_env for PyBullet
        if vec_env_cls is None:
            vec_env_cls = SubprocVecEnv

        def make_env_fn():
            env = PyBulletPushEnv(
                cfg=cfg,
                render_mode=render_mode,
                obs_type=obs_type,
            )
            env = TimeLimit(env, max_episode_steps=cfg.max_episode_steps)
            return env

        env = sb3_make_vec_env(make_env_fn, n_envs=n_envs, vec_env_cls=vec_env_cls, monitor_kwargs={"info_keywords": ("is_success",)})

        # Apply VecNormalize for state observations
        if obs_type == "state":
            env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10., gamma=0.99)

        return env

    elif backend == "pybullet_wm":
        if vec_env_cls is None:
            vec_env_cls = SubprocVecEnv

        def make_env_fn():
            env = WMPyBulletPushEnv(
                cfg=cfg,
                render_mode=render_mode,
                obs_type=obs_type,
            )
            env = TimeLimit(env, max_episode_steps=cfg.max_episode_steps)
            return env

        env = sb3_make_vec_env(make_env_fn, n_envs=n_envs, vec_env_cls=vec_env_cls, monitor_kwargs={"info_keywords": ("is_success",)})

        if obs_type == "state":
            env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10., gamma=0.99)

        return env

    elif backend == "isaac_lab":
        # Use Isaac Lab's native GPU parallelization
        from .isaac_lab import IsaacLabVecEnvWrapper

        env = IsaacLabVecEnvWrapper(
            cfg=cfg,
            num_envs=n_envs,
            device=device,
        )

        # Note: VecNormalize can still be applied if needed
        # but Isaac Lab's GPU tensors need special handling
        return env

    else:
        raise ValueError(f"Unknown backend: {backend}")


def load_config_from_yaml(yaml_path: str) -> BasePushEnvConfig:
    """Load environment configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file.

    Returns:
        BasePushEnvConfig loaded from the file.
    """
    import yaml

    with open(yaml_path, 'r', encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return BasePushEnvConfig.from_dict(config_dict)
