"""
Base environment interface for push task.

This module defines the abstract interface that all environment backends
(PyBullet, Isaac Lab) must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np


class BasePushEnvConfig:
    """Configuration class for push environment.

    This class holds all configuration parameters that are common across
    different backends (PyBullet, Isaac Lab).
    """

    def __init__(
        self,
        # Episode settings
        max_episode_steps: int = 200,

        # End-effector settings
        fixed_ee_z: float = 0.1,
        fixed_ee_initial_pos: Tuple[float, float] = (0.15, 0.0),

        # Success thresholds
        success_threshold: float = 0.05,
        orientation_threshold: float = 3.14,

        # Object and target dimensions
        object_half_extents: Tuple[float, float, float] = (0.06, 0.04, 0.05),
        target_half_extents: Tuple[float, float, float] = (0.06, 0.04, 0.001),

        # Spawn ranges
        object_r_range: Tuple[float, float] = (0.2, 0.5),
        object_theta_range: Tuple[float, float] = (-1.57, 0),
        target_r_range: Tuple[float, float] = (0.5, 0.9),
        target_theta_range: Tuple[float, float] = (-1.57, 1),

        # Reward coefficients
        distance_coef: float = 1.0,
        position_progress_coef: float = 30.0,
        orientation_progress_coef: float = 15.0,
        coupling_coef: float = 20.0,
        alignment_coef: float = 0.2,
        ee_approach_coef: float = 10.0,
        contact_threshold: float = 0.15,
        contact_reward: float = 0.05,
        success_bonus: float = 100.0,
        step_penalty: float = -0.01,
    ):
        # Episode settings
        self.max_episode_steps = max_episode_steps

        # End-effector settings
        self.fixed_ee_z = fixed_ee_z
        self.fixed_ee_initial_pos = fixed_ee_initial_pos

        # Success thresholds
        self.success_threshold = success_threshold
        self.orientation_threshold = orientation_threshold

        # Object and target dimensions
        self.object_half_extents = object_half_extents
        self.target_half_extents = target_half_extents

        # Spawn ranges
        self.object_r_range = object_r_range
        self.object_theta_range = object_theta_range
        self.target_r_range = target_r_range
        self.target_theta_range = target_theta_range

        # Reward coefficients
        self.distance_coef = distance_coef
        self.position_progress_coef = position_progress_coef
        self.orientation_progress_coef = orientation_progress_coef
        self.coupling_coef = coupling_coef
        self.alignment_coef = alignment_coef
        self.ee_approach_coef = ee_approach_coef
        self.contact_threshold = contact_threshold
        self.contact_reward = contact_reward
        self.success_bonus = success_bonus
        self.step_penalty = step_penalty

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BasePushEnvConfig":
        """Create config from dictionary (e.g., loaded from YAML)."""
        env_cfg = config_dict.get("env", {})
        reward_cfg = config_dict.get("reward", {})

        return cls(
            # Episode settings
            max_episode_steps=env_cfg.get("max_episode_steps", 200),

            # End-effector settings
            fixed_ee_z=env_cfg.get("fixed_ee_z", 0.1),
            fixed_ee_initial_pos=tuple(env_cfg.get("fixed_ee_initial_pos", [0.15, 0.0])),

            # Success thresholds
            success_threshold=env_cfg.get("success_threshold", 0.05),
            orientation_threshold=env_cfg.get("orientation_threshold", 3.14),

            # Object and target dimensions
            object_half_extents=tuple(env_cfg.get("object_half_extents", [0.06, 0.04, 0.05])),
            target_half_extents=tuple(env_cfg.get("target_half_extents", [0.06, 0.04, 0.001])),

            # Spawn ranges
            object_r_range=tuple(env_cfg.get("object_r_range", [0.2, 0.5])),
            object_theta_range=tuple(env_cfg.get("object_theta_range", [-1.57, 0])),
            target_r_range=tuple(env_cfg.get("target_r_range", [0.5, 0.9])),
            target_theta_range=tuple(env_cfg.get("target_theta_range", [-1.57, 1])),

            # Reward coefficients
            distance_coef=reward_cfg.get("distance_coef", 1.0),
            position_progress_coef=reward_cfg.get("position_progress_coef", 30.0),
            orientation_progress_coef=reward_cfg.get("orientation_progress_coef", 15.0),
            coupling_coef=reward_cfg.get("coupling_coef", 20.0),
            alignment_coef=reward_cfg.get("alignment_coef", 0.2),
            ee_approach_coef=reward_cfg.get("ee_approach_coef", 10.0),
            contact_threshold=reward_cfg.get("contact_threshold", 0.15),
            contact_reward=reward_cfg.get("contact_reward", 0.05),
            success_bonus=reward_cfg.get("success_bonus", 100.0),
            step_penalty=reward_cfg.get("step_penalty", -0.01),
        )


class BasePushEnv(ABC):
    """Abstract base class for push task environment.

    All environment backends must implement this interface to ensure
    compatibility with the unified training and evaluation scripts.

    The environment implements a robot arm pushing task:
    - Robot must push a rectangular object to a target position and orientation
    - Observation space: 19D state vector (or 84x84 RGB image)
    - Action space: 3D continuous (dx, dy, dz) - dz is ignored

    Observation features (state mode, 19D):
        1. ee_xy (2): End-effector position
        2. obj_xy (2): Object position
        3. target_xy (2): Target position
        4. ee_to_obj_xy (2): Relative position (EE to object)
        5. obj_to_target_xy (2): Relative position (object to target)
        6. obj_vel_xy (2): Object linear velocity
        7. obj_yaw_sincos (2): Object orientation (sin, cos encoding)
        8. target_yaw_sincos (2): Target orientation (sin, cos encoding)
        9. yaw_error_sincos (2): Orientation error (sin, cos encoding)
        10. obj_angular_vel (1): Object angular velocity (Z-axis)
    """

    # Class attributes that subclasses must define
    BACKEND_NAME: str = "base"
    SUPPORTS_GPU_PARALLEL: bool = False

    def __init__(
        self,
        cfg: BasePushEnvConfig,
        render_mode: Optional[str] = None,
        obs_type: str = "state",
        num_envs: int = 1,
        device: str = "cpu",
    ):
        """Initialize the push environment.

        Args:
            cfg: Environment configuration.
            render_mode: Render mode ("human", "rgb_array", or None).
            obs_type: Observation type ("state" or "image").
            num_envs: Number of parallel environments (only for GPU backends).
            device: Device to use ("cpu" or "cuda").
        """
        self.cfg = cfg
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.num_envs = num_envs
        self.device = device

    @abstractmethod
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options for reset.

        Returns:
            Tuple of (observation, info_dict).
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step.

        Args:
            action: Action to execute (3D: dx, dy, dz).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        pass

    @abstractmethod
    def close(self):
        """Clean up environment resources."""
        pass

    @abstractmethod
    def render(self) -> Optional[np.ndarray]:
        """Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise.
        """
        pass

    # Properties that subclasses must implement
    @property
    @abstractmethod
    def observation_space(self):
        """Return the observation space."""
        pass

    @property
    @abstractmethod
    def action_space(self):
        """Return the action space."""
        pass

    # Optional methods for debugging/visualization
    def get_ee_position(self) -> np.ndarray:
        """Get current end-effector position (for debugging)."""
        raise NotImplementedError

    def get_object_pose(self) -> Tuple[np.ndarray, float]:
        """Get current object position and yaw angle (for debugging)."""
        raise NotImplementedError

    def get_target_pose(self) -> Tuple[np.ndarray, float]:
        """Get target position and yaw angle (for debugging)."""
        raise NotImplementedError
