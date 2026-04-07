"""
World-model-driven PyBullet push environment.

This environment keeps the same reset/observation/reward logic as PyBulletPushEnv,
but changes the step transition to use learned PINNs world-model dynamics.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import pybullet as p
from scipy.spatial.transform import Rotation as R

from .push_env import PyBulletPushEnv
from ..base import BasePushEnvConfig


class MLPV(torch.nn.Module):
    """Velocity model used by current PINN notebooks (predicts 2D translational velocity)."""

    def __init__(
        self,
        input_dim: int = 24,
        hidden_dims: Tuple[int, ...] = (32, 32, 32, 32, 32, 32, 32, 32, 32, 32),
    ):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers += [torch.nn.Linear(last, h), torch.nn.Tanh()]
            last = h
        self.shared = torch.nn.Sequential(*layers)
        self.fc_v = torch.nn.Linear(last, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        return self.fc_v(h)


class MLPW(torch.nn.Module):
    """Angular model used by current PINN notebooks (predicts scalar angular velocity)."""

    def __init__(
        self,
        input_dim: int = 24,
        hidden_dims: Tuple[int, ...] = (32, 32, 32, 32, 32, 32, 32, 32, 32, 32),
    ):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers += [torch.nn.Linear(last, h), torch.nn.Tanh()]
            last = h
        self.shared = torch.nn.Sequential(*layers)
        self.fc_w = torch.nn.Linear(last, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        return self.fc_w(h)


class VPrediction(torch.nn.Module):
    def __init__(self, net: torch.nn.Module, device: str = "cpu"):
        super().__init__()
        self.net = net.to(device)
        self.device = device

    def forward(self, contact: torch.Tensor, vo: torch.Tensor, ve: torch.Tensor) -> torch.Tensor:
        x = torch.cat([contact, vo, ve], dim=0).unsqueeze(0).to(self.device).float()
        v = self.net(x)
        return v.squeeze(0)


class WPrediction(torch.nn.Module):
    def __init__(self, net: torch.nn.Module, device: str = "cpu"):
        super().__init__()
        self.net = net.to(device)
        self.device = device

    def forward(self, contact: torch.Tensor, vo: torch.Tensor, ve: torch.Tensor) -> torch.Tensor:
        x = torch.cat([contact, vo, ve], dim=0).unsqueeze(0).to(self.device).float()
        w = self.net(x)
        return w.squeeze(0)


def _load_v_prediction_model(path: str, device: str = "cpu") -> VPrediction:
    net = MLPV(input_dim=24)
    model = VPrediction(net, device)

    state_dict = torch.load(path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        if new_key.startswith("net."):
            new_key = new_key[len("net."):]
        new_key = new_key.replace("fc_trans", "fc_v")
        new_key = "net." + new_key
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model


def _load_w_prediction_model(path: str, device: str = "cpu") -> WPrediction:
    net = MLPW(input_dim=24)
    model = WPrediction(net, device)

    state_dict = torch.load(path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        if new_key.startswith("net."):
            new_key = new_key[len("net."):]
        new_key = new_key.replace("fc_rot", "fc_w")
        new_key = "net." + new_key
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model


@torch.no_grad()
def _infer_pose(
    xt: np.ndarray,
    at: np.ndarray,
    contact_info: np.ndarray,
    model_v: VPrediction,
    model_w: WPrediction,
    dt: float,
) -> np.ndarray:
    p_w = xt[0:3]
    r_w = xt[3:6]
    v_w = xt[6:9]
    w_w = xt[9:12]

    v_tcp_w = at[6:9]
    w_tcp_w = at[9:12]

    r_wb = R.from_rotvec(r_w)

    vo = np.hstack([r_wb.inv().apply(v_w), r_wb.inv().apply(w_w)])
    ve = np.hstack([r_wb.inv().apply(v_tcp_w), r_wb.inv().apply(w_tcp_w)])

    contact_b = []
    for i in range(2):
        pw = contact_info[i * 6:i * 6 + 3]
        nw = contact_info[i * 6 + 3:i * 6 + 6]
        contact_b += list(r_wb.inv().apply(pw - p_w))
        contact_b += list(r_wb.inv().apply(nw))

    contact_b = torch.from_numpy(np.array(contact_b)).float()
    vo = torch.from_numpy(vo).float()
    ve = torch.from_numpy(ve).float()

    vxy_pred = model_v(contact_b, vo, ve).cpu().numpy()
    wz_pred = model_w(contact_b, vo, ve).item()

    w_next_b = np.zeros(3, dtype=np.float32)
    v_next_b = np.zeros(3, dtype=np.float32)
    w_next_b[0] = wz_pred
    v_next_b[1] = vxy_pred[0]
    v_next_b[2] = vxy_pred[1]

    v_next_w = r_wb.apply(v_next_b)
    w_next_w = r_wb.apply(w_next_b)

    xt_next = np.zeros(12, dtype=np.float32)
    xt_next[0:3] = p_w + v_next_w * dt
    xt_next[3:6] = (R.from_rotvec(w_next_w * dt) * r_wb).as_rotvec()
    xt_next[6:9] = v_next_w
    xt_next[9:12] = w_next_w

    return xt_next


class WMPyBulletPushEnv(PyBulletPushEnv):
    """PyBullet environment with world-model dynamics for step transitions."""

    BACKEND_NAME = "pybullet_wm"

    def __init__(
        self,
        cfg: BasePushEnvConfig,
        render_mode: Optional[str] = None,
        obs_type: str = "state",
        num_envs: int = 1,
        device: str = "cuda",
    ):
        super().__init__(cfg, render_mode, obs_type, num_envs, device)

        model_dir = os.path.join("envs", "WM_models")
        self.model_v_path = os.getenv("WM_V_MODEL_PATH", os.path.join(model_dir, "v_prediction.pth"))
        self.model_w_path = os.getenv("WM_W_MODEL_PATH", os.path.join(model_dir, "w_prediction.pth"))
        self.wm_device = device
        self.control_dt = self.dt * self.frame_skip

        if not os.path.exists(self.model_v_path) or not os.path.exists(self.model_w_path):
            raise FileNotFoundError(
                "World-model weights not found. "
                "Expected files: "
                f"{self.model_v_path} and {self.model_w_path}. "
                "You can override paths via WM_V_MODEL_PATH and WM_W_MODEL_PATH."
            )

        self.model_v = _load_v_prediction_model(self.model_v_path, device=self.wm_device)
        self.model_w = _load_w_prediction_model(self.model_w_path, device=self.wm_device)

        self._wm_xt = np.zeros(12, dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        obs, info = super().reset(seed=seed, options=options)

        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.objectId)
        obj_rotvec = R.from_quat(obj_orn).as_rotvec()

        self._wm_xt = np.zeros(12, dtype=np.float32)
        self._wm_xt[0:3] = np.array(obj_pos, dtype=np.float32)
        self._wm_xt[3:6] = obj_rotvec.astype(np.float32)

        self.ee_pos = self.get_ee_position()

        return obs, info

    def step(self, action: np.ndarray):
        """Execute one environment step using PINNs world-model transition."""
        dx, dy = action * 0.1
        dx = np.clip(dx, -1.0 - self.ee_pos[0], 1.0 - self.ee_pos[0])
        dy = np.clip(dy, -1.0 - self.ee_pos[1], 1.0 - self.ee_pos[1])

        target_ee_pos = self.ee_pos.copy()
        target_ee_pos[0] += dx
        target_ee_pos[1] += dy

        at = np.zeros(12, dtype=np.float32)
        at[6:9] = np.array([dx / self.control_dt, dy / self.control_dt, 0.0], dtype=np.float32)
        at[9:12] = np.zeros(3, dtype=np.float32)

        # Teleport EE directly according to action (no physics stepping).
        _, ee_orn = p.getBasePositionAndOrientation(self.eeId)
        p.resetBasePositionAndOrientation(self.eeId, target_ee_pos.tolist(), ee_orn)
        p.resetBaseVelocity(self.eeId, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

        p.performCollisionDetection()

        # Only use WM transition when there is actual contact.
        contacts = p.getContactPoints(bodyA=self.objectId, bodyB=self.eeId)
        has_collision = len(contacts) > 0

        if has_collision:
            contact_info = np.zeros(12, dtype=np.float32)
            for i, c in enumerate(contacts[:2]):
                contact_info[i * 6:i * 6 + 3] = np.array(c[5], dtype=np.float32)
                contact_info[i * 6 + 3:i * 6 + 6] = np.array(c[7], dtype=np.float32)

            self._wm_xt = _infer_pose(
                xt=self._wm_xt,
                at=at,
                contact_info=contact_info,
                model_v=self.model_v,
                model_w=self.model_w,
                dt=self.control_dt,
            )

            next_quat = R.from_rotvec(self._wm_xt[3:6]).as_quat()
            p.resetBasePositionAndOrientation(self.objectId, self._wm_xt[0:3].tolist(), next_quat.tolist())
            p.resetBaseVelocity(self.objectId, self._wm_xt[6:9].tolist(), self._wm_xt[9:12].tolist())

        self.ee_pos = target_ee_pos

        if self.render_mode == "human":
            time.sleep(self.control_dt)

        self.step_count += 1

        obs = self._get_obs()
        reward, terminated, truncated = self._compute_reward()
        info = {"is_success": terminated}

        return obs, reward, terminated, truncated, info
