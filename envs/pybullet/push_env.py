"""
PyBullet implementation of the push task environment.

This module implements the push task using PyBullet physics engine.
It supports both state-based and image-based observations.
"""

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import math
import time
from typing import Dict, Optional, Tuple, Any
from collections import namedtuple

from ..base import BasePushEnv, BasePushEnvConfig


class PyBulletPushEnv(BasePushEnv, gym.Env):
    """PyBullet implementation of push task environment.

    Uses Kuka iiwa 7-DOF robot arm to push a rectangular object
    to a target position and orientation.

    Physics: 120Hz simulation, 10Hz control (12 physics steps per action)
    """

    BACKEND_NAME = "pybullet"
    SUPPORTS_GPU_PARALLEL = False

    def __init__(
        self,
        cfg: BasePushEnvConfig,
        render_mode: Optional[str] = None,
        obs_type: str = "state",
        num_envs: int = 1,
        device: str = "cpu",
    ):
        """Initialize PyBullet push environment.

        Args:
            cfg: Environment configuration.
            render_mode: "human" for GUI, "rgb_array" for rendering, None for headless.
            obs_type: "state" for 19D vector, "image" for 84x84 RGB.
            num_envs: Ignored for PyBullet (always 1).
            device: Ignored for PyBullet (always CPU).
        """
        super().__init__(cfg, render_mode, obs_type, num_envs=1, device="cpu")


        self.asset_dir = r"C:\Users\Lenovo\projects\robotrl\envs\assets"
        self.fixed_ee_z = self.cfg.fixed_ee_z
        # Ranges
        self.object_r_range = self.cfg.object_r_range
        self.object_theta_range = self.cfg.object_theta_range
        self.target_r_range = self.cfg.target_r_range
        self.target_theta_range = self.cfg.target_theta_range

        # Define spaces
        self._action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.img_width = 84
        self.img_height = 84
        if self.obs_type == "image":
            self._observation_space = spaces.Box(
                low=0, high=255,
                shape=(self.img_height, self.img_width, 3),
                dtype=np.uint8
            )
        else:
            # 19D state observation
            self._observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(19,),
                dtype=np.float32
            )


        # Connect to PyBullet
        self.phisics_client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setGravity(0, 0, -9.8)
        self.dt = 1.0 / 240.0  # Internal physics step
        self.frame_skip = 24   # Control step: 240Hz / 24 = 10Hz control
        p.setTimeStep(self.dt)

        # State tracking for incremental rewards
        self.prev_dist_obj_target = None
        self.prev_dist_ee_obj = None
        self.prev_yaw_error = None
        self.step_count = 0

        # Target pose (set in reset)
        self.target_pos = np.zeros(3)
        self.target_yaw = 0.0

        # Fixed end-effector orientation (pointing down)
        self.fixed_orientation = p.getQuaternionFromEuler([math.pi, 0, 0])

        # Camera matrices for image observation
        self._load_static_resources()
        self._create_dynamic_actors()
        self._setup_cameras()

    def _load_static_resources(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = p.loadURDF("plane.urdf")
        
        # Robot
        p.setAdditionalSearchPath(self.asset_dir)
        base_pos = (0, 0, 0)
        base_ori = p.getQuaternionFromEuler((0, 0, 0))
        # Ensure file exists or handle error
        try:
            self.robotId = p.loadURDF("urdf/ur5_robotiq_85.urdf", base_pos, base_ori, useFixedBase=True)
        except Exception as e:
            print(f"Error loading robot URDF from {self.asset_dir}: {e}")
            raise e

        self.eef_id = 7
        self.fixed_orientation = p.getQuaternionFromEuler([0, math.pi/2, 0])
        
        # Joint setup
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.57, -1.54, 1.34, -1.37, -1.57, 0.0]
        self.__parse_joint_info__()
        self.__setup_mimic_joints__()

    def _create_dynamic_actors(self):
        # --- 尺寸定义 (与原代码一致) ---
        base_half_extents = [0.1, 0.025, 0.025]  # 横杠
        link_half_extents = [0.025, 0.075, 0.025] # 竖杠
        link_pos = [0, -0.1, 0] 

        # ==================================
        # 1. 创建可操作物体 (红色 T 形块)
        # ==================================
        baseCol = p.createCollisionShape(p.GEOM_BOX, halfExtents=base_half_extents)
        baseVis = p.createVisualShape(p.GEOM_BOX, halfExtents=base_half_extents, rgbaColor=[0.8, 0.1, 0.1, 1])
        
        linkCol = p.createCollisionShape(p.GEOM_BOX, halfExtents=link_half_extents)
        linkVis = p.createVisualShape(p.GEOM_BOX, halfExtents=link_half_extents, rgbaColor=[0.8, 0.1, 0.1, 1])
        
        self.objectId = p.createMultiBody(
            baseMass=0.5,
            baseCollisionShapeIndex=baseCol,
            baseVisualShapeIndex=baseVis,
            basePosition=[0, 0, 0.025],
            linkMasses=[0.5],
            linkCollisionShapeIndices=[linkCol],
            linkVisualShapeIndices=[linkVis],
            linkPositions=[link_pos],
            linkOrientations=[[0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 0]]
        )

        p.changeDynamics(self.objectId, -1, lateralFriction=0.6, spinningFriction=0.1)
        p.changeDynamics(self.objectId, 0, lateralFriction=0.6, spinningFriction=0.1)

        # ==================================
        # 2. 创建目标 (半透明绿色 T 形块 Ghost)
        # ==================================
        # 注意：Target 不需要 CollisionShape (设为 -1)，以免物理干扰
        # 颜色设置为半透明绿色 (Alpha=0.3)
        targetBaseVis = p.createVisualShape(p.GEOM_BOX, halfExtents=base_half_extents, rgbaColor=[0, 1, 0, 0.3])
        targetLinkVis = p.createVisualShape(p.GEOM_BOX, halfExtents=link_half_extents, rgbaColor=[0, 1, 0, 0.3])

        self.targetId = p.createMultiBody(
            baseMass=0, # 静态物体
            baseCollisionShapeIndex=-1, # 无碰撞
            baseVisualShapeIndex=targetBaseVis,
            basePosition=[0, 0, 0],
            # 必须构建完全相同的 Link 结构，才能在视觉上成为 T 形
            linkMasses=[0],
            linkCollisionShapeIndices=[-1], # 无碰撞
            linkVisualShapeIndices=[targetLinkVis],
            linkPositions=[link_pos],
            linkOrientations=[[0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 0]]
        )

    def _setup_cameras(self):
        """Set up camera matrices for rendering."""
        # Top-down view for image observation
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.5, 0, 2.0],
            cameraTargetPosition=[0.5, 0, 0],
            cameraUpVector=[1, 0, 0]
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.img_width) / self.img_height,
            nearVal=0.1,
            farVal=100.0
        )

        # Global camera for video recording
        self.global_view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1.2, -0.5, 0.8],
            cameraTargetPosition=[0.5, 0, 0.1],
            cameraUpVector=[0, 0, 1]
        )
        self.global_proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.img_width) / self.img_height,
            nearVal=0.1,
            farVal=100.0
        )

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        self.step_count = 0

        # Reset reward tracking
        self.prev_dist_obj_target = None
        self.prev_dist_ee_obj = None
        self.prev_yaw_error = None

        self.reset_arm()
        self.move_gripper(0.0)

        center_x, center_y = 0., 0.

        # --- 1. 重置物体 (Object) ---
        # 从配置中读取距离和方位角范围
        obj_r = self.np_random.uniform(*self.object_r_range)
        obj_theta = self.np_random.uniform(*self.object_theta_range)
        
        object_x = center_x + obj_r * np.cos(obj_theta)
        object_y = center_y + obj_r * np.sin(obj_theta)
        object_yaw = self.np_random.uniform(-np.pi, np.pi)
        
        self.object_pos = [object_x, object_y, 0.025]
        orn = p.getQuaternionFromEuler([0, 0, object_yaw])
        
        p.resetBasePositionAndOrientation(self.objectId, self.object_pos, orn)
        p.resetBaseVelocity(self.objectId, [0,0,0], [0,0,0])

        # --- 2. 重置目标 (Target) ---
        # 目标可以相对于物体进行偏移，或者也相对于中心点重置
        # 这里演示相对于中心点重置
        tar_r = self.np_random.uniform(*self.target_r_range)
        tar_theta = self.np_random.uniform(*self.target_theta_range)
        
        target_x = center_x + tar_r * np.cos(tar_theta)
        target_y = center_y + tar_r * np.sin(tar_theta)
        target_yaw = self.np_random.uniform(-np.pi, np.pi)

        self.target_pos = np.array([target_x, target_y, 0.0])
        self.target_yaw = target_yaw
        target_orn = p.getQuaternionFromEuler([0, 0, target_yaw])
        
        p.resetBasePositionAndOrientation(self.targetId, self.target_pos, target_orn)

        # 步进几帧让物体落稳
        for _ in range(20): p.stepSimulation()
        
        # Init Reward Vars
        self.prev_dist_obj_target = None
        self.prev_dist_ee_obj = None
        self.prev_ang_dist = None

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        # Scale action
        dx, dy = action * 0.1

        # Get Current EE Pos (Fast retrieval)
        current_ee_state = p.getLinkState(self.robotId, self.eef_id)
        current_ee_pos = current_ee_state[0]
        
        # Target Position
        new_ee_pos = [current_ee_pos[0] + dx, current_ee_pos[1] + dy, self.fixed_ee_z]

        # IK Calculation
        joint_poses = p.calculateInverseKinematics(
            self.robotId, self.eef_id, new_ee_pos, self.fixed_orientation,
            self.arm_lower_limits, self.arm_upper_limits, self.arm_joint_ranges, self.arm_rest_poses,
            maxNumIterations=10 # Reduced iterations for speed, usually enough for small deltas
        )
        
        # Motor Control
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(self.robotId, joint_id, p.POSITION_CONTROL, joint_poses[i], force=self.joints[joint_id].maxForce)

        # Physics Stepping (Frame Skip)
        for _ in range(self.frame_skip):
            p.stepSimulation()
            # Enforce flat object constraint simply by re-asserting Z/Orientation IF necessary
            # But relying on correct friction/inertia is better physics.
            # If "teleport" is strictly needed, do it ONLY if object tips excessively.
            # Here we skip the hack to respect physics, assuming box CoM is low.

        if self.render_mode == "human":
            time.sleep(self.dt * self.frame_skip)

        self.step_count += 1
        
        # Observation & Reward
        obs = self._get_obs()
        reward, terminated, truncated = self._compute_reward()

        return obs, reward, terminated, truncated, {}
    
    def _angle_normalize(self, angle):
        """将角度归一化到 [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        if self.obs_type == "image":
            return self._get_image_obs()
        else:
            return self._get_state_obs()

    def _get_state_obs(self) -> np.ndarray:
        """Get 19D state observation."""
        ee_pos = np.array(p.getLinkState(self.robotId, self.eef_id)[0][:2])
        obj_pos_3d, obj_orn = p.getBasePositionAndOrientation(self.objectId)
        obj_pos = np.array(obj_pos_3d[:2])
        obj_vel, _ = p.getBaseVelocity(self.objectId)
        
        _, _, obj_yaw = p.getEulerFromQuaternion(obj_orn)

        # Target position
        target_pos = np.array(self.target_pos[:2])
        target_yaw = self.target_yaw

        # Relative positions
        ee_to_obj = obj_pos - ee_pos
        obj_to_target = target_pos - obj_pos

        # Object velocity
        obj_vel_full, obj_ang_vel_full = p.getBaseVelocity(self.objectId)
        obj_vel = np.array(obj_vel_full[:2])
        obj_angular_vel = obj_ang_vel_full[2]

        # Yaw error
        yaw_error = self._normalize_angle(target_yaw - obj_yaw)

        # Sin/cos encoding
        obj_yaw_sincos = [np.sin(obj_yaw), np.cos(obj_yaw)]
        target_yaw_sincos = [np.sin(target_yaw), np.cos(target_yaw)]
        yaw_error_sincos = [np.sin(yaw_error), np.cos(yaw_error)]

        # Concatenate
        obs = np.concatenate((
            ee_pos,
            obj_pos,
            target_pos,
            ee_to_obj,
            obj_to_target,
            obj_vel,
            obj_yaw_sincos,
            target_yaw_sincos,
            yaw_error_sincos,
            [obj_angular_vel]
        ))

        return obs.astype(np.float32)

    def _get_image_obs(self) -> np.ndarray:
        """Get 84x84 RGB image observation."""
        w, h, rgb, _, _ = p.getCameraImage(
            width=self.img_width,
            height=self.img_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )
        rgb = np.array(rgb, dtype=np.uint8)
        rgb = np.reshape(rgb, (h, w, 4))
        rgb = rgb[:, :, :3]
        return rgb

    def _compute_reward(self) -> Tuple[float, bool, bool]:
        """Compute reward, terminated, truncated."""
        # Get positions
        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.objectId)
        obj_pos = np.array(obj_pos[:2])
        obj_euler = p.getEulerFromQuaternion(obj_orn)
        obj_yaw = obj_euler[2]

        target_pos = np.array(self.target_pos[:2])
        target_yaw = self.target_yaw

        ee_pos = np.array(p.getLinkState(self.robotId, self.eef_id)[0][:2])

        # Distances and errors
        dist_obj_target = np.linalg.norm(obj_pos - target_pos)
        dist_ee_obj = np.linalg.norm(ee_pos - obj_pos)
        yaw_error = abs(self._normalize_angle(target_yaw - obj_yaw))

        # Initialize previous values
        if self.prev_dist_obj_target is None:
            self.prev_dist_obj_target = dist_obj_target
        if self.prev_dist_ee_obj is None:
            self.prev_dist_ee_obj = dist_ee_obj
        if self.prev_yaw_error is None:
            self.prev_yaw_error = yaw_error

        # Incremental changes
        delta_dist_target = self.prev_dist_obj_target - dist_obj_target
        delta_dist_ee_obj = self.prev_dist_ee_obj - dist_ee_obj
        delta_yaw_error = self.prev_yaw_error - yaw_error

        # Update previous values
        self.prev_dist_obj_target = dist_obj_target
        self.prev_dist_ee_obj = dist_ee_obj
        self.prev_yaw_error = yaw_error

        # Reward components
        reward = self.cfg.step_penalty

        # Position progress
        reward += delta_dist_target * self.cfg.position_progress_coef

        # Orientation progress
        reward += delta_yaw_error * self.cfg.orientation_progress_coef

        # Coupling bonus
        if delta_dist_target > 0 and delta_yaw_error > 0:
            coupling = min(delta_dist_target * 10, delta_yaw_error) * self.cfg.coupling_coef
            reward += coupling

        # Alignment reward
        vec_ee_to_obj = obj_pos - ee_pos
        vec_obj_to_target = target_pos - obj_pos
        norm_ee_obj = np.linalg.norm(vec_ee_to_obj)
        norm_obj_target = np.linalg.norm(vec_obj_to_target)

        if norm_ee_obj > 0.01 and norm_obj_target > 0.01:
            alignment = np.dot(vec_ee_to_obj, vec_obj_to_target) / (norm_ee_obj * norm_obj_target)
            if dist_ee_obj < self.cfg.contact_threshold * 2:
                reward += alignment * self.cfg.alignment_coef

        # EE approach reward
        if dist_ee_obj > self.cfg.contact_threshold:
            reward += delta_dist_ee_obj * self.cfg.ee_approach_coef
        else:
            if delta_dist_target > 0:
                reward += self.cfg.contact_reward * 2.0
            else:
                reward += self.cfg.contact_reward * 0.5

        # Success check
        terminated = False
        position_success = dist_obj_target < self.cfg.success_threshold
        orientation_success = yaw_error < self.cfg.orientation_threshold

        if position_success and orientation_success:
            reward += self.cfg.success_bonus
            terminated = True

        truncated = (not terminated) and (self.step_count >= self.cfg.max_episode_steps)

        return reward, terminated, truncated

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def reset_arm(self):
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_controllable_joints):
            p.resetJointState(self.robotId, joint_id, rest_pose)

    def move_gripper(self, open_length):
        # Simplified mimic logic trigger
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)
        p.setJointMotorControl2(self.robotId, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle)

    def __parse_joint_info__(self):
        numJoints = p.getNumJoints(self.robotId)
        jointInfo = namedtuple('jointInfo', 
            ['id','name','type','damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.robotId, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != p.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                p.setJointMotorControl2(self.robotId, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                            jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
            self.joints.append(info)

        assert len(self.controllable_joints) >= self.arm_num_dofs
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]

        self.arm_lower_limits = [info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [info.upperLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [info.upperLimit - info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]

    def __setup_mimic_joints__(self):
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': 1,
                                'left_inner_knuckle_joint': 1,
                                'right_inner_knuckle_joint': 1,
                                'left_inner_finger_joint': -1,
                                'right_inner_finger_joint': -1}
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.robotId, self.mimic_parent_id,
                                   self.robotId, joint_id,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)  # Note: the mysterious `erp` is of EXTREME importance

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            w, h, rgb, _, _ = p.getCameraImage(
                width=320,
                height=240,
                viewMatrix=self.global_view_matrix,
                projectionMatrix=self.global_proj_matrix,
                renderer=p.ER_TINY_RENDERER
            )
            rgb = np.array(rgb, dtype=np.uint8)
            rgb = np.reshape(rgb, (240, 320, 4))
            return rgb[:, :, :3]
        return None

    def close(self):
        """Close the environment."""
        p.disconnect()

    # Debug methods
    def get_ee_position(self) -> np.ndarray:
        """Get current end-effector position."""
        ee_state = p.getLinkState(self.robotId, self.eef_id)
        return np.array(ee_state[0])

    def get_object_pose(self) -> Tuple[np.ndarray, float]:
        """Get current object position and yaw angle."""
        pos, orn = p.getBasePositionAndOrientation(self.objectId)
        euler = p.getEulerFromQuaternion(orn)
        return np.array(pos), euler[2]

    def get_target_pose(self) -> Tuple[np.ndarray, float]:
        """Get target position and yaw angle."""
        return self.target_pos.copy(), self.target_yaw
