"""
Implementation of the Walker2d environment supporting
domain randomization optimization.
"""

import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

class CustomWalker(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
        "render_fps": 125,
    }

    def __init__(
        self,
        xml_file: str = "walker2d.xml",
        frame_skip: int = 4,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        # Walker ha range leggermente diversi dall'Hopper
        healthy_z_range: Tuple[float, float] = (0.8, 2.0), 
        healthy_angle_range: Tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
        domain: Optional[str] = None,
        # PARAMETRI AGGIUNTIVI PER GESTIRE LA UDR/ADR
        enable_randomization: bool = False,
        uniform_randomization_range: float = 0.05,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            domain,
            enable_randomization,
            uniform_randomization_range,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self.enable_randomization = enable_randomization
        self.uniform_randomization_range = uniform_randomization_range
        self.boundary_mode = False

        if xml_file == "walker2d.xml":
            xml_file = os.path.join(os.path.dirname(__file__), "assets/walker2d.xml")

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 1 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 1 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }

        # Walker2d structure: 0:World, 1:Torso, 2,3,4: Right Leg, 5,6,7: Left Leg
        self.original_masses = np.copy(self.model.body_mass[1:]) 

        if domain == 'source':
            # Riduciamo la massa del torso (indice 1 nel body_mass globale)
            # Simile a quanto fatto per Hopper
            self.model.body_mass[1] -= 1.0

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]
        
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle
        is_healthy = healthy_z and healthy_angle

        return is_healthy

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = np.clip(self.data.qvel.flatten(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, False, info

    def _get_rew(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost

        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        if self.enable_randomization:
            self.set_random_parameters()

        observation = self._get_obs()
        return observation

    def set_boundary_mode(self, mode: bool):
        self.boundary_mode = mode

    def set_random_parameters(self):
        self.set_parameters(self.sample_parameters())

    def set_parameters(self, task):
        # Walker ha più parti mobili. 
        # Modifichiamo le masse da indice 2 in poi (Thigh Right... Foot Left)
        # Saltiamo World(0) e Torso(1)
        self.model.body_mass[2:] = task

    def sample_parameters(self):
        # original_masses[0] è il torso (che non cambiamo)
        # Quindi prendiamo da [1:] in poi (ovvero indices 2,3,4,5,6,7 del modello fisico)
        masses = self.original_masses[1:]

        if self.boundary_mode:
            signs = self.np_random.choice([-1.0, 1.0], size=masses.shape)
            sampled_masses = masses * (1.0 + signs * self.uniform_randomization_range)
        else:
            min_masses = masses * (1 - self.uniform_randomization_range)
            max_masses = masses * (1 + self.uniform_randomization_range)
            sampled_masses = self.np_random.uniform(min_masses, max_masses)
        
        return sampled_masses

    def set_udr_range(self, new_range: float):
        self.uniform_randomization_range = max(0.0, new_range)

# Registrazione ambienti
gym.register(
    id="CustomWalker-v0",
    entry_point="%s:CustomWalker" % __name__,
    max_episode_steps=1000,
)

gym.register(
    id="CustomWalker-source-v0",
    entry_point="%s:CustomWalker" % __name__,
    max_episode_steps=1000,
    kwargs={"domain": "source"}
)

gym.register(
    id="CustomWalker-target-v0",
    entry_point="%s:CustomWalker" % __name__,
    max_episode_steps=1000,
    kwargs={"domain": "target"}
)