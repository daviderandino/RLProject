"""
Implementation of the Hopper environment supporting
domain randomization optimization.

E' l'ambiente in cui vive l'agente.
"""

import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


# setta la posizione della telecamera quando si apre la finestra
# grafica di MuJoCo
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

# classe che definisce l'hopper
class CustomHopper(MujocoEnv, utils.EzPickle):
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
        xml_file: str = "hopper.xml",
        frame_skip: int = 4,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0, # se se viaggia a 1ms/s è +1 reward, se 2ms è +2
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0, # ad ogni step se il robot è sano prende +1 reward
        terminate_when_unhealthy: bool = True,
        healthy_state_range: Tuple[float, float] = (-100.0, 100.0),
        healthy_z_range: Tuple[float, float] = (0.7, float("inf")),
        healthy_angle_range: Tuple[float, float] = (-0.2, 0.2),
        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
        domain: Optional[str] = None, # indica il domain in cui vivrà l'hopper
        enable_randomization: bool = False, # parametro per usare o meno la UDR
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
            healthy_state_range,
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
        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        # ATTRIBUTI AGGIUNTIVI DELLA CLASSE
        self.enable_randomization = enable_randomization
        self.uniform_randomization_range = uniform_randomization_range
        self.boundary_mode = False

        if xml_file == "hopper.xml":
             xml_file = os.path.join(os.path.dirname(__file__), "assets/hopper.xml")

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

        # in MuJoCo, 'self.model.body_mass' contiene le masse di tutti i corpi, incluso
        # il mondo ovvero il pavimento (che è statico).
        self.original_masses = np.copy(self.model.body_mass[1:]) # Default link masses

        # nell'ambiente sorgente (dove dovremmo allenare il robot) la massa
        # del torso è stata ridotta di 1kg. E' stata modificata solo una delle
        # 4 masse (torso, thigh, leg e foot).
        #
        # Questo errore dovrà essere sistematico tra simulatore e realtà, quindi
        # non devo randomizzarlo in training. 
        if domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
            # effettuiamo la modifica della massa direttamente nel motore fisico
            self.model.body_mass[1] -= 1.0


    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        """
        Definisce se il robot è healty o meno, così da poter terminare
        eventualmente l'episodio.
        """
        z, angle = self.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z # il robot è abbastanza alto?
        healthy_angle = min_angle < angle < max_angle # il robot è abbastanza dritto?

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    def _get_obs(self):
        """
        Ottiene il vettore di osservazioni dall'ambiente.
        """
        position = self.data.qpos.flatten()
        velocity = np.clip(self.data.qvel.flatten(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        """
        Chiamata per far fare all'agente un'azione.

        Misura dov'è il robot, applica la forza ai motori e fa avanzare il tempo.
        Infine, misura dove è arrivato il robot.
        """
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": x_position_after,
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
            "x_velocity": x_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, False, info

    def _get_rew(self, x_velocity: float, action):
        """
        Composta da 3 termini:
        - reward di sopravvivenza
        - reward di velocità
        - penalty di input di controllo
        """
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
        """
        Chiamata ad inizio episodio.
        """
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        # Custom domain randomization
        if self.enable_randomization:
            self.set_random_parameters()

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
        }

    def set_boundary_mode(self, mode: bool):
        self.boundary_mode = mode

    def set_random_parameters(self):
        """
        Set random masses.
        Chiama il campionamento delle masse e le setta.
        """
        # se faccio self.set_parameters(*self.sample_parameters()) passo più di un
        # argomento e quindi uscirebbe errore, per come è la firma di set_parameters(...)
        self.set_parameters(self.sample_parameters())

    def set_parameters(self, task):
        """
        Set each hopper link's mass to a new value.
        Inietta i numeri generati nel motore fisico, direttamente
        nell'array dell'engine.
        """
        # task contiene 3 valori: [nuova_massa_thigh, nuova_massa_leg, nuova_massa_foot].
        # In MuJoCo body_mass ha indici: 0=World, 1=Torso, 2=Thigh, 3=Leg, 4=Foot.
        # Quindi devo modificare dall'indice 2 in poi.
        self.model.body_mass[2:] = task # ci va 1: o 2: ???

    def sample_parameters(self):
        """
        OpenAI Style:
        - Training: Uniforme (impara tutto il range)
        - Eval: Boundary (testa solo il peggio)

        Note:
        - Each mass is samples from his uniform distribution ~ U(m_min, m_max).
        - You can't change the mass of the torso (first link).
        """
        # prendo le masse originali di [Thigh, Leg, Foot].
        # 'original_masses[0]' è il torso che non devo cambiare,        
        # quindi lo saltiamo partendo da [1:].
        masses = self.original_masses[1:]

        if self.boundary_mode:
            # --- LOGICA BOUNDARY (Test severo) ---
            # Genera array di fatto casualmente di -1 e +1
            signs = self.np_random.choice([-1.0, 1.0], size=masses.shape)
            # Applica -> massa * (1 +/- range)
            sampled_masses = masses * (1.0 + signs * self.uniform_randomization_range)
        else:
            # --- LOGICA UNIFORME (Training standard) ---
            min_masses = masses * (1 - self.uniform_randomization_range)
            max_masses = masses * (1 + self.uniform_randomization_range)
            sampled_masses = self.np_random.uniform(min_masses, max_masses)
        
        return sampled_masses

    def set_udr_range(self, new_range: float):
        """
        Metodo chiamato dalla callback ADR per aggiornare la
        difficoltà e modificare il range della distribuzione
        uniforme da cui samplare i parametri.
        """
        self.uniform_randomization_range = max(0.0, new_range)

    def get_parameters(self):
        """
        Get value of mass for each link.
        """
        masses = np.array(self.model.body_mass[1:])
        return masses


"""
Registered environments.
"""
gym.register(
    # ambiente base
    id="CustomHopper-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
)

gym.register(
    # ambiente source di training
    id="CustomHopper-source-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "source"}
)

gym.register(
    # ambiente vero di testing
    id="CustomHopper-target-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "target"}
)
