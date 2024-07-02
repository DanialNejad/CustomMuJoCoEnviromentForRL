import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os
import mujoco  # Import mujoco to use mj_name2id function

class BallBalanceEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, episode_len=500, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            os.path.abspath("assets/ball_balance1.xml"),
            5,
            observation_space=observation_space,
            **kwargs
        )
        self.step_number = 0
        self.episode_len = episode_len
        self.target_pos = np.zeros(3)  # Initialize the target position

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        self.step_number += 1

        obs = self._get_obs()
        ball_pos = self.data.joint("ball").qpos[:3]

        # Calculate distance to the target position in the plane (x, y)
        distance_to_target = np.linalg.norm(ball_pos[:2] - self.target_pos[:2])

        # Reward: Negative distance to the target (closer is better)
        reward = -distance_to_target

        done = bool(not np.isfinite(obs).all() or (obs[2] < 0))
        truncated = self.step_number > self.episode_len
        return obs, reward, done, truncated, {}

    def reset_model(self):
        self.step_number = 0

        # Randomize the target position within the plane boundaries
        plane_size = 2.0  # The plane size is 2x2 as defined in the XML
        self.target_pos[:2] = self.np_random.uniform(low=-plane_size/2, high=plane_size/2, size=2)
        self.target_pos[2] = -0.5  # Set z = -0.5 to be on the plane

        # Update the position of the target geom in the simulation
        target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'target')
        self.model.body_pos[target_id] = self.target_pos

        # Add noise to positions and velocities
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        obs = np.concatenate((
            np.array(self.data.joint("ball").qpos[:3]),
            np.array(self.data.joint("ball").qvel[:3]),
            np.array(self.data.joint("rotate_x").qpos),
            np.array(self.data.joint("rotate_x").qvel),
            np.array(self.data.joint("rotate_y").qpos),
            np.array(self.data.joint("rotate_y").qvel)
        ), axis=0)
        return obs
