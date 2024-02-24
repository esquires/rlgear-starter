import random
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt

Array = npt.NDArray[Any]
ObsType = Array
ActType = np.int64


class SimpleEnv(gym.Env[ObsType, ActType]):
    def __init__(self, max_steps: int, sampling_range: tuple[int, int]) -> None:
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,))

        self.max_steps = max_steps
        self.sampling_range = sampling_range

        self.x = 0
        self.step_num = 0

        # this is just an example of how to log something to tensorboard
        # with an end of episode summary. see InfoToCustomMetricsCallback
        self.num_neg_actions = 0

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if action == 0:
            self.x -= 1
            self.num_neg_actions += 1
        elif action == 2:
            self.x += 1

        self.step_num += 1

        if self.x == 0:
            done = terminated = True
            reward = 1

            # this is just an example of how to log something to tensorboard
            info = {"num_neg_actions": self.num_neg_actions}
        else:
            done = terminated = False
            reward = 0
            info = {}

        if self.step_num >= self.max_steps:
            done = terminated = True

        obs = np.array([self.x], dtype=np.float32)

        return obs, reward, done, terminated, info

    # pylint: disable=unused-argument
    def reset(self, **kwargs: Any) -> tuple[ObsType, dict[str, Any]]:
        self.num_neg_actions = 0
        self.step_num = 0
        self.x = random.randint(self.sampling_range[0], self.sampling_range[1])
        obs = np.array([self.x], dtype=np.float32)
        return obs, {}

    def render(self) -> None:
        raise NotImplementedError


def env_creator_simple_env(cfg: dict[str, Any]) -> SimpleEnv:
    env = SimpleEnv(cfg["max_steps"], cfg["sampling_range"])
    return env
