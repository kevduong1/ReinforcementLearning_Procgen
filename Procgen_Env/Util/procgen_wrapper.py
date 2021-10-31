import time
import gym
import numpy as np
from Util.Configs import procGenConfig
from ray.tune import registry
from procgen.env import ENV_NAMES as VALID_ENV_NAMES

# TODO: Stack frames, preprocess observation

class ProcgenEnvWrapper(gym.Env):
    """
    Procgen Wrapper file
    """
    def __init__(self, type):
        self.config = procGenConfig(type).get_config()

        self.env_name = self.config.pop("env_name")
        assert self.env_name in VALID_ENV_NAMES

        # Store game rewards
        self.return_min = self.config.pop("return_min")
        self.return_blind = self.config.pop("return_blind")
        self.return_max = self.config.pop("return_max")

        env = gym.make(f"procgen:procgen-{self.env_name}-v0", **self.config)
        self.env = env
        # Enable video recording features
        self.metadata = self.env.metadata #TODO: figure out how to use this instead of gif

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._done = True

    def reset(self):
        assert self._done, "procgen envs cannot be early-restarted"
        return self.env.reset()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._done = done
        return obs, rew, done, info

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def __repr__(self):
        return self.env.__repr()

    @property
    def spec(self):
        return self.env.spec

# Register Env in Ray
registry.register_env(
    "procgen_env_wrapper",
    lambda config: ProcgenEnvWrapper(config)
)