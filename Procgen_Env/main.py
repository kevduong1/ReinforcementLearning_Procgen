import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from procgen import ProcgenGym3Env


ray.init()

tune.run(
    DQNTrainer,
    config={"env": "CartPole-v0", "framework": "torch"},stop={"episode_reward_mean": 50},
    checkpoint_at_end=True,
    checkpoint_freq=5,
    #save checkpoint to local directory
    local_dir="./checkpoint"
    )



"""
import ray
from ray import tune

ray.init()
tune.run(
    "PPO",
    stop={"episode_reward_mean": 200},
    config={
        "env": "CartPole-v0",
        "num_gpus": 0.2,
        "num_workers": 1,
        "lr": tune.grid_search([0.01, 0.001, 0.0001]),
        "framework": "torch"
    },
)
"""

ray.shutdown()