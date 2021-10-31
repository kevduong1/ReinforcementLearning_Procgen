import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from Util.procgen_wrapper import ProcgenEnvWrapper
from Util.Configs import DQN_Model

# Import environment (configs for the environments are changed depending on type is passed into the constructor)
env = ProcgenEnvWrapper("training")

# Register the model, and get config files
config = DQN_Model().get_config()

# Stop conditions:
stop = {
    "agent_timesteps_total": 5000
    }


ray.init()

# TODO figure out memory leak issue
# TODO figure out how to take advantage of vectorized training/make training more efficient with apex?
# TODO figure out how to get gpu
tune.run(
    DQNTrainer,
    config=config,
    stop=stop,
    checkpoint_at_end=True,
    checkpoint_freq=5,
    local_dir="./checkpoint"    #TODO: set up a path to folder for checkpoints
    )


ray.shutdown()
