import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from Util.procgen_wrapper import ProcgenEnvWrapper
from Models.Dqn_Model import DQN_Model

# Import environment (configs for the environments are changed depending on type is passed into the constructor)
env = ProcgenEnvWrapper("training")

# Register the model, and get config files
config = DQN_Model().get_config()

# Stop conditions:
stop = {
    "agent_timesteps_total": 20000000
    }


ray.init()

tune.run(
    DQNTrainer,
    config=config,
    stop=stop,
    checkpoint_at_end=True,
    checkpoint_freq=50,
    local_dir="./checkpoint"    #TODO: set up a path to folder for checkpoints
    )


ray.shutdown()
