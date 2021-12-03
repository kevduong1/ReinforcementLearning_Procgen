import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from Models.PPO_Model import PPO_Model
from Util.procgen_wrapper import ProcgenEnvWrapper
from Models.Dqn_Model import DQN_Model

# set algorithm
algorithmType = "DQN"

# Import environment (configs for the environments are changed depending on type is passed into the constructor)
env = ProcgenEnvWrapper("training")

# Register the model, and get config files


# Stop conditions:
stop = {
    "agent_timesteps_total": 200000000,
    }

if algorithmType == "DQN":
    config = DQN_Model().get_config()
    trainer = DQNTrainer
elif algorithmType == "PPO":
    config = PPO_Model().get_config()
    trainer = "PPO"


ray.init()

tune.run(
    trainer,
    config=config,
    stop=stop,
    checkpoint_at_end=True,
    checkpoint_freq=50,
    local_dir="./Procgen_Env/checkpoint"    #TODO: set up a path to folder for checkpoints
    )


ray.shutdown()
