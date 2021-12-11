import ray
import os
import pickle
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from Models.PPO_Model import PPO_Model
from Util.procgen_wrapper import ProcgenEnvWrapper
from Models.Dqn_Model import DQN_Model
from ray.rllib.models import ModelCatalog
from Models.Dqn_Model import DQN

# Functions for loading checkpoints for training
def load_checkpoint(algorithmType='DQN', level_type='1-lvl', training_steps=100):
    
    pathstring = "{0}/{0}_{1}_200mil/".format(algorithmType,level_type)
    return "checkpoint/" + pathstring + "checkpoint_000{0}/checkpoint-{0}".format(training_steps)
    # set algorithm





algorithmType = "DQN"
load = True


# Import environment (configs for the environments are changed depending on type is passed into the constructor)
env = ProcgenEnvWrapper("training")

# Stop conditions:
stop = {
    "agent_timesteps_total": 102000000,
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
    restore=load_checkpoint(),
    checkpoint_at_end=True,
    checkpoint_freq=50,
    local_dir="./Procgen_Env/checkpoint" 

    )


ray.shutdown()


