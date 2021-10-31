from ray.rllib.models import ModelCatalog
from Models.Dqn_Model import DQN

"""
Config for procgen environment
"""
class procGenConfig():
    def __init__(self,type):
        self.type = type
        self.config = {
        "num_levels" : 1,  # The number of unique levels that can be generated. Set to 0 to use unlimited levels.
        "env_name" : "coinrun",  # Name of environment, or comma-separate list of environment names to instantiate as each env in the VecEnv
        "start_level" : 0,  # The lowest seed that will be used to generated levels. 'start_level' and 'num_levels' fully specify the set of possible levels
        "paint_vel_info" : False,  # Paint player velocity info in the top left corner. Only supported by certain games.
        "use_generated_assets" : False,  # Use randomly generated assets in place of human designed assets
        "center_agent" : True,  # Determines whether observations are centered on the agent or display the full level. Override at your own risk.
        "use_sequential_levels" : False,  # When you reach the end of a level, the episode is ended and a new level is selected. If use_sequential_levels is set to True, reaching the end of a level does not end the episode, and the seed for the new level is derived from the current level seed. If you combine this with start_level=<some seed> and num_levels=1, you can have a single linear series of levels similar to a gym-retro or ALE game.
        "distribution_mode" : "easy",  # What variant of the levels to use, the options are "easy", "hard", "extreme", "memory", "exploration". All games support "easy" and "hard", while other options are game-specific. The default is "hard". Switching to "easy" will reduce the number of timesteps required to solve each game and is useful for testing or when working with limited compute resources. NOTE : During the evaluation phase (rollout), this will always be overriden to "easy"
        "return_min": 0, # Minimum possible reward for the environment
        "return_blind": 0, # Reward obtained when the agent has no access to observations
        "return_max": 0, # Maximum possible reward for the environment
        "render_mode":"human"
    }

    def get_config(self):
        if self.type == "training":
            self.config["render_mode"] = "rgb_array" # figure out if there's a none render mode
        elif self.type == "evaluation":
            self.config["render_mode"] = "rgb_array"
        elif self.type == "display":
            self.config["render_mode"] = "human"
        
        return self.config

"""
Register custom DQN model and set up configs
"""
class DQN_Model():
    def __init__(self):
        
        ModelCatalog.register_custom_model("DQN_c", DQN)

        self.config ={
        "env": "procgen_env_wrapper",
        "framework": "tf",
        "model": { "custom_model": "DQN_c"},

        "gamma": 0.99,
        "double_q": True,
        "dueling": True,  
        "prioritized_replay": True,
        "prioritized_replay_alpha": 0.5,
        "prioritized_replay_beta": 0.4,
        "final_prioritized_replay_beta": 1.0,
        #"prioritized_replay_beta_annealing_timesteps": 2000000,

        #"num_gpus": 1,
        #"num_workers": 0,
        #"num_gpus_per_worker": 0.2,
        #"num_envs_per_worker": 1,
        #"rollout_fragment_length": 32,
        #"train_batch_size": 512,
        #"target_network_update_freq": 50000,
        #"timesteps_per_iteration": 25000,
        #"learning_starts": 80000, # what does this parameter do?

        "render_env": False
        }

    def get_config(self):
        return self.config
