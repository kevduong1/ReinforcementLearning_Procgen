from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

# https://docs.ray.io/en/latest/rllib-models.html
class PPO(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(PPO, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()

"""
Register custom PPO model and set up configs
"""
# https://github.com/ray-project/ray/blob/master/rllib/examples/fractional_gpus.py
class PPO_Model():
    def __init__(self):
        
        ModelCatalog.register_custom_model("PPO_c", PPO)

        self.config ={
            "env": "procgen_env_wrapper",
            "framework": "tf",
            "model": { "custom_model": "PPO_c", "framestack": True},

            "use_critic": True,
            "use_gae": True,
            "lambda": 1,
            "clip_rewards": True,
            "clip_param": 0.3,
            "kl_coeff": 0.2,
            "sgd_minibatch_size": 128,
            "shuffle_sequences": True,
            "kl_target": 0.01,

            "num_gpus": 1,
            "num_workers": 5,
            "num_envs_per_worker": 1,

            #"rollout_fragment_length": 32,
            #"train_batch_size": 512,
            "timesteps_per_iteration": 1000000,
        }

    def get_config(self):
        return self.config