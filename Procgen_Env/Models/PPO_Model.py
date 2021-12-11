from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

tf1, tf, tfv = try_import_tf()
"""
class PPO(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name="PPO_Model"):
        super(PPO, self).__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        inputs = tf.keras.layers.Input(shape=(64,64,3), name='observations')
        # Convolutions on the frames on the screen
        layer1 = tf.keras.layers.Conv2D(
                32,
                [8, 8],
                strides=(4, 4),
                activation="relu",
                data_format='channels_last')(inputs)
        layer2 = tf.keras.layers.Conv2D(
                64,
                [4, 4],
                strides=(2, 2),
                activation="relu",
                data_format='channels_last')(layer1)
        layer3 = tf.keras.layers.Conv2D(
                64,
                [3, 3],
                strides=(1, 1),
                activation="relu",
                data_format='channels_last')(layer2)
        layer4 = tf.keras.layers.Flatten()(layer3)
        layer5 = tf.keras.layers.Dense(
                512,
                activation="relu",
                kernel_initializer=normc_initializer(1.0))(layer4) 
        action = tf.keras.layers.Dense(
                num_outputs,
                activation="linear",
                name="actions",
                kernel_initializer=normc_initializer(0.01))(layer5)
        value_out = tf.keras.layers.Dense(
                1,
                activation=None,
                name="value_out",
                kernel_initializer=normc_initializer(0.01))(layer5)
        self.base_model = tf.keras.Model(inputs, [action, value_out])
        #self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

"""
# https://docs.ray.io/en/latest/rllib-models.html
# should of used an image net
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


#Register custom PPO model and set up configs

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
            "shuffle_sequences": True,
            "kl_target": 0.01,

            "num_sgd_iter":30,
            "num_gpus": 1,
            "num_workers": 23,
            "num_envs_per_worker": 1,
            "rollout_fragment_length": 200,
            "sgd_minibatch_size": 128,
            "train_batch_size": 4000,
            "timesteps_per_iteration": 1000000,
        }

    def get_config(self):
        return self.config