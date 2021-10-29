from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

tf1, tf, tfv = try_import_tf()

class DQN(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name="atari_model"):
        super(DQN, self).__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        inputs = tf.keras.layers.Input(shape=(64,64,3), name='observations')
        #inputs2 = tf.keras.layers.Input(shape=(1,), name='agent_indicator')
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
        #concat_layer = tf.keras.layers.Concatenate()([layer4, inputs2])
        layer5 = tf.keras.layers.Dense(
                512,
                activation="relu",
                kernel_initializer=normc_initializer(1.0))(layer4) # changed
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