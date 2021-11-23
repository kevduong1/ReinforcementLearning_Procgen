import os
import pickle
import PIL
from ray.rllib.agents.dqn import DQNTrainer
from Util.procgen_wrapper import ProcgenEnvWrapper
from Models.Dqn_Model import DQN
from ray.rllib.models import ModelCatalog

# TODO plot graphs for rewards

ModelCatalog.register_custom_model("DQN_c", DQN)
env = ProcgenEnvWrapper("display")

pathstring = "DQN_100_mil/DQN/"

# TODO add pipelining code for evaluation
config_path = os.path.join("./Procgen_Env/checkpoint/" + pathstring, "params.pkl")
with open(config_path, "rb") as f:
    config = pickle.load(f)

# Do I need to specify these congfigs if im restoring from pickle
config['num_gpus']=0
config['num_workers']=1

Trainer = DQNTrainer
RLAgent = Trainer(env="procgen_env_wrapper", config=config)
RLAgent.restore("./Procgen_Env/checkpoint/" + pathstring + "checkpoint_000100/checkpoint-100")


num_steps = 0
frame_list = []
obs = env.reset()
total_reward = 0
total_runs = 0
while num_steps < 1000:
        num_steps += 1
        frame_list.append(PIL.Image.fromarray(env.render(mode='rgb_array')))
        action = RLAgent.compute_single_action(obs)
        #action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(num_steps)
        total_reward += reward
        if done:
            total_runs += 1
            env.reset()

# Save gif
# TODO: add pipeline for saving code to location with automated naming
frame_list[0].save(f"procgen_steps3.gif", save_all=True, append_images=frame_list[1:], duration=3, loop=0)

