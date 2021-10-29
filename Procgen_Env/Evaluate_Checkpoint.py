import os
import ray
import time
import pickle
import PIL
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from matplotlib import animation
import matplotlib.pyplot as plt
from procgen_wrapper import ProcgenEnvWrapper
import gym
from Models.Dqn_Model import DQN
from procgen import ProcgenGym3Env
from ray.rllib.models import ModelCatalog


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


#print(os.listdir("./Procgen_Env/checkpoint/DQN_2021-10-28_21-11-37/DQN_procgen_env_wrapper_8c3c9_00000_0_2021-10-28_21-11-37"))

ModelCatalog.register_custom_model("DQN_c", DQN)

config = {
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

        }

env = ProcgenEnvWrapper(config)
#env.seed(1)
config_path = os.path.join("./Procgen_Env/checkpoint/DQN_2021-10-28_22-43-06/DQN_procgen_env_wrapper_53929_00000_0_2021-10-28_22-43-06", "params.pkl")
with open(config_path, "rb") as f:
    config = pickle.load(f)

config['num_gpus']=0
config['num_workers']=1

Trainer = DQNTrainer
RLAgent = Trainer(env="procgen_env_wrapper", config=config)
RLAgent.restore("./Procgen_Env/checkpoint/DQN_2021-10-28_22-43-06/DQN_procgen_env_wrapper_53929_00000_0_2021-10-28_22-43-06/checkpoint_000030/checkpoint-30")
#env = gym.make('CartPole-v0')

num_steps = 0   # one num_step includes all agent steps 
frame_list = []
obs = env.reset()
total_reward = 0
total_runs = 0
while num_steps < 500:
        num_steps += 1
        frame_list.append(PIL.Image.fromarray(env.render(mode='rgb_array')))
        action = RLAgent.compute_single_action(obs)
        #action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            total_runs += 1
            env.reset()




frame_list[0].save(f"procgen_steps1.gif", save_all=True, append_images=frame_list[1:], duration=3, loop=0)
#save_frames_as_gif(frames)
