import os
import ray
import pickle
import PIL
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from matplotlib import animation
import matplotlib.pyplot as plt
import gym
from procgen import ProcgenGym3Env

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


# print file name of files in checkpoint folder of this current folder
"""
for file in os.listdir("./checkpoint/DQN_2021-10-28_15-53-15/DQN_CartPole-v0_12604_00000_0_2021-10-28_15-53-15/checkpoint_000005"):
    print(file)
"""

config_path = os.path.join("./checkpoint/DQN_2021-10-28_15-53-15/DQN_CartPole-v0_12604_00000_0_2021-10-28_15-53-15/", "params.pkl")
with open(config_path, "rb") as f:
    config = pickle.load(f)

config['num_gpus']=0
config['num_workers']=1

Trainer = DQNTrainer
RLAgent = Trainer(env="CartPole-v0", config=config)
RLAgent.restore("./checkpoint/DQN_2021-10-28_15-53-15/DQN_CartPole-v0_12604_00000_0_2021-10-28_15-53-15/checkpoint_000005/checkpoint-5")
env = gym.make('CartPole-v0')
num_steps = 0   # one num_step includes all agent steps 
frames = []
obs = env.reset()
total_reward = 0
total_runs = 0
while num_steps < 500:
        num_steps += 1
        frames.append(env.render(mode="rgb_array"))
        #action = RLAgent.compute_single_action(obs)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            total_runs += 1
            env.reset()


print(total_reward/total_runs)


#save_frames_as_gif(frames)
