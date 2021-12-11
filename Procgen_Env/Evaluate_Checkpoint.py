import os
import pickle
import PIL
import pandas as pd
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer
from Util.procgen_wrapper import ProcgenEnvWrapper
from Models.Dqn_Model import DQN
from Models.PPO_Model import PPO
from ray.rllib.models import ModelCatalog


# ================= configs =================
algorithmType = "PPO"
training_steps = "200" # in millions
level_type = "multi-lvl"
#level_type = "1-lvl"
#level_type = "500-lvl"
#level_type = "hybrid-lvl"
# ===========================================
# for creating csv
data_frame = pd.DataFrame(columns=['Episode', 'Reward'])

file_result_name = algorithmType + "_" + level_type + "_" + training_steps + "mil"

env = ProcgenEnvWrapper("training")

pathstring = "{0}/{0}_{1}_200mil/".format(algorithmType,level_type)

# TODO add pipelining code for evaluation
config_path = os.path.join("checkpoint/" + pathstring, "params.pkl")
with open(config_path, "rb") as f:
    config = pickle.load(f)


config['num_gpus']=0
config['num_workers']=1

if algorithmType == "DQN":
    ModelCatalog.register_custom_model("DQN_c", DQN)
    Trainer = DQNTrainer
elif algorithmType == "PPO":
    ModelCatalog.register_custom_model("PPO_c", PPO)
    Trainer = PPOTrainer



RLAgent = Trainer(env="procgen_env_wrapper", config=config)
RLAgent.restore("checkpoint/" + pathstring + "checkpoint_000{0}/checkpoint-{0}".format(training_steps))




frame_buffer = [] # for gif

obs = env.reset()
     
num_games = 0
total_reward = 0
best_score = 0
best_frame_list = []
episode_score = 0   
while num_games < 500:
    action = RLAgent.compute_action(obs)
    obs, reward, done, info = env.step(action)

    episode_score += reward
    frame_buffer.append(PIL.Image.fromarray(env.render(mode='rgb_array')))

    if done:

        num_games += 1
        total_reward += episode_score
        avg = total_reward / num_games
        # print the data type of num_games

        #data_frame = data_frame.append({'Episode': num_games, 'Reward': avg}, ignore_index=True)
        if episode_score > best_score:
            best_score = episode_score
            best_frame_list = frame_buffer + best_frame_list
            print(str(num_games) + " | episode score: " + str(episode_score) + "| avg score: " + str(avg) + " - new best score=" + str(best_score))
        else:
            print(str(num_games) + " | episode score: " + str(episode_score) + " | avg score: " + str(avg))
        # reset the environment
        env.reset()
        frame_buffer = []
        episode_score = 0


#data_frame['Episode'] = data_frame['Episode'].astype(int)

#data_frame.to_csv(f'results/evaluation_csv/' + file_result_name, index=False)
best_frame_list[0].save(f"results/gif_folder/" + file_result_name + ".gif", save_all=True, append_images=best_frame_list[1:700], duration=1, loop=0)

