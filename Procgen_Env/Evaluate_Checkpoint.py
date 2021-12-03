import os
import pickle
import PIL
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer
from Util.procgen_wrapper import ProcgenEnvWrapper
from Models.Dqn_Model import DQN
from Models.PPO_Model import PPO
from ray.rllib.models import ModelCatalog


algorithmType = "PPO"
training_steps = "50" # in millions
level_type = "multi-level"
#level_type = "one-level"


# TODO plot graphs for rewards


env = ProcgenEnvWrapper("training")

pathstring = "{0}/{0}_{1}_200mil/".format(algorithmType,level_type)

# TODO add pipelining code for evaluation
config_path = os.path.join("./Procgen_Env/checkpoint/" + pathstring, "params.pkl")
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
RLAgent.restore("./Procgen_Env/checkpoint/" + pathstring + "checkpoint_0000{0}/checkpoint-{0}".format(training_steps))



frame_buffer = [] # for gif

obs = env.reset()
     
num_games = 0
total_reward = 0
best_score = 0
best_frame_list = []
episode_score = 0   
while num_games < 20:
    action = RLAgent.compute_action(obs)
    obs, reward, done, info = env.step(action)

    episode_score += reward
    frame_buffer.append(PIL.Image.fromarray(env.render(mode='rgb_array')))

    if done:

        num_games += 1
        total_reward += episode_score
        avg = total_reward / num_games
        if episode_score > best_score:
            best_score = episode_score
            best_frame_list = []
            best_frame_list.extend(frame_buffer)
            print(str(num_games) + " | episode score: " + str(episode_score) + "| avg score: " + str(avg) + " - new best score=" + str(best_score))
        else:
            print(str(num_games) + " | episode score: " + str(episode_score) + " | avg score: " + str(avg))


        
        # reset the environment
        env.reset()
        frame_buffer = []
        episode_score = 0

# Save gif
# TODO: add pipeline for saving code to location with automated naming
best_frame_list[0].save(f"procgen_steps8.gif", save_all=True, append_images=best_frame_list[1:], duration=3, loop=0)

