import collections
import seaborn as sns
import pandas as pd
import os
from matplotlib import pyplot as plt
import matplotlib.transforms as mtrans


# ====================== Configs ====================== 
graph_name = "multi-level_evaluation"
plot_type = "training"
#plot_type = "evaluation"
smooth = True
# ====================== Configs ======================



if plot_type == "training":
    path = "./results/training_csv/plot_buffer/"
    reward_column = "episode_reward_mean"
    iterations_column = "training_iteration"
elif plot_type == "evaluation":
    path = "./results/evaluation_csv/plot_buffer/"
    reward_column = "Reward"
    iterations_column = "Episode"

csvFiles = os.listdir(path)
"""
tempVar = csvFiles[3]
csvFiles[3] = csvFiles[2]
csvFiles[2] = tempVar
"""
first_pass = True
for file in csvFiles:
    data = pd.read_csv(path + file)
    
    data = data[[reward_column,iterations_column]]
    #get moving average od reward_column on data
    if smooth:
        data[reward_column] = data[reward_column].rolling(window=10).mean()
    data.rename(columns={reward_column: file.split('.')[0]}, inplace=True)
    if first_pass: 
        data_final = data
        first_pass = False
    else:
        data_final = data_final.append(data)


data_final = data_final.melt(iterations_column, var_name='Agent', value_name='Avg. Reward')

g = sns.lineplot(x=iterations_column, y="Avg. Reward", hue='Agent', data=data_final)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.savefig("results/graphs/" + graph_name + '.png', bbox_inches='tight')
