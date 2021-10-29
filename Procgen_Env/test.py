import gym
#env = gym.make(f"procgen:procgen-coinrun-v0")
# create cartpole environment
#env = gym.make("CartPole-v1")
env = gym.make("procgen:procgen-coinrun-v0",render_mode="rgb_array")
obs = env.reset()

for _ in range(100):
    print(env.render(mode="rgb_array"))
    obs, reward, done, info = env.step(env.action_space.sample()) # take a random action
    print(reward)
    #print(_)