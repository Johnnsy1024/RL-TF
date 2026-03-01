import gym
from flags import args

lr = args.lr
gamma = args.gamma
hidden_dim = args.hidden_dim
num_episodes = args.num_episodes
env_name = args.env_name

env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
