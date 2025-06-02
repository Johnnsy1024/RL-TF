import gym
from flags import args

epsilon_end = args.epsilon_end
epsilon_start = args.epsilon_start
epsilon_decay = args.epsilon_decay
gamma = args.gamma
n_episodes = args.n_episodes
alpha = args.alpha
method = args.method
env = gym.make("CliffWalking-v0")
