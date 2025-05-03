import gym
from flags import args


lr = args.lr
epsilon = args.epsilon
epsilon_min = args.epsilon_min
epsilon_decay = args.epsilon_decay
batch_size = args.batch_size
target_update_freq = args.target_update_freq
n_episodes = args.n_episodes
gamma = args.gamma
dqn_type = args.dqn_type

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n