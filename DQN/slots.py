import gym
from flags import args

lr = args.lr
memory_size = args.memory_size
epsilon_start = args.epsilon_start
epsilon_min = args.epsilon_end
epsilon_decay = args.epsilon_decay
batch_size = args.batch_size
target_update_freq = args.target_update_freq
n_episodes = args.n_episodes
gamma = args.gamma
dqn_type = args.dqn_type
env_name = args.env_name

env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
