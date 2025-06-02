import gym
from flags import args

hidden_dim = args.hidden_dim
actor_lr = args.actor_lr
critic_lr = args.critic_lr
sigma = args.sigma
sigma_end = args.sigma_end
tau = args.tau
batch_size = args.batch_size
buffer_size = args.buffer_size
gamma = args.gamma
num_episodes = args.num_episodes

env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
