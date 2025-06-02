import gym
from flags import args

actor_lr = args.actor_lr
critic_lr = args.critic_lr
alpha_lr = args.alpha_lr
batch_size = args.batch_size
gamma = args.gamma
tau = args.tau
hidden_dim = args.hidden_dim
buffer_size = args.buffer_size
num_episodes = args.num_episodes
alpha_init = args.alpha_init


env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]
target_entropy = -env.action_space.shape[0]
