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
env_action_type = args.env_action_type
env_name = args.env_name

env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = (
    env.action_space.shape[0] if env_action_type == "continuous" else env.action_space.n
)
target_entropy = -env.action_space.shape[0] if env_action_type == "continuous" else -1
