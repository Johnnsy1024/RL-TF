import gym
from flags import args

actor_lr = args.actor_lr
critic_lr = args.critic_lr
num_episodes = args.num_episodes
hidden_dim = args.hidden_dim
gamma = args.gamma
lmbda = args.lmbda
epochs = args.epochs
eps = args.eps

env = gym.make('CliffWalking-v0')
state_dim = env.observation_space.n
action_dim = env.action_space.n