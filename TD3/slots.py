import gym
from flags import args

HIDDEN_DIM = args.hidden_dim
ACTOR_LR = args.actor_lr
CRITIC_LR = args.critic_lr
SIGMA = args.sigma
SIGMA_END = args.sigma_end
TAU = args.tau
BATCH_SIZE = args.batch_size
BUFFER_SIZE = args.buffer_size
GAMMA = args.gamma
NUM_EPISODES = args.num_episodes
NOISE_TYPE = args.noise_type
POLICY_NOISE = args.policy_noise
NOISE_CLIP = args.noise_clip
POLICY_DELAY = args.policy_delay

env = gym.make("Pendulum-v1")
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
