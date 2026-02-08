import gymnasium as gym
import minigrid
from flags import args

LR = args.lr
BUFFER_SIZE = args.buffer_size
HIDDEN_DIM = args.hidden_dim
EPSILON_START = args.epsilon_start
EPSILON_MIN = args.epsilon_end
EPSILON_DECAY = args.epsilon_decay
BATCH_SIZE = args.batch_size
TARGET_UPDATE_FREQ = args.target_update_freq
NUM_EPISODES = args.num_episodes
GAMMA = args.gamma
DQN_TYPE = args.dqn_type
env_name = args.env_name

env = gym.make(env_name)
STATE_IMG_SHAPE = env.observation_space["image"].shape
STATE_DIR_DIM = 1
STATE_DIR_CNT = env.observation_space["direction"].n
ACTION_DIM = env.action_space.n
