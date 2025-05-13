import gym
import tensorflow as tf
from model import build_actor, build_critic
class SAC:
    def __init__(self, env: gym.Env, state_dim: int, action_dim: int, hidden_dim: int, actor_lr: float, critic_lr: float, gamma: float, tau: float, alpha_init: float, target_entropy: float):
        self.acotr = build_actor(state_dim, hidden_dim, action_dim, "continuous", env.action_space.high[0])
        
        self.critic_1 = build_critic(state_dim, hidden_dim, action_dim)
        self.critic_2 = build_critic(state_dim, hidden_dim, action_dim)
        
        self.target_critic_1 = build_critic(state_dim, hidden_dim, action_dim)
        self.target_critic_2 = build_critic(state_dim, hidden_dim, action_dim)
        
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_1_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.critic_2_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha_init
        self.target_entropy = target_entropy
        
        self.log_alpha = tf.Variable(tf.math.log(alpha_init), trainable=True)
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
    
    def take_action(self, state: tf.Tensor):
        state = tf.reshape(state, [-1, self.state_dim])
        action, _ = self.actor(state)
        return action.numpy()[0]