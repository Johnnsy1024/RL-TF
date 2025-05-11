from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument("--hidden_dim", type=int, help="hidden_dim", default=64)
arg_parser.add_argument("--actor_lr", type=float, help="actor_lr", default=3e-4)
arg_parser.add_argument("--critic_lr", type=float, help="critic_lr", default=3e-3)
arg_parser.add_argument("--sigma", type=float, help="normal noise std", default=1)
arg_parser.add_argument("--sigma_end", type=float, help="final normal noise std", default=1e-5)
arg_parser.add_argument("--tau", type=float, help="soft update rate", default=0.02)
arg_parser.add_argument("--batch_size", type=int, help="batch_size", default=64)
arg_parser.add_argument("--buffer_size", type=int, help="buffer_size", default=10000)
arg_parser.add_argument("--gamma", type=float, help="discount factor", default=0.98)
arg_parser.add_argument("--num_episodes", type=int, help="num_episodes", default=5000)
args = arg_parser.parse_args()