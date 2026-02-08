from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument("--lr", type=float, help="epsilon end", default=0.001)
arg_parser.add_argument("--hidden_dim", type=int, help="hidden dimension", default=256)
arg_parser.add_argument(
    "--buffer_size", type=int, help="replay buffer size", default=100000
)
arg_parser.add_argument("--epsilon_start", type=float, help="epsilon start", default=0.95)
arg_parser.add_argument("--epsilon_end", type=float, help="epsilon end", default=0.1)
arg_parser.add_argument("--epsilon_decay", type=float, help="epsilon decay", default=200)
arg_parser.add_argument("--batch_size", type=int, help="batch size", default=128)
arg_parser.add_argument(
    "--target_update_freq", type=int, help="target update frequency", default=100
)
arg_parser.add_argument(
    "--num_episodes", type=int, help="number of episodes", default=5000
)
arg_parser.add_argument("--gamma", type=float, help="number of episodes", default=0.99)
arg_parser.add_argument(
    "--dqn_type",
    type=str,
    choices=["dqn", "double_dqn", "dueling_dqn"],
    help="dqn type",
    default="dqn",
)
arg_parser.add_argument(
    "--env_name", type=str, help="environment name", default="MiniGrid-Empty-5x5-v0"
)

args = arg_parser.parse_args()
