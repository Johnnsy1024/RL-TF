from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
arg_parser.add_argument("--gamma", type=float, help="discount factor", default=0.99)
arg_parser.add_argument("--hidden_dim", type=int, help="hidden dimension", default=128)
arg_parser.add_argument(
    "--num_episodes", type=int, help="number of episodes", default=1000
)
arg_parser.add_argument(
    "--env_name", type=str, help="environment name", default="CartPole-v1"
)

args = arg_parser.parse_args()
