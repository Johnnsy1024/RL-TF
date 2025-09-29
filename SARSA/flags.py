from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument("--epsilon_end", type=float, help="epsilon end", default=0.01)
arg_parser.add_argument("--epsilon_start", type=float, help="epsilon start", default=0.95)
arg_parser.add_argument("--epsilon_decay", type=int, help="epsilon decay", default=200)
arg_parser.add_argument("--gamma", type=float, help="gamma", default=0.95)
arg_parser.add_argument("--n_episodes", type=int, help="number of episodes", default=1000)
arg_parser.add_argument("--alpha", type=float, help="alpha", default=0.05)
arg_parser.add_argument(
    "--method", type=str, help="method to use (sarsa or q-learning)", default="sarsa"
)
args = arg_parser.parse_args()
