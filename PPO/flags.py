from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument("--actor_lr", type=float, help="actor learning rate", default=1e-3)
arg_parser.add_argument("--critic_lr", type=float, help="critic learning rate", default=1e-2)
arg_parser.add_argument("--num_episodes", type=int, help="number of episodes", default=3000)
arg_parser.add_argument("--hidden_dim", type=int, help="hidden dimension", default=128)
arg_parser.add_argument("--gamma", type=float, help="discount factor", default=0.9)
arg_parser.add_argument("--lmbda", type=float, help="lambda for GAE", default=0.9)
arg_parser.add_argument("--epochs", type=int, help="number of epochs", default=50)
arg_parser.add_argument("--eps", type=float, help="epsilon for clipping", default=0.02)
args = arg_parser.parse_args()