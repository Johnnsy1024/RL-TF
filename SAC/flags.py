from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument("--actor_lr", type=float, default=3e-4)
arg_parser.add_argument("--critic_lr", type=float, default=3e-3)
arg_parser.add_argument("--alpha_lr", type=float, default=3e-4)
arg_parser.add_argument("--batch_size", type=int, default=128)
arg_parser.add_argument("--gamma", type=float, default=0.99)
arg_parser.add_argument("--tau", type=float, default=0.05)
arg_parser.add_argument("--hidden_dim", type=int, default=128)
arg_parser.add_argument("--buffer_size", type=int, default=1000000)
arg_parser.add_argument("--num_episodes", type=int, default=1000)
arg_parser.add_argument("--alpha_init", type=float, default=2.0)

args = arg_parser.parse_args()
