from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument("--hidden_dim", type=int, help="hidden_dim", default=256)
arg_parser.add_argument("--actor_lr", type=float, help="actor_lr", default=1e-3)
arg_parser.add_argument("--critic_lr", type=float, help="critic_lr", default=2e-3)
arg_parser.add_argument("--sigma", type=float, help="normal noise std", default=0.01)
arg_parser.add_argument(
    "--sigma_end", type=float, help="final normal noise std", default=1e-4
)
arg_parser.add_argument(
    "--noise_type",
    type=str,
    help="noise type",
    choices=["normal", "ou"],
    default="normal",
)
arg_parser.add_argument("--policy_noise", type=float, help="policy noise", default=0.2)
arg_parser.add_argument("--noise_clip", type=float, help="noise clip", default=0.5)
arg_parser.add_argument("--policy_delay", type=int, help="policy_delay", default=2)
arg_parser.add_argument("--tau", type=float, help="soft update rate", default=0.005)
arg_parser.add_argument("--batch_size", type=int, help="batch_size", default=64)
arg_parser.add_argument("--buffer_size", type=int, help="buffer_size", default=1000000)
arg_parser.add_argument("--gamma", type=float, help="discount factor", default=0.99)
arg_parser.add_argument("--num_episodes", type=int, help="num_episodes", default=2000)
args = arg_parser.parse_args()
