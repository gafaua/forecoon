# TODO add argument parser

from argparse import ArgumentParser
from time import time


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--experiment", type=str)

    parser.add_argument("--backbone", type=str)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--hidden_dim", type=int)

    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--weight_decay", type=float, default=0)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--log_interval", type=int, default=10)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=str(time()))
    parser.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()

    # Post treatment on arguments

    return args
