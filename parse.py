# TODO add argument parser

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    # Dataset
    parser.add_argument("--data_path", type=str)


    parser.add_argument("--backbone", type=str)

    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--save_dir", type=str)

    parser.add_argument("--use_wandb", type=bool)

    args = parser.parse_args()

    # Post treatment on arguments

    return args
