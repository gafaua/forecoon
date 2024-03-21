# TODO add main training pipeline

from pprint import pprint
import random

import numpy as np
from lib.trainers.image import ImageTrainer
from lib.trainers.moco import MocoTrainer
from lib.trainers.temporal import TemporalTrainer
from lib.trainers.time_series import TimeSeriesTrainer
from lib.trainers.time_series_date import TimeSeriesDateTrainer
from lib.utils.dataloaders import get_dataloaders
from parse import parse_args
import torch

def train():
    args = parse_args()
    pprint(args.__dict__)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    train_loader, val_loader, test_loader = get_dataloaders(args)


    print("\nDATALOADERS LOADED, STARING TRAINING\n")

    if args.experiment == "simple":
        ImageTrainer(train_loader,
                    val_loader,
                    args).train()
    elif args.experiment == "temporal":
        TemporalTrainer(train_loader,
                        val_loader,
                        args).train()
    elif args.experiment == "ts":
        if args.use_date:
            TimeSeriesDateTrainer(train_loader,
                            val_loader,
                            args).train()
        else:
            TimeSeriesTrainer(train_loader,
                            val_loader,
                            args).train()
    elif args.experiment == "moco":
        MocoTrainer(train_loader,
                    val_loader,
                    args).train()

    else:
        raise NotImplementedError()


if __name__ == "__main__":
    train()