# TODO add main training pipeline

from pprint import pprint
import random

import numpy as np
from lib.trainers.image import ImageTrainer
from parse import parse_args
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset

def train():
    args = parse_args()
    pprint(args.__dict__)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    transforms = T.Compose([
        T.ToTensor(),
        T.Resize(256),
        T.RandomApply([T.GaussianBlur(3, [.1, 2.])], p=0.5),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.Normalize(mean=269.15, std=24.14),
    ])

    def transform_func(obj):
        img, labels = obj
        return transforms(img.astype(np.float32)), labels-3

    dataset = DigitalTyphoonDataset(
        image_dir="/fs9/gaspar/data/WP/image/",
        metadata_dir="/fs9/gaspar/data/WP/metadata/",
        metadata_json="/fs9/gaspar/data/WP/metadata.json",
        #get_images_by_sequence=True,
        labels=("grade"),
        split_dataset_by="sequence",
        filter_func= lambda x: x.grade() > 2 and x.grade() < 6,
        ignore_list=[],
        transform=transform_func,
        verbose=False
    )

    train, val, test = dataset.random_split([0.7, 0.15, 0.15], split_by="sequence")
    train_loader = DataLoader(train,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)
    print("DATALOADERS LOADED, STARING TRAINING")

    ImageTrainer(train_loader,
                 val_loader,
                 args).train()


if __name__ == "__main__":
    train()