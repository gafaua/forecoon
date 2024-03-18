from datetime import datetime
from pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset

from lib.utils.dataset import SequenceTyphoonDataset as STD
from lib.utils.dataset import DatasetFromSubset, NestedDigitalTyphoonDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np

from lib.utils.fisheye import FishEye

IMAGE_DIR="/fs9/gaspar/data/WP/image/"
METADATA_DIR="/fs9/gaspar/data/WP/metadata/"
METADAT_JSON="/fs9/gaspar/data/WP/metadata.json"

def get_simple_dataloader(args):
    dataset = DigitalTyphoonDataset(
        image_dir=IMAGE_DIR,
        metadata_dir=METADATA_DIR,
        metadata_json=METADAT_JSON,
        #get_images_by_sequence=True,
        labels=("grade"),
        split_dataset_by="sequence",
        filter_func= lambda x: x.grade() < 6,
        ignore_list=[],
        transform=None,
        verbose=False
    )
    train_transforms = T.Compose([
        T.ToTensor(),
        #T.Resize(256),
        #T.RandomResizedCrop(256, (0.5,1)),
        T.CenterCrop(256),
        # T.RandomApply([T.GaussianBlur(3, [.1, 3.])], p=0.5),
        # T.RandomSolarize(threshold=0.5,p=0.5),
        #T.RandAugment(),
        # T.RandomHorizontalFlip(p=0.5),
        # T.RandomVerticalFlip(p=0.5),
        #T.Normalize(mean=269.15, std=24.14),
    ])

    val_transforms = T.Compose([
        T.ToTensor(),
        #T.Resize(256),
        T.CenterCrop(256),
    ])

    def transform_func(obj, trans):
        img_range = [150, 350]
        img, labels = obj
        img = img.clip(img_range[0], img_range[1])
        img = (img - img_range[0])/(img_range[1]-img_range[0])

        return trans(img.astype(np.float32)), labels-2

    def get_tranform_func(split):
        if split == "train":
            return lambda x: transform_func(x, trans=train_transforms)
        else:
            return lambda x: transform_func(x, trans=val_transforms)

    train, val, test = dataset.random_split([0.7, 0.15, 0.15], split_by="sequence")
    train = DatasetFromSubset(train, get_tranform_func("train"))
    val   = DatasetFromSubset(val, get_tranform_func("val"))
    test   = DatasetFromSubset(test, get_tranform_func("val"))

    print(f"\n{len(train)} train images")
    print(f"{len(val)} val images")
    print(f"{len(test)} test images")

    train_loader = DataLoader(train,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)

    return train_loader, val_loader, test_loader


def get_temporal_sequence_dataloader(args):
    train_transforms = T.Compose([
        T.ToTensor(),
        FishEye(256, 0.2),
        #T.Resize(256),
        #T.RandomResizedCrop(256, (0.5,1)),
        #T.CenterCrop(256),
        # T.RandomApply([T.GaussianBlur(3, [.1, 3.])], p=0.5),
        # T.RandomSolarize(threshold=0.5,p=0.5),
        #T.RandAugment(),
        # T.RandomHorizontalFlip(p=0.5),
        # T.RandomVerticalFlip(p=0.5),
        #T.Normalize(mean=269.15, std=24.14),
    ])

    val_transforms = T.Compose([
        T.ToTensor(),
        FishEye(256, 0.2),
        #T.Resize(256),
        #T.CenterCrop(256),
    ])

    def transform_func(obj, trans):
        img_range = [150, 350]
        img, labels = obj
        img = img.clip(img_range[0], img_range[1])
        img = (img - img_range[0])/(img_range[1]-img_range[0])

        y, m, d, h = labels
        label = datetime(year=y, month=m, day=d, hour=h)

        return trans(img.astype(np.float32)), label

    def get_tranform_func(split):
        if split == "train":
            return lambda x: transform_func(x, trans=train_transforms)
        else:
            return lambda x: transform_func(x, trans=val_transforms)

    dataset = NestedDigitalTyphoonDataset(
        image_dir=IMAGE_DIR,
        metadata_dir=METADATA_DIR,
        metadata_json=METADAT_JSON,
        labels=("year", "month", "day", "hour"),
        split_dataset_by="sequence",
        filter_func= lambda x: x.grade() < 6 and x.mask_1() == 0 and x.year() > 1990,
        ignore_list=[],
        transform=get_tranform_func("train"),
        verbose=False
    )

    train, val, test = dataset.random_split([0.7, 0.15, 0.15], split_by="sequence")
    # train = DatasetFromSubset(train, None)
    # val   = DatasetFromSubset(val, None)
    # test   = DatasetFromSubset(test, None)

    print(f"\n{len(train)} train sequences")
    print(f"{len(val)} val sequences")
    print(f"{len(test)} test sequences")

    train_loader = DataLoader(train,
                              batch_size=1,
                              shuffle=True,
                              num_workers=0,
                              collate_fn=lambda x:x)
    val_loader = DataLoader(val,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=lambda x:x)
    test_loader = DataLoader(test,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=lambda x:x)

    return train_loader, val_loader, test_loader


def get_TS_dataloader(args):
    # transforms = T.Compose([
    #     T.ToTensor(),
    #     FishEye(256, 0.2),
    # ])

    # def transform_func(obj, trans=transforms):
    #     img_range = [150, 350]
    #     img, labels = obj
    #     img = img.clip(img_range[0], img_range[1])
    #     img = (img - img_range[0])/(img_range[1]-img_range[0])

    #     y, m, d, h = labels
    #     label = datetime(year=y, month=m, day=d, hour=h)

    #     return trans(img.astype(np.float32)), label

    #TODO make these arguments
    dataset = STD(labels=args.labels,#["month", "day", "hour", "pressure", "wind"],
                preprocessed_path=args.preprocessed_path,
                x=args.labels_input,#[0,1,2,3,4],
                y=args.labels_output,#[3,4],
                num_inputs=args.num_inputs,
                num_preds=args.num_outputs,
                interval=args.interval,
                filter_func= lambda x: x.grade() < 6,
                )

    train, val, test = dataset.random_split([0.7, 0.15, 0.15], split_by="sequence")

    print(f"\n{len(train)} train sequences")
    print(f"{len(val)} val sequences")
    print(f"{len(test)} test sequences")

    train_loader = DataLoader(train,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)

    return train_loader, val_loader, test_loader



experiments = dict(
    simple=get_simple_dataloader,
    temporal=get_temporal_sequence_dataloader,
    ts=get_TS_dataloader,
)


def get_dataloaders(args):
    return experiments[args.experiment](args)