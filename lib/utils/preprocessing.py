from os import makedirs
from pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset
import torch
from tqdm import tqdm
import numpy as np
from lib.models.feature_extractors import get_resnet18
import torchvision.transforms as T

from lib.utils.dataloaders import IMAGE_DIR, METADATA_DIR, METADAT_JSON
from lib.utils.fisheye import FishEye


def preprocess_images_sequences(model, out_dir, transform):
    makedirs(out_dir, exist_ok=True)
    dataset = DigitalTyphoonDataset(
        image_dir=IMAGE_DIR,
        metadata_dir=METADATA_DIR,
        metadata_json=METADAT_JSON,
        get_images_by_sequence=True,
        labels=[],
        filter_func= None,
        ignore_list=[],
        transform=None,
        verbose=False
    )

    model.eval()
    model = model.to("cuda")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            seq = dataset.get_ith_sequence(idx)
            images = seq.get_all_images_in_sequence()
            names = np.array([str(image.image_filepath).split("/")[-1].split(".")[0] for image in images])

            images = torch.Tensor(np.array([image.image() for image in images]))
            if transform:
                images = transform(images)
            images = images.to("cuda").unsqueeze(1)

            features = model(images).cpu().numpy()
            np.savez(f"{out_dir}/{seq.sequence_str}", features, names)


if __name__ == "__main__":
    def transform_func(img):
        img_range = [150, 350]
        img = (img - img_range[0])/(img_range[1]-img_range[0])

        return img

    model = get_resnet18()

    preprocess_images_sequences(model, "test_preprop", transform_func)
