from os import makedirs
from pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset
import torch
from tqdm import tqdm
import numpy as np
from lib.models.feature_extractors import get_resnet18
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


from lib.utils.dataloaders import IMAGE_DIR, METADATA_DIR, METADAT_JSON

class SimpleSeqDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = DigitalTyphoonDataset(
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

    def __getitem__(self, index):
        return self.dataset.get_ith_sequence(index)
    
    def __len__(self):
        return len(self.dataset)

def preprocess_images_sequences(model, out_dir, transform):
    makedirs(out_dir, exist_ok=True)
    print(f"Writing feature files to {out_dir}")
    dataset = SimpleSeqDataset()
    loader = DataLoader(dataset,
                        batch_size=1,
                        num_workers=16,
                        shuffle=False,
                        collate_fn=lambda x: x)

    model.eval()
    model = model.to("cuda")
    with torch.no_grad():
        for seq in tqdm(loader):
            seq = seq[0]
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
