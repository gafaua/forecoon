from lib.models.feature_extractors import get_moco_encoder, get_resnet18
from lib.utils.preprocessing import preprocess_images_sequences
import torchvision.transforms as T
import torch
import numpy as np

def main():
    transforms = T.Compose([
        T.CenterCrop(384),
        T.Resize(224)
    ])

    def transform_func(img):
        # APPLIED TO ALL IMAGES IN THE SEQUENCE AT ONCE
        img_range = [150, 350]
        img = (img - img_range[0])/(img_range[1]-img_range[0])
        # img = img.astype(np.float32)
        # img = torch.from_numpy(img).to("cuda").unsqueeze(0)

        return transforms(img)

    model = get_moco_encoder("resnet18", "models/moco/moco_test/checkpoint_15960.pth")#get_resnet18()
    print(f"Encoder ready, model with {sum(p.numel() for p in model.parameters()):,} parameters")

    preprocess_images_sequences(model, "resnet18_moco", transform_func)

if __name__ == "__main__":
    main()