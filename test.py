from lib.models.feature_extractors import get_resnet18
from lib.utils.preprocessing import preprocess_images_sequences


def main():
    def transform_func(img):
        img_range = [150, 350]
        img = (img - img_range[0])/(img_range[1]-img_range[0])

        return img

    model = get_resnet18()

    preprocess_images_sequences(model, "resnet18_imagenet", transform_func)

if __name__ == "__main__":
    main()