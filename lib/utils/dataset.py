from tracemalloc import start
import numpy as np
from torch import Tensor
import torch
from torch.utils.data import Dataset
from pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        data = self.subset[index]
        if self.transform:
            return self.transform(data)
        return data

    def __len__(self):
        return len(self.subset)


class TemporalSequencePairDataset(Dataset):
    def __init__(self, images, labels, transform) -> None:
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        size = len(self.images)
        #index2 = int(np.random.uniform() * np.min([6, (size - index)])) + index
        index2 = int(np.random.beta(2, 5) * np.min([6, (size - index)])) + index

        x1, lbl1 = self.images[index], self.labels[index]
        x2, lbl2 = self.images[index2], self.labels[index2]

        img1, dt1 = self.transform((x1, lbl1))
        img2, dt2 = self.transform((x2, lbl2))

        return img1, img2, (dt2-dt1).total_seconds()/3600
    
    def __len__(self):
        return len(self.images)


class NestedDigitalTyphoonDataset(DigitalTyphoonDataset):
    def __init__(self,
                 image_dir: str,
                 metadata_dir: str,
                 metadata_json: str,
                 labels,
                 split_dataset_by='image',
                 spectrum='Infrared',
                 load_data_into_memory=False,
                 ignore_list=None,
                 filter_func=None,
                 transform_func=None,
                 transform=None,
                 verbose=False) -> None:
        super().__init__(image_dir, metadata_dir, metadata_json, labels, split_dataset_by, spectrum, True, load_data_into_memory, ignore_list, filter_func, transform_func, transform, verbose)

    def __getitem__(self, idx):
        seq = self.get_ith_sequence(idx)
        images = seq.get_all_images_in_sequence()
        image_arrays = np.array([image.image() for image in images])
        labels = np.array([self._labels_from_label_strs(image, self.labels) for image in images])

        return TemporalSequencePairDataset(image_arrays, labels, self.transform)


class PreprocessedTyphoonDataset(DigitalTyphoonDataset):
    def __init__(self,
                 image_dir: str,
                 metadata_dir: str,
                 metadata_json: str,
                 labels,
                 split_dataset_by='image',
                 spectrum='Infrared',
                 load_data_into_memory=False,
                 ignore_list=None,
                 filter_func=None,
                 transform_func=None,
                 transform=None,
                 verbose=False,
                 feature_extractor=None) -> None:

        super().__init__(image_dir, metadata_dir, metadata_json, labels, split_dataset_by, spectrum, True, load_data_into_memory, ignore_list, filter_func, transform_func, transform, verbose)
        
        print("Dataset loading, now applying preprocessing on all sequences")
        assert feature_extractor is not None

        self.pp_sequences = []
        with torch.no_grad():
            for idx in tqdm(range(len(self.sequences))):
                seq = self.get_ith_sequence(idx)
                if seq.get_num_images() == 0:
                    self.pp_sequences.append([])
                    continue
                images = seq.get_all_images_in_sequence()
                labels = np.array([self._labels_from_label_strs(image, self.labels) for image in images])

                images = np.array([image.image() for image in images])
                images = self.transform(images).to("cuda")

                features = feature_extractor(images) # [BxD]
                self.pp_sequences.append([features, labels])

    def __getitem__(self, idx):
        return self.pp_sequences[idx]

LABEL_SIZE = dict(
    month=12,
    day=31,
    hour=24,
    grade=6,
)

NORMALIZATION = dict(
    pressure=[983.8, 22.5],
    wind=[36.7, 32.7],
    lat=[22.58, 10.6],
    lng=[136.2, 17.3],
)

class SequenceTyphoonDataset(DigitalTyphoonDataset):
    def __init__(self,
                 labels,
                 x,
                 y,
                 num_inputs,
                 num_preds,
                 interval=1,
                 output_all=False,
                 preprocessed_path=None,
                 image_dir: str="/fs9/gaspar/data/WP/image/",
                 metadata_dir: str="/fs9/gaspar/data/WP/metadata/",
                 metadata_json: str="/fs9/gaspar/data/WP/metadata.json",
                 spectrum='Infrared',
                 load_data_into_memory=False,
                 ignore_list=None,
                 filter_func=None,
                 transform_func=None,
                 transform=None,
                 verbose=False) -> None:
        """
        labels: labels to include in ["year", "month", "day", "hour", "grade", "lat", "lng", "pressure", "wind"]
        include_images: boolean to include or not images when generating sequences
        x: sequence data to use as inputs. should be array indices corresponding to the order in which labels are requested
        y: sequence data to use as targets. should be array indices corresponding to the order in which labels are requested
            images, if included are always included
        num_inputs: length of sequence used as input to the model
        num_preds: length of predicted datapoints
        preprocess: preprocess images to a smaller feature vector
        """
        super().__init__(image_dir, metadata_dir, metadata_json, labels, "sequence", spectrum, True, load_data_into_memory, ignore_list, filter_func, transform_func, transform, verbose)
        idx = 0
        self.x = []
        self.y = []

        for i, label in enumerate(labels):
            sz = LABEL_SIZE[label] if label in LABEL_SIZE else 1
            if i in x:
                self.x.extend(list(range(idx, idx+sz)))
            if i in y:
                self.y.extend(list(range(idx, idx+sz)))
            idx += sz

        self.num_inputs = num_inputs
        self.num_preds = num_preds
        self.interval = interval
        self.output_all = output_all

        self.slice_inputs = lambda start_idx: slice(start_idx, start_idx+(self.num_inputs*self.interval),self.interval)
        self.slice_outputs = lambda start_idx: slice(start_idx+(self.num_inputs*self.interval),start_idx+((self.num_inputs+self.num_preds)*self.interval), self.interval)

        if preprocessed_path is None:
            print("WARNING: no images used")

        self.preprocessed_path = preprocessed_path

        # Post process sequences filter out too short sequences
        for seq in self.sequences:
            if seq.get_num_images() < (self.num_inputs + self.num_preds)*self.interval+1:
                self.number_of_images -= seq.get_num_images()
                seq.images.clear()
                self.number_of_nonempty_sequences -= 1

        self.number_of_nonempty_sequences += 1


    def __getitem__(self, idx):
        seq = self.get_ith_sequence(idx)
        start_idx = np.random.randint(0, seq.get_num_images()-(self.num_inputs + self.num_preds)*self.interval)
        images = seq.get_all_images_in_sequence()

        labels = torch.stack([self._labels_from_label_strs(image, self.labels) for image in images])

        if self.output_all:
            if self.preprocessed_path is not None:
                npz = np.load(f"{self.preprocessed_path}/{seq.sequence_str}.npz")
                names_to_features = dict(zip(npz["arr_1"], npz["arr_0"]))
                features = [names_to_features[str(img.image_filepath).split("/")[-1].split(".")[0]] 
                            for img in images]
                features = torch.from_numpy(np.array(features))

                return torch.cat((labels, features), dim=1)
            else:
                return labels

        lab_inputs = labels[self.slice_inputs(start_idx), self.x]
        lab_preds = labels[self.slice_outputs(start_idx), self.y]

        if self.preprocessed_path is not None:
            # TODO handle preprocessed images
            npz = np.load(f"{self.preprocessed_path}/{seq.sequence_str}.npz")
            names_to_features = dict(zip(npz["arr_1"], npz["arr_0"]))
            features = [names_to_features[str(img.image_filepath).split("/")[-1].split(".")[0]] 
                        for img in images[self.slice_inputs(start_idx)]]
            features = torch.from_numpy(np.array(features))

            lab_inputs = torch.cat((lab_inputs, features), dim=1)


        return lab_inputs, lab_preds


    def _labels_from_label_strs(self, image, label_strs):
        """
        Given an image and the label/labels to retrieve from the image, returns a single label or
        a list of labels

        :param image: image to access labels for
        :param label_strs: either a List of label strings or a single label string
        :return: a List of label strings or a single label string
        """
        if (type(label_strs) is list) or (type(label_strs) is tuple):
            label_ray = torch.cat([self._prepare_labels(image.value_from_string(label), label) for label in label_strs])
            return label_ray
        else:
            label = self._prepare_labels(image.value_from_string(label_strs), label_strs)
            return label


    def _prepare_labels(self, value, label):
        #print(label)
        if label in LABEL_SIZE:
            one_hot = torch.zeros(LABEL_SIZE[label])
            #print(one_hot)
            if label == "hour":
                one_hot[value] = 1
            elif label == "grade":
                one_hot[value-2] = 1                
            else:
                one_hot[value-1] = 1
            return one_hot
        else:
            # Normalize
            if label in NORMALIZATION:
                #print(label, value)
                mean, std = NORMALIZATION[label]
                return (torch.Tensor([value]) - mean) / std

            return torch.Tensor([value])
