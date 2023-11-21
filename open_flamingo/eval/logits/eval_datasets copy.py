from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

from classification_utils import *

import os
import scipy.io
from PIL import Image

class ImageNetDataset(ImageFolder):
    """Class to represent the ImageNet dataset."""

    def __init__(self, root, **kwargs):
        super().__init__(root=root, **kwargs)
        self.class_id_to_name = dict(
            zip(range(len(IMAGENET_CLASSNAMES)), IMAGENET_CLASSNAMES)
        )

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        target_label = self.class_id_to_name[target]
        return {
            "id": idx,
            "image": sample,
            "class_id": target,  # numeric ID of the ImageNet class
            "class_name": target_label,  # human-readable name of ImageNet class
        }
    
class CUB200Dataset(Dataset):
    """Class to represent the CUB-200 dataset."""

    def __init__(self, root, transform=None, train=True):
        self.root = root
        self.transform = transform
        self.train = train
        self.image_paths = []
        self.labels = []
        self.class_id_to_name = dict(zip(range(len(CUB_CLASSNAMES)), CUB_CLASSNAMES))

        image_names = self._get_image_names()

        label_file = os.path.join(root, 'image_class_labels.txt')
        with open(label_file, 'r') as f:
            for line in f:
                image_id, class_id = line.strip().split()
                self.image_paths.append(os.path.join(root, 'images', image_names[int(image_id) - 1]))
                self.labels.append(int(class_id) - 1)  # Class IDs are 1-indexed

        split_file = os.path.join(root, 'train_test_split.txt')
        with open(split_file, 'r') as f:
            split_lines = f.readlines()

        if self.train:
            self.image_paths = [path for i, path in enumerate(self.image_paths) if int(split_lines[i].strip().split()[1]) == 1]
            self.labels = [label for i, label in enumerate(self.labels) if int(split_lines[i].strip().split()[1]) == 1]
        else:
            self.image_paths = [path for i, path in enumerate(self.image_paths) if int(split_lines[i].strip().split()[1]) == 0]
            self.labels = [label for i, label in enumerate(self.labels) if int(split_lines[i].strip().split()[1]) == 0]

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        class_name = CUB_CLASSNAMES[label]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return {
            "id": idx,
            "image": img,
            "class_id": label,
            "class_name": class_name,
        }
    
    def _get_image_names(self):
        image_names = []
        with open(os.path.join(self.root, 'images.txt'), 'r') as f:
            for line in f:
                _, image_name = line.strip().split()
                image_names.append(image_name)
        return image_names
    
    def __len__(self) -> int:
        return len(self.image_paths)

class StanfordCarDataset(ImageFolder):
    """Class to represent the Stanford Cars dataset."""

    def __init__(self, root, **kwargs):
        super().__init__(root=root, **kwargs)
        self.class_id_to_name = dict(
            zip(range(len(STANFORD_CAR_CLASSNAMES)), STANFORD_CAR_CLASSNAMES)
            )

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        target_label = self.class_id_to_name[target]
        return {
            "id": idx,
            "image": sample,
            "class_id": target,
            "class_name": target_label,
        }
    
class StanfordDogDataset(Dataset):
    """Class to represent the Stanford Dogs dataset."""

    def __init__(self, root, transform=None, train=True):
        self.root = root
        self.transform = transform
        self.train = train
        self.image_paths = []
        self.labels_ = []
        self.labels = []
        self.split_file = "train_list.mat" if self.train else "test_list.mat"
        self.class_id_to_name = dict(zip(range(len(STANFORD_DOG_CLASSNAMES)), STANFORD_DOG_CLASSNAMES))

        self.class_name2id = dict(
            zip(STANFORD_DOG_CLASSNAMES, range(len(STANFORD_DOG_CLASSNAMES)))
        )
        file_list = scipy.io.loadmat(os.path.join(self.root, self.split_file))['file_list']
        for item in file_list:
            file_path = item[0][0]
            self.image_paths.append(os.path.join(root, "images", file_path))
            class_name = file_path.split("/")[0][10:]
            self.labels_.append(class_name)
            self.labels.append(self.class_name2id[class_name])
            


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path, target_label = self.image_paths[idx], self.labels_[idx]
        sample = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        class_id = self.class_name2id[target_label]
        return {
            "id": idx,
            "image": sample,
            "class_id": class_id,
            "class_name": target_label,
        }

class Food101Dataset(Dataset):
    """Class to represent the Food101 dataset."""

    def __init__(self, root, transform=None, train=True):
        self.root = root
        self.transform = transform
        self.train = train
        self.image_paths = []
        # self.labels_ = []
        self.labels = []
        self.split_file = "train.txt" if self.train else "test.txt"
        self.class_id_to_name = dict(zip(range(len(FOOD101_NAMES)), FOOD101_NAMES))

        self.class_name2id = dict(
            zip(FOOD101_NAMES, range(len(FOOD101_NAMES)))
        )
        split_file = os.path.join(root, "meta", self.split_file)
        with open(split_file, 'r') as f:
            for line in f:
                self.labels.append(line.split('/')[0])
                self.image_paths.append(os.path.join(root, 'images', line[:-1])+".jpg")
            


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path, target_label = self.image_paths[idx], self.labels[idx]
        sample = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        class_id = self.class_name2id[target_label]
        return {
            "id": idx,
            "image": sample,
            "class_id": class_id,
            "class_name": target_label
        }

class Flowers102Dataset(Dataset):
    """Class to represent the Oxford Flowers102 dataset."""

    def __init__(self, root, transform=None, train=True):
        self.root = root
        self.transform = transform
        self.train = train
        self.all_image_data = []
        self.image_paths = []
        self.labels_ = []
        self.labels = []
        self.split = None
        self.class_id_to_name = dict(zip(range(len(FLOWERS102_NMAES)), FLOWERS102_NMAES))

        # self.class_name2id = dict(
        #     zip(STANFORD_DOG_CLASSNAMES, range(len(STANFORD_DOG_CLASSNAMES)))
        # )
        jpg_files = [file for file in os.listdir(os.path.join(root,"jpg")) if file.endswith(".jpg")]
        jpg_files.sort()
        for file in jpg_files:
            file_path = os.path.join(os.path.join(root,"jpg"), file)
            self.all_image_data.append(file_path)
        file_list = scipy.io.loadmat(os.path.join(self.root, "setid.mat"))
        all_image_labels = list(scipy.io.loadmat(os.path.join(self.root, "imagelabels.mat"))['labels'][0])
        if not train:
            self.split = list(file_list['tstid'][0])
        else:
            self.split = list(file_list['trnid'][0]) \
                            + list(file_list['valid'][0])
        for _, idx in enumerate(self.split):
            self.image_paths.append(self.all_image_data[idx - 1])
            self.labels.append(all_image_labels[idx - 1])
            
            


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # return idx
        image_path, target_label = self.image_paths[idx], self.labels[idx]
        sample = Image.open(image_path).convert('RGB')
        class_name = FLOWERS102_NMAES[target_label-1]

        return {
            "id": idx,
            "image": sample,
            "class_id": target_label,
            "class_name": class_name,
        }

class OxfordPetDataset(Dataset):
    """Class to represent the Food101 dataset."""

    def __init__(self, root, transform=None, train=True):
        self.root = root
        self.transform = transform
        self.train = train
        self.image_paths = []
        self.labels = []
        # self.class_names= {}
        self.split_file = "trainval.txt" if self.train else "test.txt"
        self.class_id_to_name = dict(zip(range(len(PETS_NAMES)), PETS_NAMES))

        # self.class_name2id = dict(
        #     zip(FOOD101_NAMES, range(len(FOOD101_NAMES)))
        # )
        split_file = os.path.join(root, "annotations", self.split_file)
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip("\n").split(' ')
                self.image_paths.append(os.path.join(self.root, 'images', line[0]+".jpg"))
                self.labels.append(line[1])
                # class_name = line[0].rsplit('_', 1)[0]
                # self.class_names[class_name] = line[1]
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # return idx
        image_path, target_label = self.image_paths[idx], self.labels[idx]
        sample = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        class_name= self.class_id_to_name[int(target_label)-1]
        return {
            "id": idx,
            "image": sample,
            "class_id": target_label,
            "class_name": class_name
        }


class DTDDataset(Dataset):
    """Class to represent the Oxford Flowers102 dataset."""

    def __init__(self, root, transform=None, train=True):
        self.root = root
        self.transform = transform
        self.train = train
        self.all_image_data = []
        self.image_paths = []
        self.labels_ = []
        self.labels = []
        self.split = None
        self.class_id_to_name = dict(zip(range(len(FLOWERS102_NMAES)), FLOWERS102_NMAES))

        # self.class_name2id = dict(
        #     zip(STANFORD_DOG_CLASSNAMES, range(len(STANFORD_DOG_CLASSNAMES)))
        # )
        # jpg_files = [file for file in os.listdir(os.path.join(root,"jpg")) if file.endswith(".jpg")]
        # jpg_files.sort()
        # for file in jpg_files:
        #     file_path = os.path.join(os.path.join(root,"jpg"), file)
        #     self.all_image_data.append(file_path)
        imdb = scipy.io.loadmat(os.path.join(self.root, "imdb","imdb.mat"))
        imdb_img = imdb['images']
        image_names,splits,class_ids = imdb_img['name'].item()[0], imdb_img['set'].item()[0], imdb_img['class'].item()[0]
        if self.train:
            for name, split, idx in zip(image_names, splits, class_ids):
                if split == 1:
                    self.labels.append(idx)
                    self.image_paths.append(os.path.join(root, "images", name.item()))
        else:
            for name, split, idx in zip(image_names, splits, class_ids):
                if split != 1:
                    self.labels.append(idx)
                    self.image_paths.append(os.path.join(root, "images", name.item()))
                    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # return idx
        image_path, target_label = self.image_paths[idx], self.labels[idx]
        sample = Image.open(image_path).convert('RGB')
        class_name = DTD_NAMES[target_label-1]

        return {
            "id": idx,
            "image": sample,
            "class_id": target_label,
            "class_name": class_name,
        }