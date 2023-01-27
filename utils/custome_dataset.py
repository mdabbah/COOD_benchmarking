import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class CustomedDataset(Dataset):

    def __init__(self, img_files, labels, num_shots=None,
                 given_transforms=None):
        self.image_files = img_files
        self.labels = labels
        self.classes = np.unique(labels)
        self.num_classes = len(self.classes)
        self.num_shots = num_shots
        if num_shots is not None:
            self.pick_shots(num_shots)

        if given_transforms is not None:
            self.transform = given_transforms
        else:
            raise ValueError('give me transforms')

    def pick_shots(self, num_shots):
        label_set = np.unique(self.labels)
        all_samples = []
        new_labels = []
        for c_ in label_set:
            class_samples = self.image_files[self.labels == c_]
            np.random.shuffle(class_samples)
            class_samples = class_samples[:num_shots]
            all_samples.append(class_samples)
            new_labels.extend([c_] * class_samples.shape[0])

        self.image_files = np.concatenate(all_samples, axis=0)
        self.labels = np.array(new_labels)
        assert len(self.labels) == len(self.image_files)
        self.shuffle()

    def shuffle(self):
        perm = np.random.permutation(len(self.labels))
        self.image_files = self.image_files[perm]
        self.labels = self.labels[perm]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx].strip()
        img = self.transform(image_file)
        label = torch.tensor(self.labels[idx])
        return img, label


class DummyDataset(Dataset):

    def __init__(self, num_samples_per_class, num_classes, given_transforms):
        self.labels = np.repeat(np.arange(num_classes), num_samples_per_class)
        self.classes = np.arange(num_classes)
        self.num_classes = num_classes

        if given_transforms is not None:
            self.transform = given_transforms
        else:
            raise ValueError('give me transforms')

        self.shuffle()

    def shuffle(self):
        perm = np.random.permutation(len(self.labels))
        self.labels = self.labels[perm]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.rand(3, 224, 224)
        label = torch.tensor(self.labels[idx])
        return img, label


def get_base_transforms(input_size, use_random_erase=False):
    out_size = int(192 / 224 * input_size), int(192 / 224 * input_size)
    if use_random_erase:
        base_transforms = transforms.Compose([
            transforms.Lambda(lambda path: Image.open(path)),
            transforms.Lambda(lambda image: image.convert('RGB') if image.mode == 'L' else image),
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode == 'RGBA' else img),
            transforms.Resize(input_size),
            transforms.RandomCrop(out_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda image_tnsr: image_tnsr.repeat([3, 1, 1]) if image_tnsr.shape[0] == 1 else image_tnsr),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing()
        ])
    else:
        base_transforms = transforms.Compose([
            transforms.Lambda(lambda path: Image.open(path)),
            transforms.Lambda(lambda image: image.convert('RGB') if image.mode == 'L' else image),
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode == 'RGBA' else img),
            transforms.Resize(input_size),
            transforms.RandomCrop(out_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda image_tnsr: image_tnsr.repeat([3, 1, 1]) if image_tnsr.shape[0] == 1 else image_tnsr),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return base_transforms


def get_testing_transform(input_size=(224, 224), normalize=True):
    if normalize:
        data_transforms = transforms.Compose([
            transforms.Lambda(lambda path: Image.open(path)),
            transforms.Lambda(lambda image: image.convert('RGB') if image.mode == 'L' else image),
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode == 'RGBA' else img),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda image_tnsr: image_tnsr.repeat([3, 1, 1]) if image_tnsr.shape[0] == 1 else image_tnsr),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.Lambda(lambda path: Image.open(path)),
            transforms.Lambda(lambda image: image.convert('RGB') if image.mode == 'L' else image),
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode == 'RGBA' else img),
            transforms.Lambda(lambda image: image.repeat([3, 1, 1]) if image.shape[0] == 1 else image),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ])

    return data_transforms


def open_img(path):
    return Image.open(path)

def convert_to_RGB(image):
    return image.convert('RGB')

def convert_RGBL(image):
    return image.convert('RGB') if image.mode == 'L' else image


def covert_RGBA(img):
    return img.convert('RGB') if img.mode == 'RGBA' else img


def convert_greyscale(image):
    return image.repeat([3, 1, 1]) if image.shape[0] == 1 else image


def get_open_img_transforms():
    open_img_transforms = transforms.Compose(
        [transforms.Lambda(open_img),
         transforms.Lambda(convert_to_RGB)])

    return open_img_transforms
