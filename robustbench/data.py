import os
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from robustbench.model_zoo.enums import BenchmarkDataset
from robustbench.zenodo_download import DownloadError, zenodo_download
from robustbench.loaders import CustomImageFolder


PREPROCESSINGS = {
    'Res256Crop224': transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor()]),
    'Crop288': transforms.Compose([transforms.CenterCrop(288),
                                   transforms.ToTensor()]),
    'none': transforms.Compose([transforms.ToTensor()]),
}


def _load_dataset(
        dataset: Dataset,
        n_examples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = 100
    test_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=0)

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)

    if n_examples is not None:
        x_test_tensor = x_test_tensor[:n_examples]
        y_test_tensor = y_test_tensor[:n_examples]

    return x_test_tensor, y_test_tensor


def load_cifar10(
        n_examples: Optional[int] = None,
        data_dir: str = './data',
        prepr: Optional[str] = 'none') -> Tuple[torch.Tensor, torch.Tensor]:
    transforms_test = PREPROCESSINGS[prepr]
    dataset = datasets.CIFAR10(root=data_dir,
                               train=False,
                               transform=transforms_test,
                               download=True)
    return _load_dataset(dataset, n_examples)


def load_cifar100(
        n_examples: Optional[int] = None,
        data_dir: str = './data',
        prepr: Optional[str] = 'none') -> Tuple[torch.Tensor, torch.Tensor]:
    transforms_test = PREPROCESSINGS[prepr]
    dataset = datasets.CIFAR100(root=data_dir,
                                train=False,
                                transform=transforms_test,
                                download=True)
    return _load_dataset(dataset, n_examples)


def load_imagenet(
        n_examples: Optional[int] = 5000,
        data_dir: str = './data',
        prepr: str = 'Res256Crop224') -> Tuple[torch.Tensor, torch.Tensor]:
    transforms_test = PREPROCESSINGS[prepr]
    imagenet = CustomImageFolder(data_dir + '/val', transforms_test)
    
    test_loader = data.DataLoader(imagenet, batch_size=n_examples,
                                  shuffle=False, num_workers=4)

    x_test, y_test, paths = next(iter(test_loader))
    
    return x_test, y_test


CleanDatasetLoader = Callable[[Optional[int], str], Tuple[torch.Tensor,
                                                          torch.Tensor]]
_clean_dataset_loaders: Dict[BenchmarkDataset, CleanDatasetLoader] = {
    BenchmarkDataset.cifar_10: load_cifar10,
    BenchmarkDataset.cifar_100: load_cifar100,
    BenchmarkDataset.imagenet: load_imagenet,
}


def load_clean_dataset(dataset: BenchmarkDataset, n_examples: Optional[int],
                       data_dir: str, prepr: Optional[str] = 'none') -> Tuple[torch.Tensor, torch.Tensor]:
    return _clean_dataset_loaders[dataset](n_examples, data_dir, prepr)


CORRUPTIONS = ("shot_noise", "motion_blur", "snow", "pixelate",
               "gaussian_noise", "defocus_blur", "brightness", "fog",
               "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
               "jpeg_compression", "elastic_transform")

ZENODO_CORRUPTIONS_LINKS: Dict[BenchmarkDataset, Tuple[str, Set[str]]] = {
    BenchmarkDataset.cifar_10: ("2535967", {"CIFAR-10-C.tar"}),
    BenchmarkDataset.cifar_100: ("3555552", {"CIFAR-100-C.tar"})
}

CORRUPTIONS_DIR_NAMES: Dict[BenchmarkDataset, str] = {
    BenchmarkDataset.cifar_10: "CIFAR-10-C",
    BenchmarkDataset.cifar_100: "CIFAR-100-C",
    BenchmarkDataset.imagenet: "ImageNet-C"
}


def load_cifar10c(
    n_examples: int,
    severity: int = 5,
    data_dir: str = './data',
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS,
    prepr: Optional[str] = 'none'
) -> Tuple[torch.Tensor, torch.Tensor]:
    return load_corruptions_cifar(BenchmarkDataset.cifar_10, n_examples,
                                  severity, data_dir, corruptions, shuffle)


def load_cifar100c(
    n_examples: int,
    severity: int = 5,
    data_dir: str = './data',
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS,
    prepr: Optional[str] = 'none'
) -> Tuple[torch.Tensor, torch.Tensor]:
    return load_corruptions_cifar(BenchmarkDataset.cifar_100, n_examples,
                                  severity, data_dir, corruptions, shuffle)


def load_imagenetc(
    n_examples: Optional[int] = 5000,
    severity: int = 5,
    data_dir: str = './data',
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS,
    prepr: str = 'Res256Crop224'
) -> Tuple[torch.Tensor, torch.Tensor]:
    transforms_test = PREPROCESSINGS[prepr]

    assert len(corruptions) == 1, "so far only one corruption is supported (that's how this function is called in eval.py"

    data_folder_path = Path(data_dir) / CORRUPTIONS_DIR_NAMES[BenchmarkDataset.imagenet] / corruptions[0] / str(severity)
    imagenet = CustomImageFolder(data_folder_path, transforms_test)

    test_loader = data.DataLoader(imagenet, batch_size=n_examples,
                                  shuffle=shuffle, num_workers=2)

    x_test, y_test, paths = next(iter(test_loader))

    return x_test, y_test


CorruptDatasetLoader = Callable[[int, int, str, bool, Sequence[str]],
                                Tuple[torch.Tensor, torch.Tensor]]
CORRUPTION_DATASET_LOADERS: Dict[BenchmarkDataset, CorruptDatasetLoader] = {
    BenchmarkDataset.cifar_10: load_cifar10c,
    BenchmarkDataset.cifar_100: load_cifar100c,
    BenchmarkDataset.imagenet: load_imagenetc,
}


def load_corruptions_cifar(
        dataset: BenchmarkDataset,
        n_examples: int,
        severity: int,
        data_dir: str,
        corruptions: Sequence[str] = CORRUPTIONS,
        shuffle: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    assert 1 <= severity <= 5
    n_total_cifar = 10000

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data_dir = Path(data_dir)
    data_root_dir = data_dir / CORRUPTIONS_DIR_NAMES[dataset]  # Kyle:这里指定了stamp_lib/dataset/ + CIFAR-10-C

    if not data_root_dir.exists():
        zenodo_download(*ZENODO_CORRUPTIONS_LINKS[dataset], save_dir=data_dir)

    # Download labels
    labels_path = data_root_dir / 'labels.npy' # Kyle:查找标签.npy文件
    if not os.path.isfile(labels_path):
        raise DownloadError("Labels are missing, try to re-download them.")
    labels = np.load(labels_path) # Kyle:加载

    x_test_list, y_test_list = [], []
    n_pert = len(corruptions)
    count_for_corr = 0
    for corruption in corruptions:
        corruption_file_path = data_root_dir / (corruption + '.npy')
        if not corruption_file_path.is_file():
            raise DownloadError(
                f"{corruption} file is missing, try to re-download it.")

        images_all = np.load(corruption_file_path)
        images = images_all[(severity - 1) * n_total_cifar:severity *
                            n_total_cifar] # Kyle:每个corruptin类型中是五个severity连着的 这里只取五级
        n_img = int(np.ceil(n_examples / n_pert))
        x_test_list.append(images[:n_img]) # Kyle:x_test_list是[[corrupt1],[corrupt2],....]
        # Duplicate the same labels potentially multiple times
        y_test_list.append(labels[:n_img]) # Kyle:虽然只是每轮循环取一种corruption的第五级图片接在list后 但label也跟着复制了
        count_for_corr += 1
        print(f"count_for_corr:{count_for_corr}")
    print(f"len(x_test_list) : {len(x_test_list)}")
    print(f"x_test_list[0].shape : {x_test_list[0].shape}")
    x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)  # Kyle: merge the [[],[],[]] into [    ]
    print(f"len(x_test) : {len(x_test)}")
    # Kyle:最终形态x_test = [crpt1 crpt2 ...] y_test = [y' y' ...]
    
    if shuffle:
        rand_idx = np.random.permutation(np.arange(len(x_test)))
        x_test, y_test = x_test[rand_idx], y_test[rand_idx]

    # Make it in the PyTorch format
    x_test = np.transpose(x_test, (0, 3, 1, 2))  # Kyle: Change to (N, C, W, H)
    # Make it compatible with our models
    x_test = x_test.astype(np.float32) / 255  # Kyle:Normalize pix_val [0,255] to [0,1]
    # Make sure that we get exactly n_examples but not a few samples more
    x_test = torch.tensor(x_test)[:n_examples]
    y_test = torch.tensor(y_test)[:n_examples]
    print(f"x_test = torch.tensor(x_test)[:n_examples] SHAPE:{x_test.shape}")
    print(f"y_test = torch.tensor(y_test)[:n_examples] SHAPE:{y_test.shape}")



    return x_test, y_test
