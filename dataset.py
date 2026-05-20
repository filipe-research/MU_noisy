"""
function for loading datasets
contains:
    CIFAR-10
    CIFAR-100
"""

import copy
import glob
import json
import os
import random
from shutil import move

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import stats
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, Food101, SVHN, ImageFolder
from tqdm import tqdm
import pandas as pd


def cifar10_dataloaders_no_val(
    batch_size=128, data_dir="datasets/cifar10", num_workers=2
):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)
    val_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def svhn_dataloaders(
    batch_size=128,
    data_dir="datasets/svhn",
    num_workers=2,
    class_to_replace: int = None,
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
):
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: SVHN\t 45000 images for training \t 5000 images for validation\t"
    )

    train_set = SVHN(data_dir, split="train", transform=train_transform, download=True)

    test_set = SVHN(data_dir, split="test", transform=test_transform, download=True)

    train_set.labels = np.array(train_set.labels)
    test_set.labels = np.array(test_set.labels)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.labels) + 1):
        class_idx = np.where(train_set.labels == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.labels = train_set_copy.labels[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.labels = train_set_copy.labels[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None or num_indexes_to_replace == 4454:
            test_set.data = test_set.data[test_set.labels != class_to_replace]
            test_set.labels = test_set.labels[test_set.labels != class_to_replace]

    if indexes_to_replace is not None:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


def cifar100_dataloaders(
    batch_size=128,
    data_dir="datasets/cifar100",
    num_workers=2,
    class_to_replace: int = None,
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
    noise_rate=0.0,
    noise_mode="sym",
    noise_file=None,
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")
    train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=True)

    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)
    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    # rng = np.random.RandomState(seed)
    # valid_set = copy.deepcopy(train_set)
    # valid_idx = []
    # for i in range(max(train_set.targets) + 1):
    #     class_idx = np.where(train_set.targets == i)[0]
    #     valid_idx.append(
    #         rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
    #     )
    # valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    # valid_set.data = train_set_copy.data[valid_idx]
    # valid_set.targets = train_set_copy.targets[valid_idx]

    # train_idx = list(set(range(len(train_set))) - set(valid_idx))
    train_idx = list(range(len(train_set)))

    # noisify trainset

    noise_file = f"cifar100_{noise_rate}_sym.json"
    if os.path.exists(noise_file):
        noise = json.load(open(noise_file, "r"))
        noise_labels = noise["noise_labels"]

        train_set_copy.targets = np.array(noise_labels)
        train_set.targets = train_set_copy.targets

    else:
        noise_labels = []  # all labels (some noisy, some clean)
        idx = train_idx
        num_total_noise = int(noise_rate * len(train_idx))  # total amount of noise
        # import pdb; pdb.set_trace()

        print(
            "Statistics of synthetic noisy CIFAR dataset: ",
            "num of clean samples: ",
            len(train_idx) - num_total_noise,
            " num of closed-set noise: ",
            num_total_noise,
        )

        closed_noise = idx[0:num_total_noise]  # closed set noise indices

        for i in range(50000):  # pra incluir o conjunto de validação.
            # Mas o conjunto de validacao nao vai ser alterado pq o idx é baseado no train_idx
            if i in closed_noise:
                noiselabel = random.randint(0, 99)
                noise_labels.append(noiselabel)
                train_set_copy.targets[i] = noiselabel

            else:
                noise_labels.append(train_set_copy.targets[i])

        noise_labels = [int(x) for x in noise_labels]
        clean_idx = list(set(range(len(train_idx))) - set(closed_noise))
        noise = {
            "noise_labels": noise_labels,
            "closed_noise": closed_noise,
            "clean_idx": clean_idx,
        }

        print("save noise to %s ..." % noise_file)
        json.dump(noise, open(noise_file, "w"))

    # end noisify trainset

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None:
            test_set.data = test_set.data[test_set.targets != class_to_replace]
            test_set.targets = test_set.targets[test_set.targets != class_to_replace]
    if indexes_to_replace is not None or indexes_to_replace == 450:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    # val_loader = DataLoader(
    #     valid_set,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     worker_init_fn=_init_fn if seed is not None else None,
    #     **loader_args,
    # )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, test_loader, test_loader


def cifar100_dataloaders_no_val(
    batch_size=128, data_dir="datasets/cifar100", num_workers=2
):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=True)
    val_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


class TinyImageNetDataset(Dataset):
    def __init__(self, image_folder_set, norm_trans=None, start=0, end=-1):
        self.imgs = []
        self.targets = []
        self.transform = image_folder_set.transform
        for sample in tqdm(image_folder_set.imgs[start:end]):
            self.targets.append(sample[1])
            img = transforms.ToTensor()(Image.open(sample[0]).convert("RGB"))
            if norm_trans is not None:
                img = norm_trans(img)
            self.imgs.append(img)
        self.imgs = torch.stack(self.imgs)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.imgs[idx]), self.targets[idx]
        else:
            return self.imgs[idx], self.targets[idx]


class TinyImageNet:
    """
    TinyImageNet dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = (
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if normalize
            else None
        )

        self.tr_train = [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        self.tr_test = []

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

        self.train_path = os.path.join(args.data_dir, "train/")
        self.val_path = os.path.join(args.data_dir, "val/")
        self.test_path = os.path.join(args.data_dir, "test/")

        if os.path.exists(os.path.join(self.val_path, "images")):
            if os.path.exists(self.test_path):
                os.rename(self.test_path, os.path.join(args.data_dir, "test_original"))
                os.mkdir(self.test_path)
            val_dict = {}
            val_anno_path = os.path.join(self.val_path, "val_annotations.txt")
            with open(val_anno_path, "r") as f:
                for line in f.readlines():
                    split_line = line.split("\t")
                    val_dict[split_line[0]] = split_line[1]

            paths = glob.glob(os.path.join(args.data_dir, "val/images/*"))
            for path in paths:
                file = path.split("/")[-1]
                folder = val_dict[file]
                if not os.path.exists(self.val_path + str(folder)):
                    os.mkdir(self.val_path + str(folder))
                    os.mkdir(self.val_path + str(folder) + "/images")
                if not os.path.exists(self.test_path + str(folder)):
                    os.mkdir(self.test_path + str(folder))
                    os.mkdir(self.test_path + str(folder) + "/images")

            for path in paths:
                file = path.split("/")[-1]
                folder = val_dict[file]
                if len(glob.glob(self.val_path + str(folder) + "/images/*")) < 25:
                    dest = self.val_path + str(folder) + "/images/" + str(file)
                else:
                    dest = self.test_path + str(folder) + "/images/" + str(file)
                move(path, dest)

            os.rmdir(os.path.join(self.val_path, "images"))

    def data_loaders(
        self,
        batch_size=128,
        data_dir="datasets/tiny",
        num_workers=2,
        class_to_replace: int = None,
        num_indexes_to_replace=None,
        indexes_to_replace=None,
        seed: int = 1,
        only_mark: bool = False,
        shuffle=True,
        no_aug=False,
    ):
        train_set = ImageFolder(self.train_path, transform=self.tr_train)
        train_set = TinyImageNetDataset(train_set, self.norm_layer)
        test_set = ImageFolder(self.test_path, transform=self.tr_test)
        test_set = TinyImageNetDataset(test_set, self.norm_layer)
        train_set.targets = np.array(train_set.targets)
        train_set.targets = np.array(train_set.targets)
        rng = np.random.RandomState(seed)
        valid_set = copy.deepcopy(train_set)
        valid_idx = []
        for i in range(max(train_set.targets) + 1):
            class_idx = np.where(train_set.targets == i)[0]
            valid_idx.append(
                rng.choice(class_idx, int(0.0 * len(class_idx)), replace=False)
            )
        valid_idx = np.hstack(valid_idx)
        train_set_copy = copy.deepcopy(train_set)

        valid_set.imgs = train_set_copy.imgs[valid_idx]
        valid_set.targets = train_set_copy.targets[valid_idx]

        train_idx = list(set(range(len(train_set))) - set(valid_idx))

        train_set.imgs = train_set_copy.imgs[train_idx]
        train_set.targets = train_set_copy.targets[train_idx]

        if class_to_replace is not None and indexes_to_replace is not None:
            raise ValueError(
                "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
            )
        if class_to_replace is not None:
            replace_class(
                train_set,
                class_to_replace,
                num_indexes_to_replace=num_indexes_to_replace,
                seed=seed - 1,
                only_mark=only_mark,
            )
            if num_indexes_to_replace is None or num_indexes_to_replace == 500:
                test_set.targets = np.array(test_set.targets)
                test_set.imgs = test_set.imgs[test_set.targets != class_to_replace]
                test_set.targets = test_set.targets[
                    test_set.targets != class_to_replace
                ]
                test_set.targets = test_set.targets.tolist()
        if indexes_to_replace is not None:
            replace_indexes(
                dataset=train_set,
                indexes=indexes_to_replace,
                seed=seed - 1,
                only_mark=only_mark,
            )

        loader_args = {"num_workers": 0, "pin_memory": False}

        def _init_fn(worker_id):
            np.random.seed(int(seed))

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        val_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, val_loader, test_loader


def cifar10_dataloaders(
    batch_size=128,
    data_dir="datasets/cifar10",
    num_workers=2,
    random_to_replace: int = None,
    class_to_replace: int = None,
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
    noise_rate=0.0,
    noise_mode="sym",
    noise_file=None,
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)

    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    # rng = np.random.RandomState(seed)
    # valid_set = copy.deepcopy(train_set)
    # valid_idx = []
    # for i in range(max(train_set.targets) + 1):
    #     class_idx = np.where(train_set.targets == i)[0]
    #     valid_idx.append(
    #         rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
    #     )
    # valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)
    # train_clean_set = copy.deepcopy(train_set)

    # valid_set.data = train_set_copy.data[valid_idx]
    # valid_set.targets = train_set_copy.targets[valid_idx]

    # train_idx = list(set(range(len(train_set))) - set(valid_idx))
    train_idx = list(range(len(train_set)))

    # import pdb; pdb.set_trace()
    # noise_labels = None
    # noisify trainset

    noise_file = f"cifar10_{noise_rate}_sym.json"
    if os.path.exists(noise_file):
        # import pdb; pdb.set_trace()
        noise = json.load(open(noise_file, "r"))
        noise_labels = noise["noise_labels"]

        # self.closed_noise = noise['closed_noise']
        # train_set_copy.targets[train_idx] = np.array(noise_labels)
        train_set_copy.targets = np.array(noise_labels)
        train_set.targets = train_set_copy.targets

    else:
        # inject noise
        noise_labels = []  # all labels (some noisy, some clean)
        # idx = list(range(50000))  # indices of cifar dataset
        idx = train_idx
        # random.shuffle(idx)
        # num_total_noise = int(self.r * 50000)  # total amount of noise
        num_total_noise = int(noise_rate * len(train_idx))  # total amount of noise

        print(
            "Statistics of synthetic noisy CIFAR dataset: ",
            "num of clean samples: ",
            len(train_idx) - num_total_noise,
            " num of closed-set noise: ",
            num_total_noise,
        )
        #   ' num of closed-set noise: ', num_total_noise - num_open_noise, ' num of open-set noise: ', num_open_noise)

        # target_noise_idx = list(range(50000))
        # target_noise_idx = train_idx
        # random.shuffle(target_noise_idx)
        # self.open_noise = list(
        #     zip(idx[:num_open_noise], target_noise_idx[:num_open_noise]))  # clean sample -> openset sample mapping
        # self.closed_noise = idx[num_open_noise:num_total_noise]  # closed set noise indices
        # self.closed_noise = idx[0:num_total_noise]  # closed set noise indices
        closed_noise = idx[0:num_total_noise]  # closed set noise indices
        # populate noise_labels

        transition = {
            0: 0,
            2: 0,
            4: 7,
            7: 7,
            1: 1,
            9: 1,
            3: 5,
            5: 3,
            6: 6,
            8: 8,
        }  # class transition for asymmetric noise

        for i in range(50000):  # pra incluir o conjunto de validação.
            # Mas o conjunto de validacao nao vai ser alterado pq o idx é baseado no train_idx
            # for i in idx:
            if i in closed_noise:
                if noise_mode == "sym":
                    noiselabel = random.randint(0, 9)
                elif noise_mode == "asym":
                    noiselabel = transition[train_set_copy.targets[i]]

                noise_labels.append(noiselabel)
                train_set_copy.targets[i] = noiselabel

            else:
                #     #noise_labels.append(cifar_label[i])
                # noise_labels.append(cifar_label[i])
                noise_labels.append(train_set_copy.targets[i])

        # train_set.targets = train_set_copy.targets[train_idx]
        train_set.targets = train_set_copy.targets
        # write noise to a file, to re-use
        # noise = {'noise_labels': noise_labels, 'open_noise': self.open_noise, 'closed_noise': self.closed_noise}
        # Converte cada elemento da lista em um número inteiro
        noise_labels = [int(x) for x in noise_labels]
        clean_idx = list(set(range(len(train_idx))) - set(closed_noise))
        noise = {
            "noise_labels": noise_labels,
            "closed_noise": closed_noise,
            "clean_idx": clean_idx,
        }

        print("save noise to %s ..." % noise_file)
        # import pdb; pdb.set_trace()
        # noise['noise_labels']
        # train_clean_set.targets[0:10]
        # train_set.targets[0:10]
        # train_set_copy.targets[0:10]
        # print('ok')
        json.dump(noise, open(noise_file, "w"))
        # self.cifar_label = noise_labels
        # self.open_id = np.array(self.open_noise)[:, 0] if len(self.open_noise) !=0 else None

    # end noisify trainset

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None or num_indexes_to_replace == 4500:
            test_set.data = test_set.data[test_set.targets != class_to_replace]
            test_set.targets = test_set.targets[test_set.targets != class_to_replace]
    if indexes_to_replace is not None:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    # val_loader = DataLoader(
    #     valid_set,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     worker_init_fn=_init_fn if seed is not None else None,
    #     **loader_args,
    # )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, test_loader, test_loader


def cifar10_idn_dataloaders(
    batch_size=128,
    data_dir="datasets/cifar10",
    seed: int = 1,
    no_aug=False,
    noise_rate=0.0,
    noise_file=None,
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)
    train_set.targets = np.array(train_set.targets)

    noise_file = f"cifar10_idn_{noise_rate}_sym.json"
    if os.path.exists(noise_file):
        noise = json.load(open(noise_file, "r"))
        noise_labels = noise["noise_labels"]

    else:
        # Gerar features
        data_tensor = torch.stack([img for img, _ in train_set])
        data_tensor = data_tensor.view(-1, 32 * 32 * 3)

        noise_labels = get_instance_noisy_label(
            n=noise_rate,
            dataset=list(zip(data_tensor, train_set.targets)),
            labels=np.array(train_set.targets),
            num_classes=10,
            feature_size=32 * 32 * 3,
            norm_std=0.1,
            seed=seed,
        )

        original_labels = train_set.targets
        closed_noise = np.where(noise_labels != original_labels)[0].tolist()
        clean_idx = np.where(noise_labels == original_labels)[0].tolist()

        noise = {
            "noise_labels": noise_labels.tolist(),
            "closed_noise": closed_noise,
            "clean_idx": clean_idx,
        }

        json.dump(noise, open(noise_file, "w"))
        print(f"Noise file saved to {noise_file}")

    # end noisify trainset

    # Aplicar rótulos ruidosos
    train_set.targets = np.array(noise_labels)

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, test_loader, test_loader


def cifar100_idn_dataloaders(
    batch_size=128,
    data_dir="datasets/cifar100",
    seed: int = 1,
    no_aug=False,
    noise_rate=0.0,
    noise_file=None,
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=True)
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)
    train_set.targets = np.array(train_set.targets)

    noise_file = f"cifar100_idn_{noise_rate}_sym.json"
    if os.path.exists(noise_file):
        noise = json.load(open(noise_file, "r"))
        noise_labels = noise["noise_labels"]

    else:
        # Gerar features
        data_tensor = torch.stack([img for img, _ in train_set])
        data_tensor = data_tensor.view(-1, 32 * 32 * 3)

        noise_labels = get_instance_noisy_label(
            n=noise_rate,
            dataset=list(zip(data_tensor, train_set.targets)),
            labels=np.array(train_set.targets),
            num_classes=100,
            feature_size=32 * 32 * 3,
            norm_std=0.1,
            seed=seed,
        )

        original_labels = train_set.targets
        closed_noise = np.where(noise_labels != original_labels)[0].tolist()
        clean_idx = np.where(noise_labels == original_labels)[0].tolist()

        noise = {
            "noise_labels": noise_labels.tolist(),
            "closed_noise": closed_noise,
            "clean_idx": clean_idx,
        }

        json.dump(noise, open(noise_file, "w"))
        print(f"Noise file saved to {noise_file}")

    # end noisify trainset

    # Aplicar rótulos ruidosos
    train_set.targets = np.array(noise_labels)

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, test_loader, test_loader


def cifar10_openset_dataloaders(
    batch_size=128,
    data_dir="datasets/cifar10",
    seed=1,
    no_aug=False,
    noise_rate=0.0,
    noise_file=None,
    open_ratio=0.0,
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)

    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)
    train_set_copy = copy.deepcopy(train_set)
    train_idx = list(range(len(train_set)))

    if open_ratio > 0:
        open_data = CIFAR100(data_dir, train=True, transform=train_transform, download=True)

    if noise_file is None:
        noise_file = f"cifar10_{noise_rate}_{open_ratio}_sym.json"

    if os.path.exists(noise_file):
        noise = json.load(open(noise_file, "r"))
        noise_labels = noise["noise_labels"]
        closed_noise = noise["closed_noise"]

        if open_ratio>0:
            open_noise = noise["open_noise"]

            for cleanIdx, noisyIdx in open_noise:
                train_set.data[cleanIdx] = (
                    open_data[noisyIdx][0].numpy().transpose(1, 2, 0)
                )

        train_set_copy.targets = np.array(noise_labels)
        train_set.targets = train_set_copy.targets

    else:
        noise_labels = []
        idx = train_idx
        random.shuffle(idx)
        num_total_noise = int(noise_rate * len(train_idx))
        num_open_noise = int(open_ratio * num_total_noise)

        print(
            "Statistics of synthetic noisy CIFAR dataset: ",
            "num of clean samples: ",
            len(train_idx) - num_total_noise,
            " num of closed-set noise: ",
            num_total_noise,
            " num of open-set noise: ",
            num_open_noise,
        )

        target_noise_idx = train_idx
        random.shuffle(target_noise_idx)

        open_noise = list(
            zip(idx[:num_open_noise], target_noise_idx[:num_open_noise]))
        closed_noise = idx[num_open_noise:num_total_noise]

        for i in range(50000):
            if i in closed_noise:
                noiselabel = random.randint(0, 9)
            
                noise_labels.append(noiselabel)
                train_set_copy.targets[i] = noiselabel
            
            else:
                noise_labels.append(train_set_copy.targets[i])
            

        if open_ratio>0:
            for cleanIdx, noisyIdx in open_noise:
                train_set.data[cleanIdx] = (
                    open_data[noisyIdx][0].numpy().transpose(1, 2, 0)
                )

        train_set.targets = train_set_copy.targets
        noise_labels = [int(x) for x in noise_labels]
        clean_idx = list(set(range(len(train_idx))) - set(closed_noise))

        noise = {
            "noise_labels": noise_labels,
            "closed_noise": closed_noise,
            "open_noise": open_noise,
            "clean_idx": clean_idx,
        }

        print(f"Salvando configuração do ruído em {noise_file} ...")
        json.dump(noise, open(noise_file, "w"))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    # Create dataloaders
    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, test_loader, test_loader

class Food101NDataset(Dataset):
    """
    Classe Dataset para carregar o Food-101N a partir dos arquivos
    'verified_train.tsv' e 'verified_val.tsv' e da pasta de imagens.
    """

    def __init__(self, root_dir, use_train=True, use_val=True, transform=None):
        """
        Args:
            root_dir (str): Caminho para o diretório raiz do Food-101N
                            (que contém a pasta 'images' e 'meta').
            use_train (bool): Se True, carrega dados de 'verified_train.tsv'.
            use_val (bool): Se True, carrega dados de 'verified_val.tsv'.
            transform (callable, optional): Transformações a serem aplicadas nas imagens.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.class_to_idx = {}
        classes_file_path = os.path.join(root_dir, "meta", "classes.txt")
        if not os.path.exists(classes_file_path):
            raise FileNotFoundError(
                f"Arquivo classes.txt não encontrado em: {classes_file_path}"
            )
        with open(classes_file_path, "r") as f:
            lines = f.readlines()
            for i, line_content in enumerate(
                lines[1:]
            ):  # Pula a primeira linha (cabeçalho)
                class_name_val = line_content.strip()
                if class_name_val:
                    self.class_to_idx[class_name_val] = i

        self.image_dir = os.path.join(root_dir, "images")
        self.meta_dir = os.path.join(root_dir, "meta")

        if not (use_train or use_val):
            raise ValueError("Pelo menos um 'use_train' ou 'use_val' deve ser True.")

        tsv_files_to_load = []
        if use_train:
            tsv_files_to_load.append(os.path.join(self.meta_dir, "verified_train.tsv"))
        if use_val:
            tsv_files_to_load.append(os.path.join(self.meta_dir, "verified_val.tsv"))

        all_data = []
        for tsv_file in tsv_files_to_load:
            if not os.path.exists(tsv_file):
                raise FileNotFoundError(f"Arquivo TSV não encontrado: {tsv_file}")
            try:
                df = pd.read_csv(
                    tsv_file,
                    sep="\t",
                    header=None,
                    names=["img_path", "verification_label"],
                    skiprows=1,
                )
                all_data.append(df)
            except Exception as e:
                print(f"Erro ao ler {tsv_file}: {e}")
                raise

        self.annotations = pd.concat(all_data, ignore_index=True)
        self.annotations["class_name"] = self.annotations["img_path"].apply(
            lambda x: str(x).split("/")[0]
        )

        valid_class_mask = self.annotations["class_name"].isin(self.class_to_idx.keys())
        if not valid_class_mask.all():
            num_filtered_out = sum(~valid_class_mask)
            print(
                f"Food-101N AVISO: Filtrando {num_filtered_out} amostras devido a nomes de classes desconhecidos derivados do img_path."
            )
            self.annotations = self.annotations[valid_class_mask].reset_index(drop=True)

        if len(self.annotations) == 0:
            raise ValueError(
                "Food-101N ERRO: Nenhuma amostra válida restante após filtrar nomes de classes desconhecidos."
            )

        self.annotations["label_idx"] = self.annotations["class_name"].apply(
            lambda x: self.class_to_idx[x]
        )

        self.image_paths = self.annotations["img_path"].tolist()  # Lista de strings
        self.targets = self.annotations["label_idx"].to_numpy(
            dtype=np.int64
        )  # Array NumPy de targets
        self.verification_labels = self.annotations["verification_label"].to_numpy(
            dtype=np.int64
        )  # Array NumPy

        print(
            f"Food-101N: Carregado {len(self.targets)} amostras após filtragem inicial."
        )
        if len(self.targets) > 0:
            num_clean = np.sum(self.verification_labels == 1)
            num_noisy = np.sum(self.verification_labels == 0)
            print(f"  -> Amostras Limpas (verificadas=1): {num_clean}")
            print(f"  -> Amostras Ruidosas (verificadas=0): {num_noisy}")

    def __len__(self):
        return len(self.targets)  # Baseado no número atual de targets

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_relative_path = self.image_paths[idx]
        label = int(
            self.targets[idx]
        )  # Targets já é um array numpy, converter para int

        img_full_path = os.path.join(self.image_dir, img_relative_path)
        try:
            image = Image.open(img_full_path).convert("RGB")
        except FileNotFoundError:
            print(
                f"Aviso: Imagem não encontrada em {img_full_path}. Pulando item {idx}."
            )
            # Tratar erro de forma mais robusta se necessário
            return self.__getitem__((idx + 1) % len(self))
        except Exception as e:
            print(f"Aviso: Erro ao carregar {img_full_path}: {e}. Pulando item {idx}.")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)
        return image, label


def food101n_dataloaders(
    batch_size=128,
    food101n_dir="datasets/food-101n",
    food101_dir="datasets/food101",
    num_workers=2,
    seed: int = 1,
    no_aug=False,
    mark_verified_noisy_forget: bool = False,
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    print(f"Carregando Food-101N Train+Val Set de {food101n_dir}...")
    train_set = Food101NDataset(
        root_dir=food101n_dir,
        use_train=True,
        use_val=True,
        transform=train_transform,
    )

    if mark_verified_noisy_forget:
        print(
            "INFO: Marcando amostras ruidosas verificadas (verification_label=0) para esquecimento no Food-101N."
        )
        num_marked = 0
        for i in range(len(train_set.targets)):
            if (
                train_set.verification_labels[i] == 0
            ):  # Se a amostra é ruidosa verificada
                original_target = train_set.targets[i]
                if (
                    original_target >= 0
                ):  # Marcar apenas se não estiver já marcado (negativo)
                    train_set.targets[i] = -original_target - 1
                    num_marked += 1
        print(
            f"INFO: {num_marked} amostras marcadas para esquecimento (baseado em verification_label=0)."
        )

    # --- Criar DataLoaders ---
    loader_args = {"num_workers": num_workers, "pin_memory": True}

    def _init_fn(worker_id):
        np.random.seed(int(seed) + worker_id)

    data_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    # Carregar o test_set do Food-101 original para avaliação
    print(f"Carregando Food-101 Test Set de {food101_dir} para avaliação...")
    test_set = Food101(
        root=food101_dir, split="test", transform=test_transform, download=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    # data_loader é o train_loader (se mark_verified_noisy_forget=False)
    # ou o marked_loader (se mark_verified_noisy_forget=True)
    return data_loader, test_loader


def replace_indexes(
    dataset: torch.utils.data.Dataset, indexes, seed=0, only_mark: bool = False
):
    if not only_mark:
        rng = np.random.RandomState(seed)
        new_indexes = rng.choice(
            list(set(range(len(dataset))) - set(indexes)), size=len(indexes)
        )
        dataset.data[indexes] = dataset.data[new_indexes]
        try:
            dataset.targets[indexes] = dataset.targets[new_indexes]
        except:
            dataset.labels[indexes] = dataset.labels[new_indexes]
        else:
            dataset._labels[indexes] = dataset._labels[new_indexes]
    else:
        # Notice the -1 to make class 0 work
        try:
            dataset.targets[indexes] = -dataset.targets[indexes] - 1
        except:
            try:
                dataset.labels[indexes] = -dataset.labels[indexes] - 1
            except:
                dataset._labels[indexes] = -dataset._labels[indexes] - 1


def replace_class(
    dataset: torch.utils.data.Dataset,
    class_to_replace: int,
    num_indexes_to_replace: int = None,
    seed: int = 0,
    only_mark: bool = False,
):
    if class_to_replace == -1:
        try:
            indexes = np.flatnonzero(np.ones_like(dataset.targets))
        except:
            try:
                indexes = np.flatnonzero(np.ones_like(dataset.labels))
            except:
                indexes = np.flatnonzero(np.ones_like(dataset._labels))
    else:
        try:
            indexes = np.flatnonzero(np.array(dataset.targets) == class_to_replace)
        except:
            try:
                indexes = np.flatnonzero(np.array(dataset.labels) == class_to_replace)
            except:
                indexes = np.flatnonzero(np.array(dataset._labels) == class_to_replace)

    if num_indexes_to_replace is not None:
        assert num_indexes_to_replace <= len(indexes), (
            f"Want to replace {num_indexes_to_replace} indexes but only {len(indexes)} samples in dataset"
        )
        rng = np.random.RandomState(seed)
        indexes = rng.choice(indexes, size=num_indexes_to_replace, replace=False)
        print(f"Replacing indexes {indexes}")
    replace_indexes(dataset, indexes, seed, only_mark)


def get_instance_noisy_label(
    n, dataset, labels, num_classes, feature_size, norm_std, seed
):
    # n: Taxa de ruído global (não é fixa por amostra!)
    # dataset: Pares (features, label_verdadeira)
    # labels: Vetor de labels originais
    # num_classes: Número de classes (10 para CIFAR-10)
    # feature_size: Dimensão das features (32x32x3=3072)
    # norm_std: Desvio padrão da distribuição truncada
    # seed: Semente para reprodutibilidade

    # ==============================================
    # 1. Inicialização e Configuração
    # ==============================================
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    P = []  # Armazenará as distribuições de probabilidade

    # Distribuição truncada para flip rates variáveis
    flip_distribution = stats.truncnorm(
        (0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std
    )
    flip_rate = flip_distribution.rvs(len(labels))  # Taxa de erro por amostra

    # ==============================================
    # 2. Matriz de Ruído (Coração do IDN)
    # ==============================================
    # W: Matriz 3D que mapeia features -> ruído específico por classe
    # Dimensões: (num_classes, feature_size, num_classes)
    W = torch.randn(num_classes, feature_size, num_classes, device=device)

    # ==============================================
    # 3. Geração de Rótulos Ruidosos
    # ==============================================
    for i, (x, y) in enumerate(dataset):
        x = x.to(device).float().view(1, -1)  # Feature vector (1x3072)

        # Calcula "afinidade" para classes incorretas
        A = x @ W[y]  # Multiplicação matricial

        # Remove a classe verdadeira
        A[0, y] = -float("inf")

        # Calcula probabilidades escaladas pelo flip rate
        A = flip_rate[i] * F.softmax(A, dim=1)

        # Mantém a classe correta com probabilidade (1 - flip_rate[i])
        A[0, y] += 1 - flip_rate[i]

        P.append(A.cpu().numpy().squeeze())

    # ==============================================
    # 4. Amostragem dos Novos Rótulos
    # ==============================================
    new_label = [np.random.choice(num_classes, p=p) for p in P]

    return np.array(new_label)


if __name__ == "__main__":
    train_loader, val_loader, test_loader = cifar10_dataloaders()
    for i, (img, label) in enumerate(train_loader):
        print(torch.unique(label).shape)
