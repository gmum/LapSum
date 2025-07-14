import random
from statistics import median

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def get_multi_mnist_input(l, n, low, high, digset):
    multi_mnist_sequences = []
    values = []
    for i in range(n):
        mnist_digits = []
        num = random.randint(low, high)
        values.append(num)

        for i in range(l):
            digit = num % 10
            num //= 10
            ref = digset[digit]
            mnist_digits.insert(0, ref[np.random.randint(0, ref.shape[0])])
        multi_mnist_sequence = np.concatenate(mnist_digits)
        multi_mnist_sequence = np.reshape(multi_mnist_sequence, (-1, 28))
        multi_mnist_sequences.append(multi_mnist_sequence)
    multi_mnist_batch = np.stack(multi_mnist_sequences)
    vals = np.array(values)
    med = int(median(values))
    arg_med = np.equal(vals, med).astype('float32')
    arg_med_sum = np.sum(arg_med)
    if arg_med_sum:
        arg_med /= np.sum(arg_med)
    return multi_mnist_batch, med, arg_med, vals


class MultiMNISTDataset(Dataset):
    def __init__(self, l, n, window_size, digits_set, num_elements):
        self.l = l
        self.n = n
        self.window_size = window_size
        self.digits_set = digits_set
        self.low, self.high = 0, 10 ** l - 1
        self.num_elements = num_elements

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        window_begin = random.randint(self.low, self.high - self.window_size)
        return get_multi_mnist_input(self.l, self.n, window_begin, window_begin + self.window_size, self.digits_set)


def get_iterators(l, n, window_size, minibatch_size=None,
                  path2data='MNIST_data', download_data=False, num_workers=4, seed=42):
    val_set_size = 10000

    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(path2data, train=True, download=download_data, transform=transform)
    mnist_test = datasets.MNIST(path2data, train=False, download=download_data, transform=transform)
    train_set_size = len(mnist_train) - val_set_size
    print(f"Training/validation/test splits of {train_set_size}/{val_set_size}/{len(mnist_test)} samples.")

    train_indices, val_indices = train_test_split(
        np.arange(mnist_train.targets.size(0)),
        test_size=val_set_size,
        stratify=mnist_train.targets,
        random_state=seed
    )

    train_digits = [[] for _ in range(len(mnist_train.classes))]
    validation_digits = [[] for _ in range(len(mnist_train.classes))]
    test_digits = [[] for _ in range(len(mnist_train.classes))]

    for idx in train_indices:
        digit, idx_class = mnist_train[idx]
        train_digits[idx_class].append(digit)

    for idx in val_indices:
        digit, idx_class = mnist_train[idx]
        validation_digits[idx_class].append(digit)

    for digit, idx_class in mnist_test:
        test_digits[idx_class].append(digit)

    for i in range(len(mnist_train.classes)):
        train_digits[i] = torch.cat(train_digits[i], dim=0)
        validation_digits[i] = torch.cat(validation_digits[i], dim=0)
        test_digits[i] = torch.cat(test_digits[i], dim=0)

    num = (train_set_size // (l * minibatch_size)) * minibatch_size if minibatch_size else train_set_size // l
    train_dataset = MultiMNISTDataset(l, n, window_size, train_digits, num)
    num = (val_set_size // (l * minibatch_size)) * minibatch_size if minibatch_size else val_set_size // l
    val_dataset = MultiMNISTDataset(l, n, window_size, validation_digits, num)
    num = (len(mnist_test) // (l * minibatch_size)) * minibatch_size if minibatch_size else len(mnist_test) // l
    test_dataset = MultiMNISTDataset(l, n, window_size, test_digits, num)

    train_loader = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=minibatch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=minibatch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def transform_target(digits, descending=False):
    """
    If descending=False returns indexes from smallest to largest relative
    to the second dimension (the largest number has the least weight).
    If descending=True returns indexes from smallest to largest (the largest number has the most weight).
    """
    factory_kwargs = {"device": digits.device, "dtype": digits.dtype}
    bs, n = digits.shape
    _, indices = torch.sort(digits, dim=-1, descending=descending)
    targets = torch.arange(n, **factory_kwargs) * torch.ones(bs, 1, **factory_kwargs)
    targets = torch.empty_like(digits).scatter_(1, indices, targets)
    return targets

# train_loader, val_loader, test_loader = get_iterators(5, 10, 100, minibatch_size=32)
