import json
import os
import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class CIFAR10_DATASET(Dataset):
    def __init__(self, dataset, ids, transform=None):
        self.dataset = dataset
        self.ids = ids
        self.transform = transform

    def __getitem__(self, item):
        x, y = self.dataset[self.ids[item]]
        assert x.shape == (3, 32, 32)

        # x = torch.tensor(x).float()
        y = torch.tensor(y).long()

        return x, y

    def __len__(self):
        return len(self.ids)


def get_cifar10_dataLoaders(batch_size=10, train_transform=None, test_transform=None):
    HOME = '/home/tdye/LightningFL/data/cifar10'

    train_data = datasets.CIFAR10(root=HOME, train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR10(root=HOME, train=False, transform=test_transform, download=True)

    print("loading latent distribution based partition cifar10 dataset")
    print("len of training data", len(train_data), "len of test data", len(test_data))

    with open(os.path.join(HOME, 'latent_distribution.json')) as f:
        client_ids = json.load(f)

    trainLoaders = {}
    testLoaders = {}

    for client_id, ids in client_ids.items():
        train_ids = ids['train']
        test_ids = ids['test']

        train_set = CIFAR10_DATASET(dataset=train_data, ids=train_ids, transform=trainTransform)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        test_set = CIFAR10_DATASET(dataset=test_data, ids=test_ids, transform=testTransform)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=4)
        trainLoaders[client_id] = train_loader
        testLoaders[client_id] = test_loader

    all_clients = list(client_ids.keys())
    return all_clients, trainLoaders, testLoaders


if __name__ == '__main__':
    trainTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    testTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    clients, _trainLoaders, _testLoaders = get_cifar10_dataLoaders(batch_size=50,
                                                                   train_transform=trainTransform,
                                                                   test_transform=testTransform)
    for client in clients:
        print("client", client)
        ls = []
        for _, (data, labels) in enumerate(_trainLoaders[client]):
            ls.extend(list(np.array(torch.unique(labels))))
        print("train", np.unique(np.array(ls)))
        ls = []
        for _, (data, labels) in enumerate(_testLoaders[client]):
            ls.extend(list(np.array(torch.unique(labels))))
        print("test", np.unique(np.array(ls)))
        print("=========")
