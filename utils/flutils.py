
from torchvision.transforms import transforms


# datasets
from data.cifar10.cifar10 import get_cifar10_dataLoaders
from data.cifar100.cifar100 import get_cifar100_dataLoaders

# models
from models import *


def setup_datasets(dataset, batch_size):
    users, trainLoaders, testLoaders = [], [], []
    if dataset == 'cifar10':
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
        users, trainLoaders, testLoaders = get_cifar10_dataLoaders(batch_size=batch_size,
                                                                   train_transform=trainTransform,
                                                                   test_transform=testTransform)
    elif dataset == 'cifar100':
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
        users, trainLoaders, testLoaders = get_cifar100_dataLoaders(batch_size=batch_size,
                                                                   train_transform=trainTransform,
                                                                   test_transform=testTransform)

    return users, trainLoaders, testLoaders


def select_model(algorithm, model_name):
    model = None
    if algorithm == 'fedavg' or algorithm == 'fedprox':
        if model_name == 'mnist':
            model = FedAvg_MNIST()
        elif model_name == 'cifar10':
            model = FedAvg_CIFAR10()
        elif model_name == 'cifar100':
            model = FedAvg_CIFAR100()
        else:
            print(f"Unimplemented Model {model_name}")
    elif algorithm == 'fedmc':
        if model_name == 'cifar10':
            model = FedMC_CIFAR10()
    elif algorithm == 'fedsp':
        if model_name == 'cifar10':
            model = FedSP_CIFAR10()
    else:
        print(f"Unimplemented Algorithm {algorithm}")
    return model


def fedAverage(updates):
    total_weight = 0
    (clientSamplesNum, new_params) = updates[0]

    for (clientSamplesNum, client_params) in updates:
        total_weight += clientSamplesNum

    for k in new_params.keys():
        for i in range(0, len(updates)):
            client_samples, client_params = updates[i]
            # weight
            w = client_samples / total_weight
            if i == 0:
                new_params[k] = client_params[k] * w
            else:
                new_params[k] += client_params[k] * w
    # return global model params
    return new_params


def avgMetric(metricList):
    total_weight = 0
    total_metric = 0
    for (samplesNum, metric) in metricList:
        total_weight += samplesNum
        total_metric += samplesNum * metric
    average = total_metric / total_weight

    return average
