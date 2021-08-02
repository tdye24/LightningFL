import torch
from torchvision.transforms import transforms

# datasets
from data.mnist.mnist import get_mnist_dataLoaders
from data.cifar10.cifar10 import get_cifar10_dataLoaders
from data.cifar100.cifar100 import get_cifar100_dataLoaders
from data.femnist.femnist import get_femnist_dataLoaders
from data.har.har import get_har_dataLoaders

# models
from models import *


def setup_datasets(dataset, batch_size):
    users, trainLoaders, testLoaders = [], [], []
    if dataset == 'mnist':
        trainTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        testTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        users, trainLoaders, testLoaders = get_mnist_dataLoaders(batch_size=batch_size, train_transform=trainTransform, test_transform=testTransform)
    elif dataset == 'cifar10':
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
    elif dataset == 'femnist':
        trainTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        testTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        users, trainLoaders, testLoaders = get_femnist_dataLoaders(batch_size=batch_size,
                                                                   train_transform=trainTransform,
                                                                   test_transform=testTransform)
    elif dataset == 'har':
        users, trainLoaders, testLoaders = get_har_dataLoaders(batch_size=10)

    return users, trainLoaders, testLoaders


def select_model(algorithm, model_name, mode='concat', **kwargs):
    model = None
    if algorithm == 'fedavg' or algorithm == 'fedprox':
        if model_name == 'mnist':
            model = FedAvg_MNIST()
        elif model_name == 'cifar10':
            model = FedAvg_CIFAR10()
        elif model_name == 'cifar100':
            model = FedAvg_CIFAR100()
        elif model_name == 'har':
            model = FedAvg_HAR()
        else:
            print(f"Unimplemented Model {model_name}")
    elif algorithm == 'fedmc' or algorithm == 'fedmc_woat':
        if model_name == 'mnist':
            if mode == 'concat':
                model = FedMC_MNIST()
            else:
                print(f"Unimplemented Mode {mode} for FedMC")
        elif model_name == 'cifar10':
            if mode == 'concat':
                model = FedMC_CIFAR10(dropout=kwargs['dropout'])
            elif mode == 'addition':
                model = FedMC_CIFAR10_Add(dropout=kwargs['dropout'])
            else:
                print(f"Unimplemented Mode {mode} for FedMC")
        elif model_name == 'cifar100':
            if mode == 'concat':
                model = FedMC_CIFAR100(dropout=kwargs['dropout'])
            elif mode == 'addition':
                model = FedMC_CIFAR100_Add(dropout=kwargs['dropout'])
            else:
                print(f"Unimplemented Mode {mode} for FedMC")
        elif model_name == 'har':
            model = FedMC_HAR()
    elif algorithm == 'fedsp':
        if model_name == 'cifar10':
            if mode == 'concat':
                model = FedSP_CIFAR10(dropout=kwargs['dropout'])
            elif mode == 'addition':
                model = FedSP_CIFAR10_Add(dropout=kwargs['dropout'])
            else:
                print(f"Unimplemented Mode {mode} for FedSP")
        elif model_name == 'cifar100':
            if mode == 'concat':
                model = FedSP_CIFAR100(dropout=kwargs['dropout'])
            elif mode == 'addition':
                model = FedSP_CIFAR100_Add(dropout=kwargs['dropout'])
            else:
                print(f"Unimplemented Mode {mode} for FedSP")
        elif model_name == 'mnist':
            if mode == 'concat':
                model = FedSP_MNIST()
        elif model_name == 'har':
            model = FedSP_HAR()
    elif algorithm == 'lgfedavg':
        if model_name == 'mnist':
            model = LG_FedAvg_MNIST()
        elif model_name == 'cifar10':
            model = LG_FedAvg_CIFAR10()
        elif model_name == 'cifar100':
            model = LG_FedAvg_CIFAR100()
        elif model_name == 'har':
            model = LG_FedAvg_HAR()
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
