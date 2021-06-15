import argparse
import torch
import numpy as np
import random
from torchvision.transforms import transforms


# datasets
from data.cifar10.cifar10 import get_cifar10_dataLoaders

# models
# FedAvg
from models import *

ALGORITHMS = ['fedavg']
DATASETS = ['cifar10']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-algorithm',
                        help='algorithm',
                        choices=ALGORITHMS,
                        required=True)

    parser.add_argument('-dataset',
                        help='name of dataset',
                        choices=DATASETS,
                        required=True)

    parser.add_argument('-model',
                        help='name of model',
                        type=str,
                        required=True)

    parser.add_argument('--num-rounds',
                        help='# of communication round',
                        type=int,
                        default=100)

    parser.add_argument('--eval-interval',
                        help='communication rounds between two evaluation',
                        type=int,
                        default=1)

    parser.add_argument('--clients-per-round',
                        help='# of selected clients per round',
                        type=int,
                        default=1)

    parser.add_argument('--epoch',
                        help='# of epochs when clients train on data',
                        type=int,
                        default=1)

    parser.add_argument('--batch-size',
                        help='batch size when clients train on data',
                        type=int,
                        default=1)

    parser.add_argument('--lr',
                        help='learning rate for local optimizers',
                        type=float,
                        default=3e-4)

    parser.add_argument('--lr-decay',
                        help='decay rate for learning rate',
                        type=float,
                        default=0.99)

    parser.add_argument('--decay-step',
                        help='decay rate for learning rate',
                        type=int,
                        default=200)

    parser.add_argument('--alpha',
                        help='alpha for dirichlet distribution partition',
                        type=float,
                        default=0.5)

    parser.add_argument('--seed',
                        help='seed for random client sampling and batch splitting',
                        type=int,
                        default=24)

    parser.add_argument('--cuda',
                        help='using cuda',
                        type=bool,
                        default=True)
    return parser.parse_args()


def setup_seed(rs):
    """
    set random seed for reproducing experiments
    :param rs: random seed
    :return: None
    """
    torch.manual_seed(rs)
    torch.cuda.manual_seed_all(rs)
    np.random.seed(rs)
    random.seed(rs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    return users, trainLoaders, testLoaders


def select_model(algorithm, model_name):
    model = None
    if algorithm == 'fedavg':
        if model_name == 'mnist':
            model = FedAvg_MNIST()
        elif model_name == 'cifar10':
            model = FedAvg_CIFAR10()
        else:
            print(f"Unimplemented Model {model_name}")
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
