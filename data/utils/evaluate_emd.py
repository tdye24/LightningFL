import numpy as np
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../')))

from pyemd import emd
from data.mnist.mnist import get_mnist_dataLoaders as get_mnist_dataLoaders
from data.cifar10.cifar10 import get_cifar10_dataLoaders as get_cifar10_dataLoaders
from data.cifar100.cifar100 import get_cifar100_dataLoaders as get_cifar100_dataLoaders
from data.har.har import get_har_dataLoaders as get_har_dataLoaders
from data.femnist.femnist import get_femnist_dataLoaders as get_femnist_dataLoaders
from data.cifar10.cifar10_diri import get_cifar10_dirichlet_dataloaders as get_cifar10_dirichlet_dataLoaders

parser = argparse.ArgumentParser()

parser.add_argument('--name',
                    help='name of dataset to evaluate; default: cifar10;',
                    type=str,
                    choices=['mnist', 'mnist_fedml', 'mnist_ld', 'cifar10', 'cifar10_diri', 'cifar10_ld', 'cifar100', 'har', 'femnist'],
                    default='cifar10')

parser.add_argument('--alpha',
                    help='alpha for dirichlet distribution',
                    type=float,
                    default=0.5)

args = parser.parse_args()
dataset = args.name

N_class = None
users, trainLoaders, testLoaders = None, None, None

if dataset == 'mnist':
    users, trainLoaders, testLoaders = get_mnist_dataLoaders(batch_size=10)
    N_class = 10
elif dataset == 'cifar10':
    users, trainLoaders, testLoaders = get_cifar10_dataLoaders(batch_size=10)
    N_class = 10
elif dataset == 'cifar10_diri':
    users, trainLoaders, testLoaders = get_cifar10_dirichlet_dataLoaders(batch_size=10, alpha=args.alpha)
    N_class = 10
elif dataset == 'cifar100':
    users, trainLoaders, testLoaders = get_cifar100_dataLoaders(batch_size=10)
    N_class = 100
elif dataset == 'har':
    users, trainLoaders, testLoaders = get_har_dataLoaders(batch_size=10)
    N_class = 6
elif dataset == 'femnist':
    users, trainLoaders, testLoaders = get_femnist_dataLoaders(batch_size=10)
    N_class = 62
else:
    pass

total_dataset_distribution = {i: 0 for i in range(N_class)}
total_dataset_num_samples = 0
users_histogram = []
for user in users:

    train_loader = trainLoaders[user]
    distribution = {i: 0 for i in range(N_class)}

    num_samples = len(train_loader.sampler)
    total_dataset_num_samples += num_samples

    for step, (data, labels) in enumerate(train_loader):
        for label in labels:
            distribution[label.item()] += 1

    histogram = []
    for label, num in distribution.items():
        total_dataset_distribution[label] += num
        histogram.append(num / num_samples)
    users_histogram.append((num_samples, histogram))

total_dataset_histogram = np.zeros(N_class)
for num, histogram in users_histogram:
    total_dataset_histogram = np.add(total_dataset_histogram, num * np.array(histogram))

total_dataset_histogram /= total_dataset_num_samples
# print(total_dataset_histogram)

distances = []

for num, histogram in users_histogram:
    # print(total_dataset_histogram, histogram)
    d = np.linalg.norm(total_dataset_histogram - histogram, ord=1)
    distances.append(d)

print("avg emd: ", sum(distances) / len(distances))
