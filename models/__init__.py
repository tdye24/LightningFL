from .fedavg.mnist.MNIST import MNIST as FedAvg_MNIST
from .fedavg.cifar10.CIFAR10 import CIFAR10 as FedAvg_CIFAR10

__all__ = [
    'FedAvg_MNIST',
    'FedAvg_CIFAR10'
]
