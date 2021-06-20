from .fedavg.mnist.MNIST import MNIST as FedAvg_MNIST
from .fedavg.cifar10.CIFAR10 import CIFAR10 as FedAvg_CIFAR10

from .fedsp.cifar10.CIFAR10 import CIFAR10 as FedSP_CIFAR10
from .fedmc.cifar10.CIFAR10 import CIFAR10 as FedMC_CIFAR10

__all__ = [
    'FedAvg_MNIST',
    'FedAvg_CIFAR10',
    'FedMC_CIFAR10',
    'FedSP_CIFAR10'
]
