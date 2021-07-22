from .fedavg.mnist.MNIST import MNIST as FedAvg_MNIST
from .fedavg.cifar10.CIFAR10 import CIFAR10 as FedAvg_CIFAR10
from .fedavg.cifar100.CIFAR100 import CIFAR100 as FedAvg_CIFAR100

from .fedsp.cifar10.CIFAR10 import CIFAR10 as FedSP_CIFAR10
from .fedsp.cifar10.CIFAR10_ADDITION import CIFAR10 as FedSP_CIFAR10_Add
from .fedsp.cifar100.CIFAR100 import CIFAR100 as FedSP_CIFAR100
from .fedsp.cifar100.CIFAR100_ADDITION import CIFAR100 as FedSP_CIFAR100_Add
from .fedsp.cifar100.CIFAR100_DropLocal import CIFAR100 as FedSP_CIFAR100_DropLocal

from .fedmc.cifar10.CIFAR10 import CIFAR10 as FedMC_CIFAR10
from .fedmc.cifar10.CIFAR10_ADDITION import CIFAR10 as FedMC_CIFAR10_Add
from .fedmc.cifar100.CIFAR100 import CIFAR100 as FedMC_CIFAR100
from .fedmc.cifar100.CIFAR100_ADDITION import CIFAR100 as FedMC_CIFAR100_Add
from .fedmc.cifar100.CIFAR100_DropLocal import CIFAR100 as FedMC_CIFAR100_DropLocal

from .lgfedavg.mnist.MNIST import MNIST as LG_FedAvg_MNIST
from .lgfedavg.cifar10.CIFAR10 import CIFAR10 as LG_FedAvg_CIFAR10
from .lgfedavg.cifar100.CIFAR100 import CIFAR100 as LG_FedAvg_CIFAR100

__all__ = [
    'FedAvg_MNIST',
    'FedAvg_CIFAR10',
    'FedAvg_CIFAR100',
    'FedMC_CIFAR10',
    'FedMC_CIFAR10_Add',
    'FedMC_CIFAR100',
    'FedMC_CIFAR100_Add',
    'FedMC_CIFAR100_DropLocal',
    'FedSP_CIFAR10',
    'FedSP_CIFAR10_Add',
    'FedSP_CIFAR100',
    'FedSP_CIFAR100_Add',
    'FedSP_CIFAR100_DropLocal',
    'LG_FedAvg_MNIST',
    'LG_FedAvg_CIFAR10',
    'LG_FedAvg_CIFAR100'
]
