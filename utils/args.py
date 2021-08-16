import argparse

ALGORITHMS = ['fedavg', 'fedmc', 'fedprox', 'fedsp', 'lgfedavg', 'fedmc_woat']
DATASETS = ['cifar10', 'mnist', 'cifar100', 'har', 'femnist', 'cifar10_diri', 'cifar10_ld']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm',
                        help='algorithm',
                        choices=ALGORITHMS,
                        required=True)

    parser.add_argument('--dataset',
                        help='name of dataset',
                        choices=DATASETS,
                        required=True)

    parser.add_argument('--model',
                        help='name of model',
                        type=str,
                        required=True)

    parser.add_argument('--numRounds',
                        help='# of communication round',
                        type=int,
                        default=100)

    parser.add_argument('--evalInterval',
                        help='communication rounds between two evaluation',
                        type=int,
                        default=1)

    parser.add_argument('--clientsPerRound',
                        help='# of selected clients per round',
                        type=int,
                        default=1)

    parser.add_argument('--epoch',
                        help='# of epochs when clients train on data',
                        type=int,
                        default=1)

    parser.add_argument('--batchSize',
                        help='batch size when clients train on data',
                        type=int,
                        default=1)

    parser.add_argument('--lr',
                        help='learning rate for local optimizers',
                        type=float,
                        default=3e-4)

    parser.add_argument('--lrDecay',
                        help='decay rate for learning rate',
                        type=float,
                        default=0.99)

    parser.add_argument('--decayStep',
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

    parser.add_argument('--mu',
                        help='coefficient for balancing cross entropy loss and critic loss',
                        type=float,
                        default=0.1)

    parser.add_argument('--omega',
                        help='coefficient for balancing w-distance loss and gradient penalty loss',
                        type=float,
                        default=0.1)

    parser.add_argument('--diffCo',
                        help='coefficient for balancing classification loss and regularizer loss (model difference) in FedProx',
                        type=float,
                        default=0.1)

    parser.add_argument('--depth',
                        help='depth for shared layers in lg-fedavg',
                        type=int,
                        default=1)

    parser.add_argument('--mode',
                        help='Integration of global encoder and local encoder',
                        choices=['concat', 'addition'],
                        default='concat')

    parser.add_argument('--earlyStopping',
                        help='do early-stopping for fedsp or fedmc',
                        type=bool,
                        default=False)

    parser.add_argument('--drop',
                        help='dropout for cifar10 and cifar100 model, (shared global encoder drop1, shared global encoder drop2, private local encoder drop1, private local encoder drop2, clf drop, critic drop)',
                        choices=['small', 'big'],
                        default='small')

    return parser.parse_args()
