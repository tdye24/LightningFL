import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))

import wandb
from utils.args import *
from algorithm.fedavg.server import SERVER as FedAvg_SERVER
from algorithm.fedmc.server import SERVER as FedMC_SERVER
from algorithm.fedmc_woat.server import SERVER as FedMC_WOAT_SERVER
from algorithm.fedprox.server import SERVER as FedProx_SERVER
from algorithm.fedsp.server import SERVER as FedSP_SERVER
from algorithm.lgfedavg.server import SERVER as LG_FedAvg_SERVER


if __name__ == '__main__':
    wandb.init(entity='tdye24', project='LightningFL')

    args = parse_args()

    # algorithm = args.algorithm
    # dataset_name = args.dataset
    # model_name = args.model
    # num_rounds = args.num_rounds
    # eval_interval = args.eval_interval
    # clients_per_round = args.clients_per_round
    # epoch = args.epoch
    # batch_size = args.batch_size
    # lr = args.lr
    # lr_decay = args.lr_decay
    # decay_step = args.decay_step
    # alpha = args.alpha
    # seed = args.seed
    # cuda = args.cuda
    wandb.watch_called = False
    config = wandb.config
    DROPOUTS = None
    if config.dataset == 'cifar100':
        DROPOUTS = {'small': [0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
                    'big': [0.75, 0.75, 0.9, 0.9, 0.9, 0.5]}
    elif config.dataset == 'cifar10' or config.dataset == 'cifar10_diri':
        if config.algorithm in ['fedsp', 'fedmc', 'fedmc_woat']:
            DROPOUTS = {'small': [0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
                        'big': [0.5, 0.5, 0.9, 0.9, 0.9, 0.5]}
        elif config.algorithm in ['lgfedavg']:
            DROPOUTS = {'small': [0.25, 0.25, 0.5],
                        'big': [0.5, 0.5, 0.75]}
    elif config.dataset == 'femnist':
        DROPOUTS = {'small': [0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
                    'big': [0.75, 0.75, 0.9, 0.9, 0.9, 0.5]}
    else:
        DROPOUTS = {'small': [0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
                    'big': [0.75, 0.75, 0.9, 0.9, 0.9, 0.5]}
    if DROPOUTS is not None:
        args.dropout = DROPOUTS[args.drop]
    config.update(args)

    server = None
    if config.algorithm == 'fedavg':
        server = FedAvg_SERVER(config=config)
    elif config.algorithm == 'fedmc':
        server = FedMC_SERVER(config=config)
    elif config.algorithm == 'fedprox':
        server = FedProx_SERVER(config=config)
    elif config.algorithm == 'fedsp':
        server = FedSP_SERVER(config=config)
    elif config.algorithm == 'lgfedavg':
        server = LG_FedAvg_SERVER(config=config)
    elif config.algorithm == 'fedmc_woat':
        server = FedMC_WOAT_SERVER(config=config)
    server.federate()
    if config.algorithm in ['fedmc', 'fedmc_woat']:
        server.save_model()
