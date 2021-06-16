import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))

import wandb
from utils import *
from algorithm.fedavg.server import SERVER as FedAvg_SERVER
from algorithm.fedmc.server import SERVER as FedMC_SERVER


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
    config.update(args)

    server = None
    if config.algorithm == 'fedavg':
        server = FedAvg_SERVER(config=config)
    elif config.algorithm == 'fedmc':
        server = FedMC_SERVER(config=config)
    server.federate()
