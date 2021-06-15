import torch
import wandb
from utils.utils import parse_args, setup_seed
from algorithm.fedavg.server import SERVER


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

    # use_cuda = config.cuda and torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    setup_seed(config.seed)
    server = SERVER(config=config)
    server.federate()
    # for epoch in range(1, config.epochs + 1):
    #     print(f"epoch: {epoch}, training")
    #     train_summary = train(model, device, train_loader, optimizer)
    #     print(f"epoch: {epoch}, test")
    #     test_summary = test(config, model, device, test_loader, classes)
    #
    #     summary = {}
    #     summary.update(train_summary)
    #     summary.update(test_summary)
    #     wandb.log(summary)
    #
    # # WandB – Save the model checkpoint. This automatically saves a file to the cloud and associates it with the
    # # current run.
    # torch.save(model.state_dict(), "model.h5")
    # wandb.save('model.h5')
