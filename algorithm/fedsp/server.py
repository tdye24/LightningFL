import copy
import wandb
import numpy as np
from utils.flutils import *
from utils.tools import *
from algorithm.fedsp.client import CLIENT
from tqdm import tqdm
from prettytable import PrettyTable


class SERVER:
    def __init__(self, config):
        self.config = config
        self.clients = self.setup_clients()
        self.surrogates = self.setup_surrogates()
        self.clientsTrainSamplesNum = {client.user_id: client.trainSamplesNum for client in self.clients}
        self.clientsTestSamplesNum = {client.user_id: client.testSamplesNum for client in self.clients}
        self.selected_clients = []
        self.losses = []
        self.accs = []
        self.updates = []
        # affect server initialization
        setup_seed(config.seed)
        kwargs = {'dropout': self.config.dropout}
        self.model = select_model(algorithm=self.config.algorithm, model_name=self.config.model, mode=self.config.mode,
                                  **kwargs)
        self.params = self.model.state_dict()
        self.optimal = {
            'round': 0,
            'trainingAcc': -1.0,
            'testAcc': -1.0,
            'trainingLoss': 10e8,
            'testLoss': 10e8,
            'params': None
        }

    def setup_clients(self):
        users, trainLoaders, testLoaders = setup_datasets(dataset=self.config.dataset,
                                                          batch_size=self.config.batchSize,
                                                          alpha=self.config.alpha)
        clients = [
            CLIENT(user_id=user_id,
                   trainLoader=trainLoaders[user_id],
                   testLoader=testLoaders[user_id],
                   config=self.config)
            for user_id in users]
        return clients

    def select_clients(self, round_th):
        np.random.seed(seed=self.config.seed + round_th)
        return np.random.choice(self.clients, self.config.clientsPerRound, replace=False)

    def setup_surrogates(self):
        surrogates = [
            CLIENT(user_id=i,
                   trainLoader=None,
                   testLoader=None,
                   config=self.config)
            for i in range(self.config.clientsPerRound)]
        return surrogates

    def clear(self):
        self.selected_clients = []
        self.losses = []
        self.accs = []
        self.updates = []

    def federate(self):
        print(f"Training with {len(self.clients)} clients!")
        for i in tqdm(range(self.config.numRounds)):
            self.selected_clients = self.select_clients(round_th=i)
            for k in range(len(self.selected_clients)):
                surrogate = self.surrogates[k]
                c = self.selected_clients[k]
                # surrogate <-- c
                surrogate.update(c)
                surrogate.set_shared_params(self.params)
                trainSamplesNum, update, loss = surrogate.train(round_th=i)
                # c <-- surrogate
                c.update(surrogate)
                self.updates.append((trainSamplesNum, copy.deepcopy(update)))

            # update global params
            self.params = fedAverage(self.updates)

            if i == 0 or (i + 1) % self.config.evalInterval == 0:
                # print(f"\nRound {i}")
                # test on training set
                trainingAccList, trainingLossList = self.test(dataset='train')
                # test on test set
                testAccList, testLossList = self.test(dataset='test')

                # print and log
                self.printAndLog(i, trainingAccList, testAccList, trainingLossList, testLossList)

            self.clear()

    def test(self, dataset='test'):
        accList, lossList = [], []
        surrogate = self.surrogates[0]
        for c in self.clients:
            surrogate.update(c)
            surrogate.set_shared_params(self.params)
            samplesNum, acc, loss = surrogate.test(dataset=dataset)
            accList.append((samplesNum, acc))
            lossList.append((samplesNum, loss))
        return accList, lossList

    def printAndLog(self, round_th, trainingAccList, testAccList, trainingLossList, testLossList):
        trainingAcc = avgMetric(trainingAccList)
        trainingLoss = avgMetric(trainingLossList)
        testAcc = avgMetric(testAccList)
        testLoss = avgMetric(testLossList)

        # post data error, encoder error, trainingAcc. format
        summary = {
            "round": round_th,
            "TrainingAcc": trainingAcc,
            "TestAcc": testAcc,
            "TrainingLoss": trainingLoss,
            "TestLoss": testLoss
        }
        wandb.log(summary)

        # table = PrettyTable(['TrainingAcc.', 'TestAcc.', 'TrainingLoss.', 'TestLoss.'])
        #
        # if trainingAcc > self.optimal['trainingAcc']:
        #     self.optimal['trainingAcc'] = trainingAcc
        #     trainingAcc = "\033[1;31m" + f"{round(trainingAcc, 3)}" + "\033[0m"
        # else:
        #     trainingAcc = round(trainingAcc, 3)
        # if testAcc > self.optimal['testAcc']:
        #     self.optimal['testAcc'] = testAcc
        #     testAcc = "\033[1;31m" + f"{round(testAcc, 3)}" + "\033[0m"
        # else:
        #     testAcc = round(testAcc, 3)
        # if trainingLoss < self.optimal['trainingLoss']:
        #     self.optimal['trainingLoss'] = trainingLoss
        #     trainingLoss = "\033[1;31m" + f"{round(trainingLoss, 3)}" + "\033[0m"
        # else:
        #     trainingLoss = round(trainingLoss, 3)
        # if testLoss < self.optimal['testLoss']:
        #     self.optimal['testLoss'] = testLoss
        #     testLoss = "\033[1;31m" + f"{round(testLoss, 3)}" + "\033[0m"
        # else:
        #     testLoss = round(testLoss, 3)
        # table.add_row([trainingAcc, testAcc, trainingLoss, testLoss])
        # print(table)
