import copy
import torch
from utils.flutils import *
import torch.optim as optim
import numpy as np


class CLIENT:
    def __init__(self, user_id, trainLoader, testLoader, config):
        self.config = config
        self.user_id = user_id
        use_cuda = config.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = select_model(algorithm=config.algorithm, model_name=config.model)
        self.trainLoader = trainLoader
        self.testLoader = testLoader
        self.startPoint = None

    @property
    def trainSamplesNum(self):
        return len(self.trainLoader) if self.trainLoader else None

    @property
    def testSamplesNum(self):
        return len(self.testLoader) if self.testLoader else None

    @staticmethod
    def model_difference(start_point, new_point):
        loss = 0
        old_params = start_point.state_dict()
        for name, param in new_point.named_parameters():
            loss += torch.norm(old_params[name] - param, 2)
        return loss

    def train(self, round_th):
        model = self.model
        model.to(self.device)

        self.startPoint = copy.deepcopy(model)

        model.train()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(params=model.parameters(),
                              lr=self.config.lr * self.config.lrDecay ** (round_th / self.config.decayStep),
                              weight_decay=1e-4)

        meanLoss = []
        for epoch in range(self.config.epoch):
            for step, (data, labels) in enumerate(self.trainLoader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                difference = self.model_difference(self.startPoint, model)
                loss = criterion(output, labels) + self.config.diffCo * difference
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                meanLoss.append(loss.item())

        trainSamplesNum, update = self.trainSamplesNum, self.get_params()

        if np.isnan(sum(meanLoss) / len(meanLoss)):
            print(f"client {self.user_id}, loss NAN")
            exit(0)

        return trainSamplesNum, update, sum(meanLoss) / len(meanLoss)

    def test(self, dataset='test'):
        model = self.model
        model.eval()
        model.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()

        if dataset == 'test':
            dataLoader = self.testLoader
        else:
            dataLoader = self.trainLoader

        total_right = 0
        total_samples = 0
        meanLoss = []
        with torch.no_grad():
            for step, (data, labels) in enumerate(dataLoader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                output = model(data)
                loss = criterion(output, labels)
                meanLoss.append(loss.item())
                output = torch.argmax(output, dim=-1)
                total_right += torch.sum(output == labels)
                total_samples += len(labels)
            acc = float(total_right) / total_samples

        return total_samples, acc, sum(meanLoss) / len(meanLoss)

    def get_params(self):
        return self.model.cpu().state_dict()

    def set_params(self, model_params):
        self.model.load_state_dict(model_params)

    def set_shared_params(self, params):
        tmp_params = self.get_params()
        for (key, value) in params.items():
            if key.startswith('shared'):
                tmp_params[key] = value
        self.set_params(tmp_params)

    def update(self, client):
        self.model.load_state_dict(client.model.state_dict())
        self.trainLoader = client.trainLoader
        self.testLoader = client.testLoader
