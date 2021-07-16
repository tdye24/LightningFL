import torch
from utils.flutils import *
from torch import autograd
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

    @property
    def trainSamplesNum(self):
        return len(self.trainLoader) if self.trainLoader else None

    @property
    def testSamplesNum(self):
        return len(self.testLoader) if self.testLoader else None

    def calc_gradient_penalty(self, model, real_data, fake_data):
        assert not (real_data.requires_grad or fake_data.requires_grad)
        alpha = torch.rand(real_data.shape[0], 1)
        alpha = alpha.expand(self.config.batchSize, real_data.shape[1]).contiguous()
        alpha = alpha.to(self.device)
        interpolates = alpha * real_data + ((torch.ones_like(alpha) - alpha) * fake_data)
        interpolates = interpolates.to(self.device)
        interpolates = interpolates.clone().detach().requires_grad_(True)

        critic_interpolates = model.metaCritic(interpolates)

        gradients = autograd.grad(outputs=critic_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(critic_interpolates.size()).to(self.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def metaTrain(self, round_th):
        model = self.model
        model.to(self.device)
        model.train()

        # frozen
        for (key, param) in model.named_parameters():
            if key.startswith('critic'):
                param.requires_grad = False

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
                gFeature, lFeature, gValue, lValue, output = model(data)
                clf_loss = criterion(output, labels)
                WD = (gValue - lValue).mean()
                gradient_penalty = self.calc_gradient_penalty(model, gFeature.data, lFeature.data)
                loss = clf_loss + self.config.mu * (- WD + self.config.omega * gradient_penalty)
                loss.backward()
                optimizer.step()
                meanLoss.append(clf_loss.item())

        # unfrozen
        for (key, param) in model.named_parameters():
            if key.startswith('critic'):
                param.requires_grad = True

        # loss NAN detection
        if np.isnan(sum(meanLoss) / len(meanLoss)):
            print(f"client {self.user_id}, loss NAN")
            exit(0)

        trainSamplesNum, update = self.trainSamplesNum, self.get_params()
        return trainSamplesNum, update, sum(meanLoss) / len(meanLoss)

    def metaTest(self, round_th):
        model = self.model
        model.to(self.device)
        model.train()

        # frozen
        for (key, param) in model.named_parameters():
            if key.startswith('shared'):
                param.requires_grad = False

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
                gFeature, lFeature, gValue, lValue, output = model(data)
                clf_loss = criterion(output, labels)
                WD = (gValue - lValue).mean()
                gradient_penalty = self.calc_gradient_penalty(model, gFeature.data, lFeature.data)
                loss = clf_loss + self.config.mu * (- WD + self.config.omega * gradient_penalty)
                loss.backward()
                optimizer.step()
                meanLoss.append(clf_loss.item())

        # unfrozen
        for (key, param) in model.named_parameters():
            if key.startswith('shared'):
                param.requires_grad = True

        # loss NAN detection
        if np.isnan(sum(meanLoss) / len(meanLoss)):
            print(f"client {self.user_id}, loss NAN")
            exit(0)

        trainSamplesNum, update = self.trainSamplesNum, self.get_params()
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
                gFeature, lFeature, gValue, lValue, output = model(data)
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

    def set_shared_critic_params(self, params):
        tmp_params = self.get_params()
        for (key, value) in params.items():
            if key.startswith('shared') or key.startswith('critic'):
                tmp_params[key] = value
        self.set_params(tmp_params)

    def update(self, client):
        self.model.load_state_dict(client.model.state_dict())
        self.trainLoader = client.trainLoader
        self.testLoader = client.testLoader
