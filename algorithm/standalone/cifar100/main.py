import torch
import sys
import os
import torch.optim as optim
import numpy
import random
from torchvision.transforms import transforms
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../../')))

from utils.tools import setup_seed
from data.cifar100.cifar100 import get_cifar100_dataLoaders
from models.fedavg.cifar100.CIFAR100 import CIFAR100 as FedAVG_CIFAR100


trainTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

testTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

users, trainLoaders, testLoaders = get_cifar100_dataLoaders(batch_size=50, train_transform=trainTransform,
                                                            test_transform=testTransform)

epoch = 100
lr = 0.1
lr_decay = 0.99 ** 10
criterion = torch.nn.CrossEntropyLoss()

users_acc_lst = {}

for user in users:
    trainLoader, testLoader = trainLoaders[user], testLoaders[user]
    setup_seed(12)
    model = FedAVG_CIFAR100()
    model.cuda()
    model.train()
    optimal_acc = -1
    acc_lst = []
    for i in range(epoch):
        optimizer = optim.SGD(params=model.parameters(), lr=lr * (lr_decay ** i), weight_decay=1e-4)
        epoch_loss = []
        for step, (data, labels) in enumerate(trainLoader):
            data = data.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)

        model.eval()
        total_right = 0
        total_samples = len(testLoader.sampler)
        with torch.no_grad():
            batch_loss = []
            for step, (data, labels) in enumerate(testLoader):
                data = data.cuda()
                labels = labels.cuda()
                output = model(data)
                loss = criterion(output, labels)
                batch_loss.append(loss.item())
                output = torch.argmax(output, dim=-1)
                total_right += torch.sum(output == labels)
            acc = float(total_right) / total_samples
            avg_batch_loss = sum(batch_loss) / len(batch_loss)
            if acc > optimal_acc:
                optimal_acc = acc

        print(f"user: {user}, acc: {acc}")
        acc_lst.append(acc)

    optimal_acc = max(acc_lst)
    users_acc_lst.update({
        user: {
            'num_samples': len(testLoader.sampler),
            'acc_lst': acc_lst,
            'optimal_acc': optimal_acc}
    })

    # torch.save(model, f'./models/{user}.pkl')

all_clients_samples = 0
all_clients_right = 0

for user_, info in users_acc_lst.items():
    print(f"user: {user_}, optimal acc: {info['optimal_acc']}")
    all_clients_right += info['num_samples'] * info['optimal_acc']
    all_clients_samples += info['num_samples']

print("=MAX=MAX=MAX=MAX=MAX=MAX=")
print("all clients right", all_clients_right)
print("all clients samples", all_clients_samples)
avg_acc = float(all_clients_right) / all_clients_samples
print("Average Accuracy: ", avg_acc)
print("=MAX=MAX=MAX=MAX=MAX=MAX=")

round_acc_lst = []
for i in range(epoch):
    total_right = 0
    total_samples = 0
    for user_, info in users_acc_lst.items():
        total_right += info['num_samples'] * info['acc_lst'][i]
        total_samples += info['num_samples']

    acc = float(total_right) / total_samples
    round_acc_lst.append(acc)
    # print(f"Round {i}, acc: {acc}")
print(f"Max Round Average Acc: {max(round_acc_lst)}")
