# The MNIST database of handwritten digits. http://yann.lecun.com/exdb/mnist/
#
# In this problem you need to implement model that will learn to recognize
# handwritten digits
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ..utils.gridsearch import GridSearch
from ..utils import solutionmanager as sm


class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lr = solution.lr
        self.momentum = solution.momentum
        self.activation = solution.activation
        self.activations = solution.activations
        self.hidden0 = solution.hidden0
        self.hidden1 = solution.hidden1
        self.hidden2 = solution.hidden2
        self.conv1 = nn.Conv2d(1, self.hidden0, 5, 1)
        self.conv2 = nn.Conv2d(self.hidden0, self.hidden1, 5, 1)
        self.fc1 = nn.Linear(4*4*self.hidden1, self.hidden2)
        self.fc2 = nn.Linear(self.hidden2, output_size)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout1(x)
        x = x.view(-1, 4*4*self.hidden1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def calc_loss(self, output, target):
        loss = F.nll_loss(output, target)
        return loss

    def calc_predict(self, output):
        predict = output.max(1, keepdim=True)[1]
        return predict


class Solution():
    def __init__(self):
        self.lr = 0.0025
        self.momentum = 0.1
        self.hidden0 = 30
        self.hidden1 = 50
        self.hidden2 = 500
        # self.momentum_grid = list(np.linspace(0.1, 1, 10))
        # self.lr_grid = list(np.linspace(0.8, .9, 50))
        # self.lr_grid = [5, 6, 7]
        # self.hidden0_grid = [25, 30, 35, 45]
        # self.hidden1_grid = [25, 30, 35, 45]
        # self.hidden2_grid = [175, 200, 225, 250]
        # self.hidden0_grid = [25, 30, 35, 45]
        # self.hidden1_grid = [40, 50, 60]
        # self.hidden2_grid = [300, 400, 500, 600]
        self.activations = {
            'relu6': nn.ReLU6(),
            'leakyrelu': nn.LeakyReLU(negative_slope=0.01),
            'relu': nn.ReLU(),
        }
        self.activation = 'relu'
        # self.activation_grid = list(self.activations.keys())
        self.grid_search = GridSearch(self)
        self.grid_search.set_enabled(False)

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        step = 0
        # Put model in train mode
        model.train()
        optimizer = optim.SGD(model.parameters(), model.lr, model.momentum)
        # optimizer = optim.Adam(model.parameters(), model.lr)
        batches = 8
        batch_size = train_data.shape[0] // batches
        while True:
            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if not self.grid_search.enabled and time_left < 0.1:
                break
            ind = step % batches
            start_ind = batch_size * ind
            end_ind = batch_size * (ind + 1)
            data = train_data[start_ind:end_ind]
            target = train_target[start_ind:end_ind]
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            output = model(data)
            # get the index of the max probability
            predict = model.calc_predict(output)
            # Number of correct predictions
            correct = predict.eq(target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = predict.view(-1).size(0)
            # calculate loss
            loss = model.calc_loss(output, target)
            # if correct == total or time_left < 0.1:
            #     stats = {
            #         'step': step,
            #         'corr': correct,
            #         'ttl': total,
            #         'loss': round(loss.item(), 5),
            #         'lr': self.lr,
            #         'h0': self.hidden0,
            #         'h1': self.hidden1,
            #         'h2': self.hidden2
            #     }
            #     if self.grid_search.enabled :#and correct==total:
            #         self.print_stats(stats)
            #     break

            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # print progress of the learning
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            step += 1
        return step

    def print_stats(self, stats):
        print("Step = {} Pred = {}/{} Loss = {}, LR={}, h0={} h1={} h2={}".format(stats['step'], stats['corr'],stats['ttl'],stats['loss'],stats['lr'],stats['h0'],stats['h1'],stats['h2']))

###
###
# Don't change code after this line
###
###


class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 1000000
        self.test_limit = 0.95


class DataProvider:
    def __init__(self):
        self.number_of_cases = 10
        print("Start data loading...")
        train_dataset = torchvision.datasets.MNIST(
            './data/data_mnist', train=True, download=True,
            transform=torchvision.transforms.ToTensor()
        )
        trainLoader = torch.utils.data.DataLoader(
            train_dataset, batch_size=len(train_dataset))
        test_dataset = torchvision.datasets.MNIST(
            './data/data_mnist', train=False, download=True,
            transform=torchvision.transforms.ToTensor()
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=len(test_dataset))
        self.train_data = next(iter(trainLoader))
        self.test_data = next(iter(test_loader))
        print("Data loaded")

    def select_data(self, data, digits):
        data, target = data
        mask = target == -1
        for digit in digits:
            mask |= target == digit
        indices = torch.arange(0, mask.size(0))[mask].long()
        return (torch.index_select(data, dim=0, index=indices), target[mask])

    def create_case_data(self, case):
        if case == 1:
            digits = [0, 1]
        elif case == 2:
            digits = [8, 9]
        else:
            digits = [i for i in range(10)]

        description = "Digits: "
        for ind, i in enumerate(digits):
            if ind > 0:
                description += ","
            description += str(i)
        train_data = self.select_data(self.train_data, digits)
        test_data = self.select_data(self.test_data, digits)
        return sm.CaseData(case, Limits(), train_data, test_data).set_description(description).set_output_size(10)


class Config:
    def __init__(self):
        self.max_samples = 1000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()


# If you want to run specific case, put number here
sm.SolutionManager(Config()).run(case_number=-1)
