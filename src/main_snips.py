import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
from models import *
from make_dataloaders import *


if __name__ == "__main__":
    train_loader = torch.load("../data/train_loader_1024.pth")
    test_loader = torch.load("../data/test_loader.pth")
    model = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    X_test = torch.Tensor(np.load("../data/X_test.npy")).float().to(device)
    y_test = torch.LongTensor(np.load("../data/y_test.npy")).to(device)
    full_rewards_test = torch.Tensor(np.load("../data/full_rewards_test.npy")).float().to(device)
    assert X_test.shape[0] == len(y_test) == full_rewards_test.shape[0]

    for epoch in range(50):
        model.train()
        correct = 0
        total = 0
        lst = []
        for batch_idx, (context, actions, rewards, props, targets) in enumerate(train_loader):
            context, rewards, props = context.float(), rewards.float(), props.float()
            context, actions, rewards, props, targets = context.to(device), actions.to(device), rewards.to(device), props.to(device), targets.to(device)
            optimizer.zero_grad()
            context, actions, rewards, props, targets = Variable(context), Variable(actions), Variable(rewards), Variable(props), Variable(targets)
            outputs = model(context)
            action_distribution = F.softmax(outputs, dim=1)
            props_w = action_distribution.gather(1, actions.unsqueeze(1)).flatten()
            numerator = torch.mean(torch.mul(-rewards, torch.div(props_w, props)))
            denominator = torch.mean(torch.div(props_w, props))
            snips = torch.div(numerator, denominator)
            snips.backward()
            optimizer.step()
            lst.append(snips.item())
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            #if batch_idx % 1000 == 0:
            #    print(snips)
        print(torch.var(torch.Tensor(lst)).item())
        print("snips estimate:", snips.item())

        # get value estimate
        model.eval()
        outputs = model(X_test)
        action_distribution = F.softmax(outputs, dim=1)
        assert action_distribution.shape == full_rewards_test.shape
        direct_estimate = torch.sum(action_distribution * full_rewards_test)/action_distribution.shape[0]
        print(direct_estimate.item())