import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MyDataset(Dataset):
    def __init__(self, train, context, best_actions, actions=None, rewards=None, propensities=None):
        self.train = train
        self.context = torch.LongTensor(context)
        self.targets = torch.LongTensor(best_actions)
        if self.train:
            self.actions = torch.LongTensor(actions)
            self.rewards = torch.Tensor(rewards)
            self.props = torch.Tensor(propensities)



    def __getitem__(self, index):
        context = self.context[index]
        targets = self.targets[index]
        if self.train:
            actions = self.actions[index]
            rewards = self.rewards[index]
            props = self.props[index]
            return context, actions, rewards, props, targets
        else:
            return context, targets

    def __len__(self):
        return len(self.targets)


if __name__ == "__main__":
    X_train = np.load("../data/X_train_uniform.npy")
    y_train = np.load("../data/y_train_uniform.npy")
    X_test = np.load("../data/X_test_uniform.npy")
    y_test = np.load("../data/y_test_uniform.npy")
    full_rewards_test = np.load("../data/full_rewards_test_uniform.npy")
    rewards = np.load("../data/rewards_uniform.npy")
    props = np.load("../data/props_uniform.npy")
    actions = np.load("../data/actions_uniform.npy")

    assert X_train.shape[0] == y_train.shape[0] == len(rewards) == len(props) == len(actions)
    assert X_test.shape[0] == full_rewards_test.shape[0] == y_test.shape[0]

    BATCH_SIZE = 1024
    train_dataset = MyDataset(train=True, context=X_train, best_actions=y_train, actions=actions, rewards=rewards, propensities=props)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataset = MyDataset(train=False, context=X_test, best_actions=y_test)
    test_loader = DataLoader(test_dataset, batch_size=64)

    torch.save(train_loader, "../data/train_loader_{}_uniform.pth".format(BATCH_SIZE))
    torch.save(test_loader, "../data/test_loader_uniform.pth")

    print("\n-----Data loaders saved-----")