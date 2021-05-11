import abc
import numpy as np
from numpy.random import default_rng
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV, Ridge


class Policy(nn.Module):
    def __init__(self, num_actions=2):
        super(Policy, self).__init__()
        self.num_actions = num_actions


    @abc.abstractmethod
    def get_action_distribution(self, X):

        raise NotImplementedError("Must override method")


    def get_action_propensities(self, X, actions):

        Pi = self.get_action_distribution(X)
        actions_one_hot = F.one_hot(actions.long(), num_classes=self.num_actions).float()
        propensities = torch.matmul(Pi.unsqueeze(1), actions_one_hot.unsqueeze(2)).squeeze()

        return propensities

    
    def select_actions(self, X, rng=default_rng(1)):

        Pi = self.get_action_distribution(X)
        actions = [np.random.choice(range(self.num_actions), p=Pi_i.detach().numpy()) for Pi_i in Pi]
        actions = torch.tensor(actions)
        propensities = self.get_action_propensities(X, actions)

        return actions, propensities
    
    def get_value_estimate(self, X, full_rewards):

        Pi = self.get_action_distribution(X)
        value = torch.sum(Pi * full_rewards, axis=1).mean()
        
        return value
    
    def get_accuracy(self):
        return

##### Uniform Policy ####################################################### 
class UniformActionPolicy(Policy):

    def __init__(self, num_actions=2):
        self.num_actions = num_actions

    def get_action_distribution(self, X):

        p = 1 / self.num_actions
        Pi = torch.zeros([X.shape[0], self.num_actions]) + p

        return Pi
    
##### Logistic Regression #######################################################
class LogisticRegression(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):

        outputs = self.linear(x)
        outputs = self.softmax(outputs)

        return outputs

class LogisticPolicy(Policy):

    def __init__(self, num_actions, num_features):

        super(LogisticPolicy, self).__init__()
        self.num_actions = num_actions
        self.model = LogisticRegression(num_features, num_actions)
    
    def get_action_distribution(self, X):

        Pi = self.model(X)
        
        return Pi
    