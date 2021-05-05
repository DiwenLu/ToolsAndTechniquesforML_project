import abc
import pandas as pd
import numpy as np
from numpy.random import default_rng



class Policy:
    def __init__(self, num_actions=2):
        self.num_actions = num_actions

    @abc.abstractmethod
    def get_action_distribution(self, X):
        """
        This method is intended to be overridden by each implementation of Policy.

        Args:
            X (pd.DataFrame): contexts

        Returns:
            2-dim numpy array with the same number of rows as X and self.num_actions columns.
                Each rows gives the policy's probability distribution over actions conditioned on the context in the corresponding row of X
        """
        raise NotImplementedError("Must override method")

    def get_action_propensities(self, X, actions):
        """
        Args:
            X (pd.DataFrame): contexts, rows correspond to entries of actions
            actions (np.array): actions taken, represented by integers, corresponding to rows of X

        Returns:
            1-dim numpy array of probabilities (same size as actions) for taking each action in its corresponding context
        """
        ## DONE
        action_distribution = self.get_action_distribution(X)
        return np.take_along_axis(action_distribution, actions.reshape(-1, 1), axis=1).flatten()

    def select_actions(self, X, rng=default_rng(1)):
        """
        Args:
            X (pd.DataFrame): contexts, rows correspond to entries of actions and propensities returned

        Returns:
            actions (np.array): 1-dim numpy array of length equal to the number of rows of X.  Each entry is an integer indicating the action selected for the corresponding context in X.
                The action is selected randomly according to the policy, conditional on the context specified in the appropriate row of X.
            propensities (np.array): 1-dim numpy array of length equal to the number of rows of X; gives the propensity for each action selected in actions

        """
        ## DONE
        action_distribution = self.get_action_distribution(X)
        actions = np.array([np.random.choice(26, 1, p=action_distribution[i]) for i in range(X.shape[0])]).flatten()
        propensities = self.get_action_propensities(X, actions)
        assert len(actions) == len(propensities) == X.shape[0]

        return actions, propensities

    def get_value_estimate(self, X, full_rewards):
        """
        Args:
            X (pd.DataFrame): contexts, rows correspond to entries of full_rewards
            full_rewards (np.array): 2-dim numpy array with the same number of rows as X and self.num_actions columns;
                each row gives the rewards that would be received for each action for the context in the corresponding row of X.
                This would only be known in a full-feedback bandit, or estimated in a direct method

        Returns:
            scalar value giving the expected average reward received for playing the policy for contexts X and the given full_rewards

        """
        ## DONE
        n = X.shape[0]
        actions, propensities = self.select_actions(X)
        action_distribution = self.get_action_distribution(X)

        return (full_rewards * action_distribution).sum() / n


class UniformActionPolicy(Policy):
    def __init__(self, num_actions=2):
        self.num_actions = num_actions

    def get_action_distribution(self, X):
        ## DONE
        return np.full((X.shape[0], self.num_actions), 1.0 / self.num_actions)


class SKLearnPolicy(Policy):
    """
    An SKLearnPolicy uses a scikit learn model to generate an action distribution.  If the SKLearnPolicy is built with is_deterministic=False,
    then the predict distribution for a context x should be whatever predict_proba for the model returns.  If is_deterministic=True, then all the probability mass
    should be concentrated on whatever predict of the model returns.
    """
    def __init__(self, model, num_actions=2, is_deterministic=False):
        self.is_deterministic = is_deterministic
        self.num_actions = num_actions
        self.model = model

    def get_action_distribution(self, X):
        ## DONE
        if (self.is_deterministic):
            predictions = self.model.predict(X)
            return np.eye(self.num_actions)[predictions.reshape(-1)] # one hot
        else:
            return self.model.predict_proba(X)


    def select_actions(self, X, rng=default_rng(1)):
        ## DONE
        if (self.is_deterministic):
            actions = self.model.predict(X)
            propensities = np.full(len(actions), 1.0)
            return actions, propensities
        else:
            actions, propensities = Policy.select_actions(self, X)
            return actions, propensities


class VlassisLoggingPolicy(Policy):
    """
    This policy derives from another deterministic policy following the recipe described in the Vlassis et al paper, on the top of the second column in section 5.3.
    For any context x, if the deterministic policy selects action a, then this policy selects action a with probability eps (a supplied parameter), and spreads the
    rest of the probability mass uniformly over the other actions.
    """
    def __init__(self, deterministic_target_policy, num_actions=2, eps=0.05):
        self.num_actions = num_actions
        self.target_policy = deterministic_target_policy
        self.eps = eps

    def get_action_distribution(self, X):
        rest = (1.0-self.eps)/(self.num_actions-1)
        actions, propensities = self.target_policy.select_actions(X)
        action_distribution = np.eye(self.num_actions)[actions.reshape(-1)]*self.eps
        action_distribution[action_distribution == 0.0] = rest
        return action_distribution