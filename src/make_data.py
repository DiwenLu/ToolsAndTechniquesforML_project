from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from pandas.api.types import is_integer_dtype
from numpy.random import default_rng
from policies import *

def get_fully_observed_bandit():
    """
    This loads in a multiclass classification problem and reformulates it as a fully observed bandit problem.

    """
    df_l = pd.read_csv('../data/letter-recognition.data',
                       names=['a'] + [f'x{i}' for i in range(16)])
    X = df_l.drop(columns=['a'])

    # Convert labels to ints and one-hot
    y = df_l['a']
    # if y is not column of integers (that represent classes), then convert
    if not is_integer_dtype(y.dtype):
        y = y.astype('category').cat.codes

    ## Full rewards
    n = len(y)
    k = max(y) + 1
    full_rewards = np.zeros([n, k])
    full_rewards[np.arange(0, n), y] = 1
    contexts = X
    best_actions = y
    return contexts, full_rewards, best_actions


if __name__ == "__main__":
    contexts, full_rewards, best_actions = get_fully_observed_bandit()
    n, k = full_rewards.shape
    _, d = contexts.shape
    print(f"There are {k} actions, the context space is {d} dimensional, and there are {n} examples.")
    print(f"For example, the first item has context vector:\n{contexts.iloc[0:1]}.")
    print(f"The best action is {best_actions[0]}.  The reward for that action is 1 and all other actions get reward 0.")
    print(f"The reward information is store in full_rewards as the row\n{full_rewards[0]}.")

    rng = default_rng(7)
    train_frac = 0.5
    train_size = round(train_frac * n)
    train_idx = rng.choice(n, size=train_size, replace=False)
    test_idx = np.setdiff1d(np.arange(n), train_idx, assume_unique=True)

    X_train = contexts.iloc[train_idx].to_numpy()
    y_train = best_actions.iloc[train_idx].to_numpy()
    X_test = contexts.iloc[test_idx].to_numpy()
    y_test = best_actions.iloc[test_idx].to_numpy()
    full_rewards_test = full_rewards[test_idx]

    model = LogisticRegression(multi_class='multinomial', solver="newton-cg")
    model.fit(X_train, y_train)
    print("train accuracy:", model.score(X_train, y_train))
    print("test accuracy:", model.score(X_test, y_test))
    policy_stochastic = SKLearnPolicy(model=model, num_actions=k, is_deterministic=False)
    actions, props = policy_stochastic.select_actions(X_train)
    rewards = (actions == y_train).astype(float)
    risks = 1-rewards


    np.save("../data/X_train.npy", X_train)
    np.save("../data/y_train.npy", y_train)
    np.save("../data/rewards.npy", rewards)
    np.save("../data/actions.npy", actions)
    np.save("../data/props.npy", props)
    np.save("../data/X_test.npy", X_test)
    np.save("../data/y_test.npy", y_test)
    np.save("../data/full_rewards_test.npy", full_rewards_test)


    print("\n-----Data saved-----")