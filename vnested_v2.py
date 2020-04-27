from functools import partial
from pyro import distributions as dist
import pyro
import torch
import numpy as np
import gym
from copy import deepcopy
from pyro.infer import Importance, EmpiricalMarginal
ACTIONS = [0, 1, 2, 3]


def argmax(iterable, func):
    """Get the argmax of an iterable"""
    return max(iterable, key=func)


def sample_action(state, i=0):
    """ Uniform sampler of actions.  Ideally learned from data"""
    probs = [1., 1., 1., 1.]
    action = pyro.sample(f'action{state}{i}',
                          dist.Categorical(torch.tensor(probs)))
    return action


def reward(state, i=0):
    """Reward function given a state"""
    # Goal is state 15, reward 1 point
    if state == 15:
        return pyro.sample(f'reward{state}{i}', dist.Delta(torch.tensor(1.)))
    # Holes are state 5, 7, 11, 12, penalize 15 points
    if state in [5, 7, 11, 12]:
        return pyro.sample(f'reward{state}{i}', dist.Delta(torch.tensor(-10.)))
    # Create a reward that grows as we get close to goal
    r = float(1 / (15 - state + 1))
    return pyro.sample(f'reward{state}{i}', dist.Delta(torch.tensor(r)))


def expected_reward(Q_function, action, env, i):
    def get_posterior_mean(posterior, n_samples=30):
        """
        Calculate posterior mean
        """
        # Sample
        marginal_dist = EmpiricalMarginal(posterior).sample((n_samples, 1)).float()
        # assumed to be all the same
        return torch.mean(marginal_dist)
    # The use of the param store is an optimization
    param_name = 'posterior_reward_state{}_{}'.format(env.s, i)
    if param_name in list(pyro.get_param_store().keys()):
        posterior_mean = pyro.get_param_store().get_param(param_name)
        return posterior_mean
    else:
        # this gets slower as we increase num_samples
        inference = Importance(Q_function, num_samples=30)
        posterior = inference.run(action, env, i)
        posterior_mean = get_posterior_mean(posterior, 30)
        pyro.param(param_name, posterior_mean)
        return posterior_mean


def model(env, i=0):
    """Model of the environment"""
    action = sample_action(env.s, i=i)
    observation, reward, done, info = env.step(int(action))
    return env, observation, reward, done, info


def imagine_next_step(env, action, i):
    """Agent imagines next time step"""
    sim_env = deepcopy(env)
    state = sim_env.s
    int_model = pyro.do(model, {f'action{state}{i}': action})
    sim_env, _, _, _, _ = int_model(sim_env, i)
    # sanity check
    assert sim_env.lastaction == action
    return sim_env


def Q(action, env, i):
    """Q function variant of Bellman equation."""
    utility = reward(env.s, i)
    if utility not in [1., -10.]:
        env_step = imagine_next_step(env, action, i)
    # check if the action got us closer to the goal. if yes only then recurse
        if reward(env_step.s) > utility:
            # Calculate expected rewards for each action but
            # exclude backtracking actions.
            expected_rewards = [
                expected_reward(Q, act, env_step, i + 1)
                for j, act in enumerate(ACTIONS)
                if ACTIONS[abs(j - 2)] != action
            ]
            # Choose reward from optimal action
            utility = utility + max(expected_rewards)

    # pyro.factor(f'utility{i}', utility)
    return utility


def policy(real_env, i=0):
    # Choose optimal action
    action = argmax(ACTIONS, partial(Q, env=real_env, i=i))
    print(action)
    return action


def main():
    fl_env = gym.make('FrozenLake-v0', is_slippery=False)
    fl_env.reset()
    fl_env.s = 1
    for t in range(25):
        pyro.clear_param_store()
        action = policy(fl_env)
        int_model = pyro.do(model,
                            {f'action{fl_env.s}{t}': torch.tensor(action)})
        fl_env, observation, reward, done, info = int_model(fl_env, t)
        fl_env.render()
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
    fl_env.close()


if __name__ == '__main__':
    main()
