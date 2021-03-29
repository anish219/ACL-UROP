import numpy as np
import scipy.signal
import gym
import matplotlib.pyplot as plt

N_POS = 15
N_VEL = 12
x_rewards = []

class DiscreteSoftmaxPolicy(object):
    def __init__(self, num_states, num_actions, temperature=1):
        self.num_states = num_states 
        self.num_actions = num_actions
        self.logits = np.zeros((self.num_states, self.num_actions))
        self.temperature = temperature

    def act(self, state):
        probs = np.exp(self.logits[state, :] / self.temperature) / \
            np.sum(np.exp(self.logits[state, :] / self.temperature))       
        sample = np.random.choice(range(self.num_actions), p=probs)
        return np.array([sample])

    def compute_gradient(self, state, action, discounted_return):
        grad = np.zeros(self.logits.shape)
        grad[state, action] += 1
        grad[state, :] -= np.exp(self.logits[state, :] / self.temperature) / \
            np.sum(np.exp(self.logits[state, :] / self.temperature))
        grad *= discounted_return / self.temperature
        return grad

    def gradient_step(self, grad, step_size):
        self.logits += step_size * grad

def flatten_state(state):
    min_state = np.array([-1.2, -0.07])
    res = np.array([(0.6 + 1.2)/N_POS, (0.07*2)/N_VEL])
    idx = ((np.array(state) - min_state) / res).astype(int).flatten()
    flattened = int(idx[0] * N_VEL + idx[1])
    return flattened

def get_discounted_returns(rewards, gamma):
    r = rewards[::-1]
    a = [1, -gamma]
    b = [1]
    y = scipy.signal.lfilter(b, a, x=r)
    return y[::-1]

def reinforce(env, policy):
    num_episodes = 1000
    gamma = 0.999
    avg_rewards = 0.0
    
    for n in range(num_episodes):
        states = []
        actions = []
        discounted_rewards = []
        rewards = []
        gammas = [1]

        state = env.reset()
        state = flatten_state(state)
        done = False

        while not done:
            action = policy.act(state)
            states.append(state)
            actions.append(action)
            state, reward, done, _  = env.step(action  - 1)
            state = flatten_state(state)
            rewards.append(reward)
            gammas.append(gammas[-1]*gamma)

        rewards = np.array(rewards)
        # print(np.sum(rewards))
        avg_rewards += np.sum(rewards)
        x_rewards.append(np.sum(rewards))
        if rewards[-1] > 0:

            discounted_returns = get_discounted_returns(rewards, gamma)
            # print(discounted_returns)
            for state, action, discounted_return, gamma_t in zip(states, actions, discounted_returns, gammas):
                grad = policy.compute_gradient(state, action, discounted_return)
                policy.gradient_step(grad * gamma_t, 1e-4)

        if n % 100 == 0 and n > 0:
            print("Episode " + str(n) + ": " + str(avg_rewards / 100))
            avg_rewards = 0


if __name__ == "__main__":
    x_list, v_list, act_list = [], [], []
    
    rewards = np.array([1, 1, 2])

    env = gym.make('MountainCarContinuous-v0')
    num_actions = 3
    num_states = N_POS * N_VEL

    policy = DiscreteSoftmaxPolicy(num_states, num_actions, temperature=0.5)
    reinforce(env, policy)

    state = env.reset()
    x_list.append(state[0])
    v_list.append
    state = flatten_state(state)

    env.render()
    done = False
    while not done:
          action = policy.act(state)
          
          print(action)
          state, reward, done, _ = env.step(action - 1)
          state = flatten_state(state)
          env.render()    
    env.close()
    
    x_rewards_avg = [sum(x_rewards[i:i+100])/100 for i in range(len(x_rewards)-100)]
    
    plt.plot(x_rewards_avg)
    plt.xlabel('Episodes')
    plt.ylabel('Performance')
    # x = np.array([i for i in range(len(data))])
    # m, b = np.polyfit(x, data, 1)
    # plt.plot(x, m*x+b)
    plt.show()
