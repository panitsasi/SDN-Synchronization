import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

import operator as op
from functools import reduce


class sdnSync1(gym.Env):
    environment_name = "sdnSync"

    """


    """

    def __init__(self, numStates, numActions, budget,numSteps):

        self.action_dim = numActions
        self.action_size = 2**numActions -1    #####ncr(numActions, budget)
        self.observation_size = numStates
        self.budget = budget
        self.numSteps = numSteps

        self.state_cap = 15
        self.probability_consider = False
        self.trials = 1

        self.state = np.zeros(self.observation_size)
        self.stepCount = 0
        self.mistakes = 0
        self.subpar = 0

        self.action_space = BinaryEncoding(self.action_dim, self.budget)
        print("action space dtype is", self.action_space.dtype)
        #self.action_space = gym.spaces.MultiBinary(n=self.dim)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #print('action is: ', action)
        binAction = bin_array(action, self.action_dim)
        reward = self.rewardFunction(binAction)
        self.state = self.get_new_state(binAction)
        done = self.doneChecker()

        if done:
            print("final state is ", self.state)
            print("final DRL action representation is ", action)
            print("final action is ",binAction)
            print("constraint_violations: ", self.mistakes)
            print("subPar_choice: ", self.subpar)
            self.mistakes = 0
            self.subpar = 0


        return self.state, reward, done, {}                        #supposed to return new state, reward and done

    def reset(self):
        self.state = np.zeros(self.observation_size)
        return np.array(self.state)

    def rewardFunction(self, action):
        binary_sum = 0 
        for encode in action:
            binary_sum += encode
        if binary_sum > self.budget:
            reward = -9999.0
            self.mistakes += 1
        else:
            Probs = abs(np.random.normal(self.state, scale = 0.5)) 
            if self.probability_consider:      
                rewards = np.sqrt ( np.multiply ( np.multiply(self.state, action) , Probs ) )
            else:
                rewards = np.sqrt (np.multiply (self.state, action))
            reward = np.sum (rewards)  
            if binary_sum < self.budget:
                self.subpar += 1
                reward = reward / 10
            #print('reward is:', reward)
        return reward


    def get_new_state(self, action):
        invert_01_action = np.ones(self.action_dim) - action
        unconstrained_new_state = np.multiply (self.state  , invert_01_action) + np.ones(self.observation_size)
        new_state = self.capStateValues (unconstrained_new_state)
        return new_state


    def capStateValues(self, observation):
        for value in range(self.observation_size):
            observation[value] = min(observation[value],self.state_cap)
        return observation

    def doneChecker(self):
        if self.numSteps < self.stepCount:   
            done = True
            self.stepCount = 0
        else:
            done = False
        self.stepCount += 1
        return done


class BinaryEncoding(gym.Space):
    """
    {0,...,1,..0,...,1,.,0}

    Example usage:
    self.observation_space = OneHotEncoding(size=4)
    """
    def __init__(self, size=None, numOnes=None):
        assert isinstance(size, int) and size > 0
        assert isinstance(numOnes, int) and size > 0
        self.size = size
        self.numOnes = numOnes
        gym.Space.__init__(self, (), np.int64)

    def sample(self):
        one_hot_vector = np.zeros(self.size)
        permuter = np.random.permutation (self.size) 
        for entry in range(self.size):
            if permuter[entry] < self.numOnes:  one_hot_vector[entry] = 1
        return one_hot_vector

    def contains(self, x):
        if isinstance(x, (list, tuple, np.ndarray)):
            number_of_zeros = list(x).contains(0)
            number_of_ones = list(x).contains(1)
            return (number_of_zeros == (self.size - 1)) and (number_of_ones == 1)
        else:
            return False

    def __repr__(self):
        return "BinaryEncoding(%d)" % self.size

    def __eq__(self, other):
        return self.size == other.size

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)