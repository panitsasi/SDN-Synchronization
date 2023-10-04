import gym
from gym import spaces
from gym.utils import seeding
from networkx.generators.random_graphs import gnp_random_graph
import numpy as np
from itertools import combinations
import random


random.seed(10)  #choose your luck number here and fix seed


from underlyingNetwork_creator_v3 import NetworkState_SP
from underlyingNetwork_creator_v3 import NetworkState_LB_C

import copy
import csv
import math


from csv import writer
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
        
        

class sdnSync_SP(gym.Env):
    environment_name = "sdnSyncSP"
    """


    """

    def __init__(self, config_alternator, numStates, numActions, budget, place_budget, numSteps):

        self.network = NetworkState_SP(numStates+place_budget, place_budget) #network class
        
        self.budget = budget #sync budget

        self.sync_action_list = list(place_ones(numActions,budget))  #list of syn actions
        self.place_action_list = list(place_one(place_budget)) #list of placement actions
        self.action_list = multi_agent_action_permute(self.sync_action_list, self.place_action_list) #combined
        print(self.action_list)
        
        self.action_dim = len(self.action_list)
        self.action_size = len(self.action_list)    
        
        self.num_sync = numStates
        self.num_place = place_budget
        self.observation_size = numStates + place_budget
        #take care to learn aout this, sync budget is "how many neighbors to talk to?", place budget is "how many controllers are in the host domain?"
        self.sync_budget = budget
        self.place_budget = place_budget
        
        self.numSteps = numSteps

        self.state_cap = 20    ##remember to update this on underlaying network as well!
        self.probability_consider = False
        self.trials = 1

        self.state = np.zeros(self.observation_size)
        self.stepCount = 0
        self.EpiCount = 0

        self.action_space = self.action_list
        #self.action_space = gym.spaces.MultiBinary(n=self.dim)
        self.allocations = np.zeros(self.observation_size)
        self.game_allocations = np.zeros(self.observation_size)
        
        self.robinNum = 0
        self.robinPlaceNum = 0
        self.robinAction = self.action_list[self.robinNum]
        self.randAction = random.choice(self.action_list)
        self.robinAction = np.zeros(self.observation_size)    ##############
        
        self.epi_dqn_detect = 0
        self.epi_rand_detect = 0
        self.epi_robin_detect = 0
        self.epi_total = 0

        
        self.alpha = 1
        self.beta = 0.5
        self.place_change_step = 10 #10
        
        self.reward_placement = 0
        self.reward_placement_rand = 0
        self.reward_placement_robin = 0
        
        self.rolling_avg_reward_sync = 0
        self.rolling_avg_reward_place = 0
        self.alt1 = 20
        self.alt2 = 5
        
        self.networkDownStatus = np.zeros(numStates)  #ignore for now
        self.numDowns = [] #ignore
        self.numUps = [] #ignore
        self.numNews = [] #ignore
         
        
        if config_alternator == True:
            self.ratio_filename = 'ratio_alternator.csv'
            with open(self.ratio_filename, 'w', newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Episode","Total","alternator_DQN","Random","RRobin"])
        else:
            self.ratio_filename = 'ratio_adaptive.csv'
            with open(self.ratio_filename, 'w', newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Episode","Total","adapt_DQN","Random","RRobin"])

        if config_alternator == True:
            self.place_filename = 'placeReward_alternator.csv'
            with open(self.place_filename, 'w', newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Episode","retrain_DQN","Random","RRobin"])
        else:
            self.place_filename = 'placeReward_adaptive.csv'
            with open(self.place_filename, 'w', newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Episode","adapt_DQN","Random","RRobin"])
        self.config_alternator = config_alternator


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):  #this is the main function where each step of the game takes place, i.e. synchroniztion step
        
            
        networkChange = False #igmore for now
            
        #print('action is: ', action)
        binAction = self.action_list[action]
        self.state = self.get_new_state(binAction)
        self.allocations = self.allocations + binAction
        self.game_allocations = self.game_allocations + binAction 

        done = self.doneChecker() #game episode done flag
        
        self.updateRandRobinActions()  #DRL updates DRL actions, random and robin updated here
        
        #the line below goes to  network environment class 
        rewards, randActionRewards, robinActionRewards, total_count = self.network.update_controllers(binAction, self.state, done, self.randAction, self.robinAction, self.networkDownStatus, self.state_cap)
        
        if self.config_alternator == False:  #this function explained in document
            reward, reward_placement = self.rewardFunction(rewards, done)
            randActionReward, reward_placement_rand = self.rewardFunction(randActionRewards, done)
            robinActionReward, reward_placement_robin = self.rewardFunction(robinActionRewards, done)
        else:
            reward, reward_placement = self.rewardFunction_alternating(rewards, done)
            randActionReward, reward_placement_rand = self.rewardFunction_alternating(randActionRewards, done)
            robinActionReward, reward_placement_robin = self.rewardFunction_alternating(robinActionRewards, done)
        
        
        self.reward_placement += reward_placement 
        self.reward_placement_rand += reward_placement_rand
        self.reward_placement_robin += reward_placement_robin
        
        self.epi_dqn_detect += np.sum(rewards[0])
        self.epi_rand_detect += np.sum(randActionRewards[0])
        self.epi_robin_detect += np.sum(robinActionRewards[0])
        self.epi_total += total_count
        
        if done:
            
            print("epiCount: ",self.EpiCount)
            row = [self.EpiCount, self.epi_total, self.epi_dqn_detect, self.epi_rand_detect, self.epi_robin_detect]
            append_list_as_row(self.ratio_filename, row)
            row_place = [self.EpiCount, self.reward_placement, self.reward_placement_rand, self.reward_placement_robin]
            append_list_as_row(self.place_filename, row_place)
            print("place_rewards_row:", row_place)
        
            self.epi_dqn_detect = 0
            self.epi_rand_detect = 0
            self.epi_robin_detect = 0
            self.epi_total = 0
            
            self.reward_placement = 0
            self.reward_placement_rand = 0
            self.reward_placement_robin = 0
            

        if done:
            print("final state is ", self.state)
            print("final DRL action representation is ", action)
            print("final action is ",binAction)
            print("throughout game allocations is: ", self.allocations)
            print("in this game allocations is: ", self.game_allocations)
            print("correct ratio raw: ", row)
            

        return self.state, reward, done, {}, randActionReward, robinActionReward, self.numDowns, self.numUps, self.numNews, networkChange                 #supposed to return new state, reward and done


    def updateRandRobinActions(self):  #randomization and round robin actions updated
    
    
        syn_randAction = self.randAction[:self.num_sync]
        randcheck = 0
        while(randcheck==0):
            randcheck = 1
            syn_randAction = random.choice(self.sync_action_list)
            for location in range(self.num_sync):
                if  self.randAction[location] > 0 and self.state[location] < 0:
                    randcheck = 0
        self.randAction[:self.num_sync] = syn_randAction
        
        
        if  self.stepCount % self.place_change_step == 0:     
            place_randAction = self.randAction[self.num_sync:]
            randcheck = 0
            while(randcheck==0):
                randcheck = 1
                place_randAction = random.choice(self.place_action_list)
            self.randAction[self.num_sync:] = place_randAction    
                
            
        '''
        robinAction = 1.5 * self.robinAction[:self.num_sync]
        #self.robinAction = 1.5 * self.robinAction
        spent = copy.deepcopy(self.budget) 
        trig = 0  
        #print(robinAction) 
        while (spent > 0):
            for a in range(len(robinAction)):
                if robinAction[a] > 0:
                    trig = trig + 1
                elif trig >= self.budget and robinAction[a] == 0 and spent > 0:
                    robinAction[a] = 1
                    #print(budget, robinAction)
                    spent = spent - 1
            trig = self.budget
        for a in range(len(robinAction)):
            if robinAction[a] > 1: robinAction[a] = 0  
        self.robinAction[:self.num_sync] = robinAction
           
        '''  
        robincheck = 0
        while(robincheck==0):
            robincheck = 1
            self.robinNum += 1
            if self.robinNum >= len(self.sync_action_list):
                self.robinNum = 0
            robinAction = self.sync_action_list[self.robinNum]
            for location in range(self.num_sync):
                if  self.robinAction[location] > 0 and self.state[location] < 0:
                    robincheck = 0
        self.robinAction[:self.num_sync] = robinAction
       
        
        if  self.stepCount % self.place_change_step == 0:     
            robincheck = 0
            while(robincheck==0):
                robincheck = 1
                self.robinPlaceNum += 1
                if self.robinPlaceNum >= len(self.place_action_list):
                    self.robinPlaceNum = 0
                robinPlaceAction = self.place_action_list[self.robinPlaceNum]
            self.robinAction[self.num_sync:] = robinPlaceAction


    def reset(self):
        self.state = np.zeros(self.observation_size)
        #self.allocations = np.zeros(self.observation_size)
        self.game_allocations = np.zeros(self.observation_size)
        
        return np.array(self.state)

    def rewardFunction(self, rewards, done):   #one of two reward methods
        reward_sync = 100 * np.square(np.sum (rewards[0])) #application reward
        reward_placement = rewards[1]    #placement reward, how close to others
        place_time = rewards[2] #oselete. 
        reward = reward_sync * (2 + reward_placement)
        #reward = reward_sync*(2*self.alpha *  (reward_placement**2)) # * (1 / (0.2 + math.exp(-self.beta*place_time)))) #replace 1 with reward_placement
        #reward = (100*reward_placement)**2
        #if done:
        #    print("Sync reward is: ", reward_sync)
        #    print("placement reward is: ", reward_placement)
        #    print("placement timer is on: ", place_time)
        #    print("total reward is: ", reward)
        #    print((2*self.alpha * 1 * (1 / (0.2 + math.exp(-self.beta*place_time)))))

        #print('reward is:', reward)
        return reward, reward_placement
    
    def rewardFunction_alternating(self, rewards, done): # one of two reward methods
        #self.stepCount
        reward_sync = 100 * np.square(np.sum (rewards[0])) #application reward
        reward_placement = rewards[1]    #placement reward, how close to others
        place_time = rewards[2] #oselete. 
        
        if self.stepCount < 10:
            self.rolling_avg_reward_sync = self.rolling_avg_reward_sync + 0.1*reward_sync
            self.rolling_avg_reward_place = self.rolling_avg_reward_place + 0.1*reward_placement    
        else:
            self.rolling_avg_reward_sync = 0.9*self.rolling_avg_reward_sync + 0.1*reward_sync
            self.rolling_avg_reward_place = 0.9*self.rolling_avg_reward_place + 0.1*reward_placement
            
        if self.stepCount % (self.alt1+self.alt2) < self.alt1:
            reward = reward_sync
        elif self.stepCount < 10:
            reward = (reward_sync / reward_placement) * reward_placement
        else:
            reward = (self.rolling_avg_reward_sync / self.rolling_avg_reward_place) * reward_placement
        #print('reward is:', reward)
        return reward, reward_placement


    def get_new_state(self, action): #new state after action
        
        syn_state = self.state[:self.num_sync]
        invert_01_action = np.ones(self.num_sync) - action[:self.num_sync]
        unconstrained_new_state = np.multiply (syn_state  , invert_01_action) + np.ones(self.num_sync)
        new_sync_state = self.capStateValues (unconstrained_new_state)
        for entry in range(len(unconstrained_new_state)):
                if self.networkDownStatus[entry] > 0:
                    unconstrained_new_state[entry] = self.stateDownVal
        
        place_state = self.state[self.num_sync:]
        unconstrained_new_place_state = [(a+1)*b for a,b in 
                           zip(place_state,action[self.num_sync:])]  
        new_place_state = self.capStateValues (unconstrained_new_place_state)
        
        new_state = np.concatenate((new_sync_state , new_place_state))       
                    
        return new_state


    def capStateValues(self, observation):  #limit state value max val
        for value in range(len(observation)):
            observation[value] = min(observation[value],self.state_cap)
        return observation

    def doneChecker(self):
        if self.numSteps < self.stepCount:   
            done = True
            self.EpiCount += 1
            self.stepCount = 0
        else:
            done = False
        self.stepCount += 1
        return done


def place_ones(size, count):
    for positions in combinations(range(size), count):
        p = [0] * size
        for i in positions:
            p[i] = 1
        yield p
        
def place_one(size):
    count = 0
    for positions in range(size):
        p = [0] * size
        p[count] = 1
        count += 1
        yield p
        
def multi_agent_action_permute(sync_actions, place_actions):  #for creating actions' list
    actions_list = []
    for sync_entry in sync_actions: 
        for place_entry in place_actions: 
            actions_list.append(sync_entry+place_entry)
    return(actions_list)
    






class sdnSync_LB(gym.Env):
    environment_name = "sdnSyncLB"

    """


    """

    def __init__(self, config_alternator, numStates, numActions, budget, place_budget, numSteps):

        self.network = NetworkState_LB_C(numStates+place_budget, place_budget)
        
        self.budget = budget
        
        self.sync_action_list = list(place_ones(numActions,budget)) 
        self.place_action_list = list(place_one(place_budget))
        self.action_list = multi_agent_action_permute(self.sync_action_list, self.place_action_list)
        print(self.action_list)
        
        self.action_dim = numActions  
        self.action_size = len(self.action_list)    #####ncr(numActions, budget)
        
        self.num_sync = numStates
        self.num_place = place_budget
        self.observation_size = numStates + place_budget
        self.sync_budget = budget
        self.place_budget = place_budget
        
        self.numSteps = numSteps

        self.state_cap = 20
        self.probability_consider = False
        self.trials = 1

        self.state = np.zeros(self.observation_size)
        self.stepCount = 0
        self.EpiCount = 0


        self.allocations = np.zeros(self.observation_size)
        

        self.robinPlaceNum = 0    
        self.robinNum = 0
        self.robinAction = self.action_list[self.robinNum]
        self.randAction = random.choice(self.action_list)
        
        self.networkDownStatus = np.zeros(numStates)
        self.Downs_TriggerEpisode = 20        #20
        self.downProbability =  0.000015        #0.000015     #0.03
        self.UpProbability = 0.00000000000005          #0.05
        self.stateDownVal = -10
        self.numDowns = []
        self.numUps = []
        self.epiDownCount = 0
        self.epiUpCount = 0
        
        self.Adds_TriggerEpisode = 20
        self.newControllerProbability = 0.00007
        self.epiNewCount = 0
        self.numNews = []
        
        self.down_episodes = [30,40]
        
        self.alpha = 1
        self.beta = 0.5 
        self.place_change_step = 10
        
        self.syn_reward = 0
        self.syn_reward_rand = 0
        self.syn_reward_robin = 0
        self.reward_placement = 0
        self.reward_placement_rand = 0
        self.reward_placement_robin = 0
        
        self.rolling_avg_reward_sync = 0
        self.rolling_avg_reward_place = 0
        self.alt1 = 20
        self.alt2 = 3
        
        if config_alternator == True:
            self.syncr_filename = 'syncReward_NN_alternator.csv'
            with open(self.syncr_filename, 'w', newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Episode","retrain_DQN","Random","RRobin"])
        else:
            self.syncr_filename = 'syncReward_adaptive.csv'
            with open(self.syncr_filename, 'w', newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Episode","adapt_DQN","Random","RRobin"])
                
        if config_alternator == True:
            self.place_filename = 'placeReward_NN_alternator.csv'
            with open(self.place_filename, 'w', newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Episode","retrain_DQN","Random","RRobin"])
        else:
            self.place_filename = 'placeReward_adaptive.csv'
            with open(self.place_filename, 'w', newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Episode","adapt_DQN","Random","RRobin"])
        self.config_alternator = config_alternator
        
        
    def networkDownStatus_update(self, states):
        for state in range(self.observation_size):
            if states[state] < 0 and random.uniform(0,1)<self.UpProbability:
                states[state] = self.state_cap
                self.epiUpCount += 1
            if states[state] >= 0 and random.uniform(0,1)<self.downProbability:
                states[state] = self.stateDownVal
                self.epiDownCount += 1
                #print("network_changed")
            if states[state] < 0:
                states[state] = self.stateDownVal
        self.networkDownStatus = 0.5 * (np.sign(-(states+0.1)) + 1 )
        return states

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def newNetwork_create(self):
        if random.uniform(0,1) < self.newControllerProbability:
            self.state = np.append(self.state,0)
            self.networkDownStatus = np.append(self.networkDownStatus,0)
            self.observation_size += 1
            self.action_dim += 1
            self.action_list = list(place_ones(self.action_dim,self.budget))
            self.action_size = len(self.action_list) 
            self.network.add_controller()
            
            self.allocations = np.append(self.allocations,0)
             
            self.epiNewCount += 1
            return True 
        else:
            return False   
        
    def temp_newNetwork_create(self):
            self.state = np.append(self.state,0)
            self.networkDownStatus = np.append(self.networkDownStatus,0)
            self.observation_size += 1
            self.action_dim += 1
            self.action_list = list(place_ones(self.action_dim,self.budget))
            self.action_size = len(self.action_list) 
            self.network.add_controller()
            
            self.allocations = np.append(self.allocations,0)
             
            self.epiNewCount += 1
            

    def step(self, action):
        
        networkChange = False
        #if self.EpiCount >= self.Adds_TriggerEpisode:  
        #    networkChange = self.newNetwork_create()
        
        #if self.EpiCount in self.down_episodes:
        #    if self.stepCount == 3:
        #        networkChange = True
        #        self.temp_newNetwork_create()
        
        #if self.EpiCount >= self.Downs_TriggerEpisode:   
        #    self.state = self.networkDownStatus_update(self.state)   #possibly trigger network down/up  
            
            
        #print('action is: ', action)
        binAction = self.action_list[action]
        self.state = self.get_new_state(binAction)
        
        self.allocations = self.allocations + binAction

        done = self.doneChecker()
        
        rewards, randActionRewards, robinActionRewards = self.network.update_controllers(binAction, self.state, done, self.randAction, self.robinAction, self.networkDownStatus)
        reward, syn_reward, reward_placement= self.rewardFunction(rewards)
        randActionReward, syn_reward_rand, reward_placement_rand = self.rewardFunction(randActionRewards)
        robinActionReward, syn_reward_robin, reward_placement_robin = self.rewardFunction(robinActionRewards)
        
        if self.config_alternator == False:
            reward, syn_reward, reward_placement = self.rewardFunction(rewards)
            randActionReward, syn_reward_rand, reward_placement_rand = self.rewardFunction(randActionRewards)
            robinActionReward, syn_reward_robin, reward_placement_robin = self.rewardFunction(robinActionRewards)
        else:
            reward, syn_reward, reward_placement = self.rewardFunction_alternating(rewards)
            randActionReward, syn_reward_rand, reward_placement_rand = self.rewardFunction_alternating(randActionRewards)
            robinActionReward, syn_reward_robin, reward_placement_robin = self.rewardFunction_alternating(robinActionRewards)
        
        self.syn_reward += syn_reward 
        self.syn_reward_rand += syn_reward_rand
        self.syn_reward_robin += syn_reward_robin
        self.reward_placement += reward_placement 
        self.reward_placement_rand += reward_placement_rand
        self.reward_placement_robin += reward_placement_robin
        
        
        self.updateRandRobinActions()

        if done:
            
            row = [self.EpiCount, self.syn_reward, self.syn_reward_rand, self.syn_reward_robin]
            append_list_as_row(self.syncr_filename, row)
            row_place = [self.EpiCount, self.reward_placement, self.reward_placement_rand, self.reward_placement_robin]
            append_list_as_row(self.place_filename, row_place)
            
            print("final state is ", self.state)
            print("final DRL action representation is ", action)
            print("final action is ",binAction)
            print("throughout game allocations is: ", self.allocations)
            print("number of network disruptions was: ", self.numDowns[self.EpiCount-1])
            print("new Additions in the episode: ", self.numNews[self.epiNewCount-1])
            
            print("rewards row:", row)
            print("place_rewards_row:", row_place)
            self.syn_reward = 0
            self.syn_reward_rand = 0
            self.syn_reward_robin = 0
            self.reward_placement = 0
            self.reward_placement_rand = 0
            self.reward_placement_robin = 0
        

        return self.state, reward, done, {}, randActionReward, robinActionReward, self.numDowns, self.numUps, self.numNews, networkChange                       #supposed to return new state, reward and done
    
    def updateRandRobinActions(self):
        
        syn_randAction = self.randAction[:self.num_sync]
        randcheck = 0
        while(randcheck==0):
            randcheck = 1
            syn_randAction = random.choice(self.sync_action_list)
            for location in range(self.num_sync):
                if  self.randAction[location] > 0 and self.state[location] < 0:
                    randcheck = 0
        self.randAction[:self.num_sync] = syn_randAction
        
        
        if  self.stepCount % self.place_change_step == 0:     
            place_randAction = self.randAction[self.num_sync:]
            randcheck = 0
            while(randcheck==0):
                randcheck = 1
                place_randAction = random.choice(self.place_action_list)
            self.randAction[self.num_sync:] = place_randAction    
                
            
 
        '''   
        robinAction = 1.5 * self.robinAction[:self.num_sync]
        #self.robinAction = 1.5 * self.robinAction
        spent = copy.deepcopy(self.budget) 
        trig = 0  
        #print(robinAction) 
        while (spent > 0):
            for a in range(len(robinAction)):
                if robinAction[a] > 0:
                    trig = trig + 1
                elif trig >= self.budget and robinAction[a] == 0 and spent > 0:
                    robinAction[a] = 1
                    #print(budget, robinAction)
                    spent = spent - 1
            trig = self.budget
        for a in range(len(robinAction)):
            if robinAction[a] > 1: robinAction[a] = 0  
        self.robinAction[:self.num_sync] = robinAction       
        ''' 

                 
        robincheck = 0
        while(robincheck==0):
            robincheck = 1
            self.robinNum += 1
            if self.robinNum >= len(self.sync_action_list):
                self.robinNum = 0
            robinAction = self.sync_action_list[self.robinNum]
            for location in range(self.num_sync):
                if  self.robinAction[location] > 0 and self.state[location] < 0:
                    robincheck = 0
        self.robinAction[:self.num_sync] = robinAction

        
        if  self.stepCount % self.place_change_step == 0:     
            robincheck = 0
            while(robincheck==0):
                robincheck = 1
                self.robinPlaceNum += 1
                if self.robinPlaceNum >= len(self.place_action_list):
                    self.robinPlaceNum = 0
                robinPlaceAction = self.place_action_list[self.robinPlaceNum]
            self.robinAction[self.num_sync:] = robinPlaceAction
            

    def reset(self):
        self.state = np.zeros(self.observation_size)
        #self.allocations = np.zeros(self.observation_size)
        return np.array(self.state)
    

    def rewardFunction(self, rewards):
        reward_sync = 100 * rewards[0]   #application reward
        
        reward_placement = rewards[1] #placement.....
        place_time = rewards[2] #obselete
        
        #reward = reward_sync*(2*self.alpha *  (reward_placement**2)) 
        reward = reward_sync * (1 + reward_placement)
        #print('reward is:', reward)
        return reward, reward_sync, reward_placement

    def rewardFunction_alternating(self, rewards):
        #self.stepCount
        reward_sync = 100 * rewards[0]   #application reward
        reward_placement = rewards[1] #placement.....
        place_time = rewards[2] #obselete
        
        if self.stepCount < 10:
            self.rolling_avg_reward_sync = self.rolling_avg_reward_sync + 0.1*reward_sync
            self.rolling_avg_reward_place = self.rolling_avg_reward_place + 0.1*reward_placement    
        else:
            self.rolling_avg_reward_sync = 0.9*self.rolling_avg_reward_sync + 0.1*reward_sync
            self.rolling_avg_reward_place = 0.9*self.rolling_avg_reward_place + 0.1*reward_placement
            
        if self.stepCount % (self.alt1+self.alt2) < self.alt1:
            reward = reward_sync
        elif self.stepCount < 10:
            reward = (reward_sync / reward_placement) * reward_placement
        else:
            reward = (self.rolling_avg_reward_sync / self.rolling_avg_reward_place) * reward_placement
        #print('reward is:', reward)
        return reward, reward_sync, reward_placement
    
    
    def get_new_state(self, action):
            
        syn_state = self.state[:self.num_sync]
        invert_01_action = np.ones(self.num_sync) - action[:self.num_sync]
        unconstrained_new_state = np.multiply (syn_state  , invert_01_action) + np.ones(self.num_sync)
        new_sync_state = self.capStateValues (unconstrained_new_state)
        for entry in range(len(unconstrained_new_state)):
                if self.networkDownStatus[entry] > 0:
                    unconstrained_new_state[entry] = self.stateDownVal
        
        place_state = self.state[self.num_sync:]
        unconstrained_new_place_state = [(a+1)*b for a,b in 
                           zip(place_state,action[self.num_sync:])]  
        new_place_state = self.capStateValues (unconstrained_new_place_state)
        
        new_state = np.concatenate((new_sync_state , new_place_state))       
                    
        return new_state

    def capStateValues(self, observation):
        for value in range(len(observation)):
            observation[value] = min(observation[value],self.state_cap)
        return observation

    def doneChecker(self):
        if self.numSteps < self.stepCount:   
            done = True
            self.EpiCount += 1
            self.stepCount = 0
            self.numDowns.append(self.epiDownCount)
            self.epiDownCount = 0
            self.numUps.append(self.epiUpCount)
            self.epiUpCount = 0
            self.numNews.append(self.epiNewCount)
            self.epiNewCount = 0
        else:
            done = False
        self.stepCount += 1
        return done
        
