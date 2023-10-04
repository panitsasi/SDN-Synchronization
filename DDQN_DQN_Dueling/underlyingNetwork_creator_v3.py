
from operator import sub
import networkx as nx
from itertools import combinations, groupby
import numpy as np
import random
import copy

random.seed(10)  #choose your luck number here and fix seed

import matplotlib.pyplot as plt
import time
import math

import csv
from csv import writer
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


class NetworkState_SP():
    
    def __init__(self, state_size, num_controllers):
        self.state_size = state_size
        
        self.minSize = 2   #smallest possible domain network
        self.maxSize = 15  #largest possible domain network 
        self.innitating_densityKnob = 0.2 #inn. graph density
        self._inter_domain_innitating_densityKnob = 0.2 #inter domain graph density
        self.changing_densityKnob = 0.02 #0.05 #0.001 #chance of link change
        self.controller_changing_density = 0.1  #ignore
        
        self.changingfull_densityKnob = 0.0001  #ignore
        
        self.serverMinPotential = 2 
        self.serverMaxPotential = 10

        self.networks = {}  #where network properties for app store, link states in this case
        self.local_network_paths = {} #store real values
        self.connection_to_subnet = {} 
        self.controller_memory = {}  #store predicted values 
        self.network_change_count = {}
        self.failure_count = {}
        self.impactful_change_count = {}
        
        self.control_node = {}
        self.size = {}
        self.density = {}
        self.distance_control = {}
        self.distance_dataPAvg = {}
        self.capacity = {}
    
        self.Robin_controller_memory = {}
        self.Rand_controller_memory = {}
        
        self.robinState = np.zeros(state_size)
        self.randState = np.zeros(state_size)
        
        self.num_controllers = num_controllers
        self.store_controllers = []
        self.sync_size = self.state_size - num_controllers
        self.place_steps = 0
        self.controller_assignments = [0]*num_controllers
        self.controller_change_count = 0
        self.bestInter_Intra = [[0]*2]*num_controllers
        self.bestAssign_count = 0
        self.bestAssign_prev = -1
        
        self.step_count = 0
        self.place_change_step = 10 #10
        
        self.total_count = 0

        #create intra-domian networks
        for subnet in range(self.sync_size):
            if subnet == 0:
                size = self.maxSize - 5 
            else:
                size = random.randint(self.minSize, self.maxSize)
            G = gnp_random_connected_graph(size, self.innitating_densityKnob)
            G = G.to_directed()
          

            self.networks[subnet] = G
            self.network_change_count[subnet] = 0
            self.failure_count[subnet] = 0
            self.impactful_change_count[subnet] = 0
            
        #create inter domain network
        self.fullG = nx.empty_graph()
        self.domainG = gnp_random_connected_graph(self.sync_size, self._inter_domain_innitating_densityKnob) 
        #self.domainG = generate_domain_graph(state_size, self._inter_domain_innitating_densityKnob)
        
        for subnet in range(self.sync_size):
            for node in self.networks[subnet]:
                self.fullG.add_node((subnet,node))
            for edge in self.networks[subnet].edges:
                self.fullG.add_edge((subnet,edge[0]),(subnet,edge[1]))

        for edge in self.domainG.edges:
                random_node1 = random.sample(list(self.networks[edge[0]].nodes),1)[0]
                random_node2 = random.sample(list(self.networks[edge[1]].nodes),1)[0]
                self.fullG.add_edge((edge[0],random_node1),(edge[1],random_node2))
                
        self.fullG = self.fullG.to_directed()
        self.control_fullG = copy.deepcopy(self.fullG)
        self.robin_fullG = copy.deepcopy(self.fullG)
        self.rand_fullG = copy.deepcopy(self.fullG)
        
        #nx.draw(self.fullG, with_labels=True)
        #plt.show()

        for subnet in range(self.sync_size): #continuation of creating intra-domain network
            G = self.networks[subnet]
            if subnet == 0:
                self.store_controllers = random.sample(list(G.nodes),self.num_controllers)
                self.core_node = (subnet,random.sample(list(G.nodes),1)[0]) 
                self.control_node[subnet] = self.core_node
            else:
                self.control_node[subnet] = (subnet,random.sample(list(G.nodes),1)[0])
            self.local_network_paths[subnet] = self.get_dijkstra_for_full_network(self.fullG, subnet, self.core_node)
            self.controller_memory[subnet] = copy.deepcopy(self.local_network_paths[subnet])
            self.Robin_controller_memory[subnet] = copy.deepcopy(self.local_network_paths[subnet])
            self.Rand_controller_memory[subnet] = copy.deepcopy(self.local_network_paths[subnet])
            
        self.rest_time = time.time()
                
                
    def update_full_graph(self, G, actions):  #update intra-domain edges from networks 
        
        for subnet in range(self.sync_size):
            if actions[subnet] == 1: # and subnet != 0:
                edges_to_remove = []
                for edge in G.edges:
                    node1 = edge[0][0]
                    node2 = edge[1][0]
                    if node1 == node2 and node1 == subnet:
                        edges_to_remove.append(edge)
                G.remove_edges_from(edges_to_remove)
               
                for edge in self.networks[subnet].edges:
                    G.add_edge((subnet,edge[0]),(subnet,edge[1]))            
                    G.add_edge((subnet,edge[1]),(subnet,edge[0])) 
            
        return G 
    
    
    def get_dijkstra_for_full_network(self, G, subnet, entryNode):
        dijkstraDict = {}
        node1 = entryNode
        for node2 in G:
            if node2[0] == subnet:
                dijkstraDict[(node1,node2)] = nx.dijkstra_path(G, source=node1, target=node2)               
        return dijkstraDict     
    
                                  
    def create_server_potential(self,subnet,G,entryNode):
        for node in G.nodes:
                self.potentials[subnet][node] = self.serverMaxPotential #random.randrange(self.serverMinPotential,self.serverMaxPotential)
        return self.potentials[subnet]

    def create_background_loads(self,subnet,G,entryNode):
        for node in G.nodes:
            randomBackTraff = random.uniform(0.1, random.uniform(0.4,0.8))*self.potentials[subnet][node]
            self.backgroundLoads[subnet][node] = randomBackTraff
        return self.backgroundLoads[subnet]
    
    #the main function where network evolve is called by DRL!!!
    def update_controllers(self,action,new_state,done,randaction, robinaction, networkDownStatus, state_cap): #update controller infos
        
        self.networkDownStatus = networkDownStatus
        
        self.update_randRobinStates(randaction, robinaction, state_cap)
        

        for subnet in range(self.sync_size):   #update all topologies in their local memory
            
            G = copy.deepcopy(self.networks[subnet])
            entryNode = random.sample(list(G.nodes),1)[0]
            
            #if subnet == 0 and  self.controller_changing_density > random.uniform(0,1):
            #    self.core_node = (subnet,random.sample(G.nodes,1)[0]) 
            #    self.control_node[subnet] = self.core_node
                
                
            Gtemp = copy.deepcopy(self.evolve_network(G, self.changing_densityKnob))
            if nx.is_strongly_connected(Gtemp):
                newDijkstra = self.get_dijkstra_for_subnetworks(G, entryNode)
                self.networks[subnet] = copy.deepcopy(Gtemp)
                self.network_change_count[subnet] += 1
            else:
                pass
            
            #if subnet == 0:
            #    nx.draw(self.networks[subnet], with_labels=True)
            #    plt.show()
            
        
            
        #udate graph according to acction parameters for: real, DRL, random, robin
        #time_start = time.time()    
        self.fullG = self.update_full_graph(self.fullG, actions = [1]*self.sync_size)
        self.control_fullG = self.update_full_graph(self.control_fullG, actions = action[:self.sync_size])
        self.rand_fullG = self.update_full_graph(self.rand_fullG, actions = randaction[:self.sync_size])
        self.robin_fullG = self.update_full_graph(self.robin_fullG, actions = robinaction[:self.sync_size])
        #time_end = time.time()      
        #print("for all graph reforming",time_end - time_start)
        
        #potentially change controller location
        if  self.step_count % self.place_change_step == 0:       
            self.place_steps += 1
            for entry in range(self.num_controllers):  #get core node / controller here
                if action[entry+self.sync_size] == 1:
                    selected_node = (0,self.store_controllers[entry])
                    if selected_node != self.core_node:
                        self.place_steps = 0
                        self.controller_change_count += 1
                    new_core_node = (0,self.store_controllers[entry])
                    self.controller_assignments[entry] += 1
                    break
            #time_start = time.time() 
            self.core_node = new_core_node
           
        local_network_paths_list = nx.single_source_shortest_path(self.fullG, self.core_node)
        controller_memory_paths_list = nx.single_source_shortest_path(self.control_fullG, self.core_node)
        Rand_controller_memory_path_list = nx.single_source_shortest_path(self.rand_fullG, self.core_node)
        Robin_controller_memory_path_list = nx.single_source_shortest_path(self.robin_fullG, self.core_node)
        
        #time_end = time.time() 
        #print("for all dijkstra calculations",time_end - time_start) 
        
        time_start = time.time()
        for subnet in range(self.sync_size):
            list_local = {}
            list_control = {}
            list_rand = {}
            list_round = {}
            
            for entry in local_network_paths_list:
                if entry[0] == subnet:
                    list_local[(self.core_node,entry)] = local_network_paths_list[entry]
                    list_control[(self.core_node,entry)] = controller_memory_paths_list[entry]
                    list_rand[(self.core_node,entry)] = Rand_controller_memory_path_list[entry]
                    list_round[(self.core_node,entry)] = Robin_controller_memory_path_list[entry]
                    
                self.local_network_paths[subnet] = list_local
                self.controller_memory[subnet] = list_control
                self.Rand_controller_memory[subnet] = list_rand
                self.Robin_controller_memory[subnet] = list_round 
        #time_end = time.time() 
        #print("for all assignments of dijkstra",time_end - time_start)
        
        controller_paths_list = {}
        for controller in self.store_controllers:
            controller_paths_list[controller] = nx.single_source_shortest_path(self.fullG, (0,controller))
        
        controller_paths_stored = {}
        for controller in self.store_controllers:
            controller_paths_stored[controller] = {}
            controller_paths_stored[controller]["intraDomain"] = []        
            controller_paths_stored[controller]["interController"] = []
            for entry in controller_paths_list[controller]:
                if entry[0] == 0:
                    path = controller_paths_list[controller][entry]   
                    controller_paths_stored[controller]["intraDomain"].append(path)
                else:
                    subnet = entry[0]
                    subnet_controller = self.control_node[subnet]
                    if entry == subnet_controller:   
                        path = controller_paths_list[controller][entry]    
                        controller_paths_stored[controller]["interController"].append(path)        
        
        #self.rest_time = time.time()

        for subnet in range(self.sync_size):   #update global picture if controller syncronizes
                                        
            for entry in self.controller_memory[subnet]:
                control_mem = self.controller_memory[subnet]
                local_mem = self.local_network_paths[subnet]
                if control_mem[entry] != local_mem[entry]:
                    self.impactful_change_count[subnet] += 1        
               
            self.size[subnet] = self.networks[subnet].number_of_nodes()


        reward_placements_ratio = self.get_placement_rewards_ratio(controller_paths_stored, action)
        reward_placements_ratio_rand = self.get_placement_rewards_ratio(controller_paths_stored, randaction)
        reward_placements_ratio_robin = self.get_placement_rewards_ratio(controller_paths_stored, robinaction)
        failureVector = [self.failureVector(new_state),reward_placements_ratio,self.place_steps]    
        randReward = [self.get_provided_failureVector(self.Rand_controller_memory, self.randState),
                      reward_placements_ratio_rand,self.place_steps] 
        robinReward = [self.get_provided_failureVector(self.Robin_controller_memory, self.robinState),
                    reward_placements_ratio_robin,self.place_steps] 
        
        self.state = new_state
        
        self.step_count += 1

        if done:
            print(" ")
            print("subNetwork change count (over all plays) is: ", self.network_change_count)
            print("failures count (over 1 play): ", self.failure_count)
            print("impactful change count (over 1 play) is: ", self.impactful_change_count)
            print("controller assignments is: ", self.controller_assignments)
            print("controller changes are: ", self.controller_change_count)
            print("placement performance [inter,entra] is: ", self.bestInter_Intra)   
            print("changes to best assignments: ", self.bestAssign_count)
            
            
            
            self.network_reset()
            print("networkDensities: ", self.density)
            print("AverageDistances: ", self.distance_dataPAvg)  
      
            #nx.draw(self.fullG, with_labels=True)
            #plt.show()

        return failureVector, randReward, robinReward, self.total_count
    

    def visualize_graph(self):
        pos = nx.spring_layout(self.fullG)
        nx.draw(self.fullG, pos, with_labels=True, node_size=700, node_color="skyblue")
        #nx.draw_networkx_edge_labels(self.fullG, pos, edge_labels={(i, j): self.fullG[i][j]['weight'] for i, j in self.fullG.edges()}, label_pos=0.5)
        plt.show()
    
    def get_placement_rewards_ratio(self,controller_paths_stored, action): #function that gives placement reward
        
        
        intraDomain_rewards = []
        interController_rewards = []
        choice = 0
        for controller in controller_paths_stored:
            intra_sum = 0
            for path in controller_paths_stored[controller]["intraDomain"]:
                intra_sum += len(path)
            inter_sum = 0                 
            for path in controller_paths_stored[controller]["interController"]:
                destination_domain = path[-1][0]
                if action[destination_domain] == 1:
                    inter_sum += len(path)
            intraDomain_rewards.append(1/intra_sum)
            interController_rewards.append(1/inter_sum)
            choice += 1
        
        #intraDomain_rewards = [1/x for x in intraDomain_rewards]   
        #interController_rewards = [1/x for x in interController_rewards] 
        
        normalized_intraDomain_rewards = [x/max(intraDomain_rewards) 
                                        for x in intraDomain_rewards]
        normalized_interController_rewards = [x/max(interController_rewards) 
                                for x in interController_rewards]
        
        rewards_nonscale = [sum(x) for x in zip(intraDomain_rewards,interController_rewards)]
        rewards = rewards_nonscale #[x - min(rewards_nonscale) for x in rewards_nonscale]
        
        choice = 0
        for controller in controller_paths_stored: 
               
            if controller == self.core_node[1]:
                place_reward = rewards[choice]
                
            val1 = self.bestInter_Intra[choice][0]
            val2 = self.bestInter_Intra[choice][1]   
            if normalized_interController_rewards[choice] == 1:
                val1 = self.bestInter_Intra[choice][0] + 1 
                #print(self.bestInter_Intra[choice][0])
                #print(self.bestInter_Intra[choice])
            if normalized_intraDomain_rewards[choice] == 1:
                val2 = self.bestInter_Intra[choice][1] + 1
            self.bestInter_Intra[choice] = [val1,val2]
                
            choice += 1
                
        best_loc = rewards.index(max(rewards))
        if best_loc != self.bestAssign_prev:
            self.bestAssign_count += 1
        self.bestAssign_prev = best_loc
        
        return  place_reward  
            
              
    def failureVector(self, states): #failure vector for sync. 
        failures = []
        self.total_count = 0
        for subnet in range(self.sync_size):
            entryNode = self.core_node
            local_paths = self.local_network_paths[subnet]
            global_paths = self.controller_memory[subnet]
            for nodeNum in self.networks[subnet]:
                node = (subnet,nodeNum)
                if local_paths[(entryNode,node)] == global_paths[(entryNode,node)] and self.networkDownStatus[subnet] == 0:
                    failures.append(1)
                else:
                    failures.append(0) 
                    self.failure_count[subnet] += 1
                self.total_count += 1
        return failures
    
    def get_provided_failureVector(self, memory, states): #failure vector for sync. 
        failures = []
        for subnet in range(self.sync_size):
            entryNode = self.core_node
            local_paths = self.local_network_paths[subnet]
            global_paths = memory[subnet]
            for nodeNum in self.networks[subnet]:
                node = (subnet,nodeNum)
                if local_paths[(entryNode,node)] == global_paths[(entryNode,node)] and self.networkDownStatus[subnet]==0:
                    failures.append(1)
                else:
                    failures.append(0) 
                    self.failure_count[subnet] += 1
        return failures
    
    def update_randRobinStates(self, randaction, robinaction, state_cap): #update for random and robin states
        
        for state in range(len(self.randState)):
            if randaction[state] == 1:
                self.randState[state] = 1
            else:
                self.randState[state] += 1 
                self.randState[state] = min(self.randState[state], state_cap)
            
                
        for state in range(len(self.robinState)):
            if robinaction[state] == 1:
                self.robinState[state] = 1
            else:
                self.robinState[state] += 1
                self.robinState[state] = min(self.robinState[state], state_cap)      

    def get_dijkstra_for_subnetworks(self, G, entryNode):
        dijkstraDict = {}
        node1 = entryNode
        for node2 in G:
            dijkstraDict[(node1,node2)] = nx.dijkstra_path(G, source=node1, target=node2)
        return dijkstraDict

    def evolve_network(self, G, prob):  
        for node1 in G.nodes:
            for node2 in G.nodes:
                if node1 != node2:
                    if prob > random.uniform(0,1):
                        if (node1,node2) in G.edges:
                            G.remove_edge(node1,node2)
                        else:
                            G.add_edge(node1,node2)
        return G
   
    def network_reset(self): #envorinment reset fn
        self.controller_assignments = [0]*self.num_controllers
        self.controller_change_count = 0
        for subnet in range(self.sync_size):
            self.failure_count[subnet] = 0
            self.impactful_change_count[subnet] = 0
        self.state = np.zeros(self.state_size)
        self.randState = np.zeros(self.state_size)
        self.robinState = np.zeros(self.state_size)
        self.bestInter_Intra = [[0]*2]*self.num_controllers
        self.bestAssign_count = 0
        self.step_count = 0






def gnp_random_connected_graph(n, p): #randomized graph generator
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is conneted
    """
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    return G


def tupled_paths(paths):
    tupled_paths = {}
    for path in paths:
        tupled=[]
        entries = paths[path]
        for hop in range(len(entries)-1):
            tupled.append((entries[hop],entries[hop+1]))
        tupled_paths[path] = tupled 
    return tupled_paths
























class XRVR():
    
    def __init__(self, state_size, num_controllers):
        self.state_size = state_size

        self.minSize = 2
        self.maxSize = 15
        self.innitating_densityKnob = 0.2
        
        self.changing_densityKnob = 0.02 #0.05 #0.001
        self.changingfull_densityKnob = 0.0001

        self.linkCapacities = 10 
        self.backGroundTrafficInnMean = 3
        self.backGroundTrafficInnSD = 1
        self.UserTrafficSD = 0.1
        self.backTrafficSD = 0.1 
        self.demandTraffic = abs(random.normalvariate(2,1))

        self.serverMaxPotential = 100
        self.serverMinPotential = 5
        self.serverChangeProb = 0.25 #1 #0.1
        self.computeRate = 1
        
        self._inter_domain_innitating_densityKnob = 0.2 
    
        self.control_node = {}
        self.size = {}
        self.density = {}
        self.distance_control = {}
        self.distance_dataPAvg = {}
        self.capacity = {}
        
        
        self.networks = {}
        self.network_paths = {}
        self.server_potential = {}
        self.connection_to_subnet = {}
        self.local_network_memory = {}
        self.controller_memory = {}
        self.backgroundTraffics = {}
        self.potentials = {}
        self.backgroundLoads = {}
        self.network_size = []
        self.bestcount = {}
        
        self.local_network_paths = {}
        
        
        self.Robin_controller_memory = {}
        self.Rand_controller_memory = {}
        
        
        self.robinState = np.zeros(state_size)
        self.randState = np.zeros(state_size)
        
        self.num_controllers = num_controllers
        self.store_controllers = []
        self.sync_size = self.state_size - num_controllers
        self.place_steps = 0
        self.controller_assignments = [0]*num_controllers
        self.controller_change_count = 0
        self.bestInter_Intra = [[0]*2]*num_controllers
        self.bestAssign_count = 0
        self.bestAssign_prev = -1
        
        self.step_count = 0
        self.place_change_step = 10
        
        self.comp_filename = 'compute_diff.csv'
        with open(self.comp_filename, 'w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Episode","adapt_DQN","Random","RRobin"])
        self.comp_reward = 0
        self.comp_reward_rand = 0
        self.comp_reward_robin = 0
        
        self.episode = 1
        

        #create intra-domian networks
        for subnet in range(self.sync_size):
            if subnet == 0:
                size = self.maxSize - 5 
            else:
                size = random.randint(self.minSize, self.maxSize)
            G = gnp_random_connected_graph(size, self.innitating_densityKnob)
            G = G.to_directed()

            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = random.randint(1, 10)





            self.networks[subnet] = G
            #entryNode = random.sample(G.nodes,1)[0]
            #self.connection_to_subnet[subnet] = entryNode
            #self.paths = self.get_dijkstra_for_subnetworks(G, entryNode)
            #self.network_paths[subnet] = tupled_paths(self.paths)
            self.network_size.append(size)

            #self.backgroundTraffics[subnet] = {}
            #self.potentials[subnet] = {}
            #self.backgroundLoads[subnet] = {}

            #self.server_potential[subnet] = self.create_server_potential(subnet,G,entryNode)
            #traffics = self.create_background_traffics(subnet,G,entryNode,self.network_paths[subnet])
            #abilities = self.create_background_loads(subnet,G,entryNode)
            #self.local_network_memory[subnet] = [traffics, abilities]
            #self.controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
            #self.bestcount[subnet] = 0
            
            #self.Robin_controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
            #self.Rand_controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
            
        #create inter domain network
        self.fullG = nx.empty_graph()
        self.domainG = gnp_random_connected_graph(self.sync_size, self._inter_domain_innitating_densityKnob) 
        
        for subnet in range(self.sync_size):
            for node in self.networks[subnet]:
                self.fullG.add_node((subnet,node))
            for edge in self.networks[subnet].edges:
                self.fullG.add_edge((subnet,edge[0]),(subnet,edge[1]))

        for edge in self.domainG.edges:
                random_node1 = random.sample(list(self.networks[edge[0]].nodes),1)[0]
                random_node2 = random.sample(list(self.networks[edge[1]].nodes),1)[0]
                self.fullG.add_edge((edge[0],random_node1),(edge[1],random_node2))
                self.fullG[(edge[0],random_node1)][(edge[1],random_node2)]['weight'] = random.randint(1, 10)
                

        self.fullG = self.fullG.to_directed()
        self.control_fullG = copy.deepcopy(self.fullG)
        self.robin_fullG = copy.deepcopy(self.fullG)
        self.rand_fullG = copy.deepcopy(self.fullG)
        
        #nx.draw(self.fullG, with_labels=True)
        #plt.show()
        
        for subnet in range(self.sync_size):
            G = self.networks[subnet]
            if subnet == 0:
                self.store_controllers = random.sample(list(G.nodes),3)
                self.core_node = (subnet,random.sample(list(G.nodes),1)[0]) 
                self.control_node[subnet] = self.core_node
            else:
                self.control_node[subnet] = (subnet,random.sample(list(G.nodes),1)[0])
                
            self.network_paths[subnet] = self.get_dijkstra_for_full_network(self.fullG, subnet, self.core_node)
            #self.controller_memory[subnet] = copy.deepcopy(self.local_network_paths[subnet])
            #self.Robin_controller_memory[subnet] = copy.deepcopy(self.local_network_paths[subnet])
            #self.Rand_controller_memory[subnet] = copy.deepcopy(self.local_network_paths[subnet])
            
            self.backgroundTraffics[subnet] = {}
            self.potentials[subnet] = {}
            self.backgroundLoads[subnet] = {}

            self.server_potential[subnet] = self.create_server_potential(subnet,G)
            
            traffics = self.create_background_traffics(subnet,self.network_paths[subnet])
            
            abilities = self.create_background_loads(subnet,G)
            self.local_network_memory[subnet] = [traffics, abilities]
            
            self.controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
            self.bestcount[subnet] = 0
            self.Robin_controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
            self.Rand_controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])            
            
            
            self.size[subnet] = self.networks[subnet].number_of_nodes()
            self.density[subnet] = self.get_network_density(self.networks[subnet])
            self.distance_control[subnet] = self.get_avgCP_dist(self.control_node[subnet], self.network_paths[subnet])
            self.distance_dataPAvg[subnet] = self.get_avgDataP_dist(self.network_paths[subnet])
            self.capacity[subnet] = random.randrange(self.serverMinPotential,self.serverMaxPotential) 
            
                   
    def get_dijkstra_for_full_network(self, G, subnet, entryNode):
        dijkstraDict = {}
        node1 = entryNode
        for node2 in G:
            if node2[0] == subnet:
                dijkstraDict[(node1,node2)] = nx.dijkstra_path(G, source=node1, target=node2)               
        return dijkstraDict     
            

    def create_background_traffics(self,subnet,network_paths):

        for node in network_paths:
            backgroundTraffic = {}  
            for hop in network_paths[node]:
                backgroundTraffic[hop] = random.normalvariate(self.backGroundTrafficInnMean, self.backGroundTrafficInnSD)
            index = node[1][1]
            self.backgroundTraffics[subnet][index] = backgroundTraffic    
        return self.backgroundTraffics[subnet]
        

    def generate_background_traffics(self,subnet,network_paths):
        for node in network_paths:          
            for hop in network_paths[node]:
                self.backgroundTraffics[subnet][node[1][1]][hop] = self.backgroundTraffics[subnet][node[1][1]][hop] + self.randomized_traffic_change("background") 
        return self.backgroundTraffics[subnet]          

    def create_server_potential(self,subnet,G):
        for node in G.nodes:
            self.potentials[subnet][node] = self.serverMaxPotential #random.randrange(self.serverMinPotential,self.serverMaxPotential)
        return self.potentials[subnet]

    def create_background_loads(self,subnet,G):
        for node in G.nodes:
            randomBackTraff = random.uniform(0.1, random.uniform(0.4,0.8))*self.potentials[subnet][node]
            self.backgroundLoads[subnet][node] = randomBackTraff
        return self.backgroundLoads[subnet]

    def generate_background_loads(self,subnet,G):
        for node in G.nodes:
            if self.serverChangeProb > random.uniform(0,1):
                randomBackTraff = random.uniform(0.1, random.uniform(0.4,0.8))*self.potentials[subnet][node]
                self.backgroundLoads[subnet][node] = randomBackTraff
        return self.backgroundLoads[subnet]   
    

    def randomized_traffic_change(self,flag):
        if flag == "background":
            return random.normalvariate(0,self.backTrafficSD)
        else:
            return random.normalvariate(0,self.UserTrafficSD)     
    

    def get_network_density(self,G):
        num_edges = len(G.edges())
        return  float(num_edges) / float(G.number_of_nodes())
    
    def get_avgDataP_dist(self, localNets):
        sum = 0.0
        size = 0.0
        for entry in localNets:
            sum += len(localNets[entry])
            size += 1
        return sum/size
                    
    def get_avgCP_dist(self, entryP, localNets):   
        for entry in localNets:
            if entry[-1] == entryP:
                return float(len(entry))
            

    def update_full_graph(self, G, actions):  #update intra-domain edges from networks 
        
        for subnet in range(self.sync_size):
            if actions[subnet] == 1: # and subnet != 0:
                edges_to_remove = []
                for edge in G.edges:
                    node1 = edge[0][0]
                    node2 = edge[1][0]
                    if node1 == node2 and node1 == subnet:
                        edges_to_remove.append(edge)
                G.remove_edges_from(edges_to_remove)
               
                for edge in self.networks[subnet].edges:
                    G.add_edge((subnet,edge[0]),(subnet,edge[1]))            
                    G.add_edge((subnet,edge[1]),(subnet,edge[0])) 
            
        return G 
    
    
    def get_dijkstra_for_full_network(self, G, subnet, entryNode):
        dijkstraDict = {}
        node1 = entryNode
        for node2 in G:
            if node2[0] == subnet:
                dijkstraDict[(node1,node2)] = nx.dijkstra_path(G, source=node1, target=node2)               
        return dijkstraDict     
            
            
    def evolve_full_network(self):    #evolve interdomain edges
        
        for node in self.domainG.node:
            for edge in self.domainG.edges:
                if self.changingfull_densityKnob > random.uniform(0,1):

                    if node == edge[0]:
                        random_node1 = random.sample(list(self.networks[edge[0]].nodes),1)[0]
                        random_node2 = random.sample(list(self.networks[edge[1]].nodes),1)[0]
                        self.fullG.add_edge((edge[0],random_node1),(edge[1],random_node2))
                        self.fullG.add_edge((edge[1],random_node2),(edge[0],random_node1))

                        count = 0
                        for edgeFG in self.fullG.edges:    
                            if edgeFG[0][0] == edge[0] and edgeFG[1][0] == edge[1] and count == 0:
                                self.fullG.remove_edge((edge[0],edgeFG[0][1]),(edge[1],edgeFG[1][1]))
                                self.fullG.remove_edge((edge[1],edgeFG[1][1]),(edge[0],edgeFG[0][1]))
                                count += 1                
    '''        
    def add_controller(self):
        
        self.state_size += 1
        subnet = self.state_size - 1
        
        size = random.randint(self.minSize, self.maxSize)
        #size = random.randint(self.maxSize-4, self.maxSize)
        G = gnp_random_connected_graph(size, self.innitating_densityKnob)
        G = G.to_directed()
        self.networks[subnet] = G
        entryNode = random.sample(G.nodes,1)[0]
        self.connection_to_subnet[subnet] = entryNode
        self.paths = self.get_dijkstra_for_subnetworks(G, entryNode)
        self.network_paths[subnet] = tupled_paths(self.paths)
        self.network_size.append(size)

        self.backgroundTraffics[subnet] = {}
        self.potentials[subnet] = {}
        self.backgroundLoads[subnet] = {}

        self.server_potential[subnet] = self.create_server_potential(subnet,G,entryNode)
        traffics = self.create_background_traffics(subnet,G,entryNode,self.network_paths[subnet])
        abilities = self.create_background_loads(subnet,G,entryNode)
        self.local_network_memory[subnet] = [traffics, abilities]
        self.controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
        self.bestcount[subnet] = 0
        
        self.Robin_controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
        self.Rand_controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
    '''
     
     
    def update_controllers(self,action,new_state,done, randaction, robinaction, networkDownStatus):
        
        self.networkDownStatus = networkDownStatus

        for subnet in range(self.sync_size):
            G = self.networks[subnet]
            newTraffics = self.generate_background_traffics(subnet,self.network_paths[subnet])
            newLoads = self.generate_background_loads(subnet,G)
            self.local_network_memory[subnet] = [newTraffics,newLoads]
        
        for subnet in range(self.sync_size):    
            G = copy.deepcopy(self.networks[subnet])
            entryNode = random.sample(list(G.nodes),1)[0]
            Gtemp = copy.deepcopy(self.evolve_network(G, self.changing_densityKnob))
            if nx.is_strongly_connected(Gtemp):
                #newDijkstra = self.get_dijkstra_for_subnetworks(G, entryNode)
                self.networks[subnet] = copy.deepcopy(Gtemp)
                #self.network_change_count[subnet] += 1
            else:
                pass
            
        self.fullG = self.update_full_graph(self.fullG, actions = [1]*self.sync_size)
        self.control_fullG = self.update_full_graph(self.control_fullG, actions = action[:self.sync_size])
        self.rand_fullG = self.update_full_graph(self.rand_fullG, actions = randaction[:self.sync_size])
        self.robin_fullG = self.update_full_graph(self.robin_fullG, actions = robinaction[:self.sync_size])

        try:
            for subnet in range(self.sync_size):   #update global picture if controller syncronizes
                if action[subnet] == 1:
                    self.controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
                if randaction[subnet] == 1:
                    self.Rand_controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
                if robinaction[subnet] == 1:
                    self.Robin_controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
        except:
            pass
        
        if  self.step_count % self.place_change_step == 0:       
            self.place_steps += 1
            for entry in range(self.num_controllers):  #get core node / controller here
                if action[entry+self.sync_size] == 1:
                    selected_node = (0,self.store_controllers[entry])
                    if selected_node != self.core_node:
                        self.place_steps = 0
                        self.controller_change_count += 1
                    new_core_node = (0,self.store_controllers[entry])
                    self.controller_assignments[entry] += 1
                    break
            #time_start = time.time() 
            self.core_node = new_core_node
            
        
        controller_paths_list = {}
        for controller in self.store_controllers:
            controller_paths_list[controller] = nx.single_source_shortest_path(self.fullG, (0,controller))
        
        controller_paths_stored = {}
        for controller in self.store_controllers:
            controller_paths_stored[controller] = {}
            controller_paths_stored[controller]["intraDomain"] = []        
            controller_paths_stored[controller]["interController"] = []
            for entry in controller_paths_list[controller]:
                if entry[0] == 0:
                    path = controller_paths_list[controller][entry]   
                    controller_paths_stored[controller]["intraDomain"].append(path)
                else:
                    subnet = entry[0]
                    subnet_controller = self.control_node[subnet]
                    if entry == subnet_controller:   
                        path = controller_paths_list[controller][entry]    
                        controller_paths_stored[controller]["interController"].append(path)       
                        
        reward_placements_ratio = self.get_placement_rewards_ratio(controller_paths_stored, action)
        reward_placements_ratio_rand = self.get_placement_rewards_ratio(controller_paths_stored, randaction)
        reward_placements_ratio_robin = self.get_placement_rewards_ratio(controller_paths_stored, robinaction)
        
        syn_reward, comp_reward = self.get_reward()
        syn_reward_rand, comp_reward_rand = self.get_provided_reward(self.Rand_controller_memory)
        syn_reward_robin, comp_reward_robin  = self.get_provided_reward(self.Robin_controller_memory)
        reward = [syn_reward ,
                         reward_placements_ratio,self.place_steps]    
        randReward = [syn_reward_rand,
                      reward_placements_ratio_rand,self.place_steps] 
        robinReward = [syn_reward_robin,
                    reward_placements_ratio_robin,self.place_steps]
        
        self.comp_reward += comp_reward
        self.comp_reward_rand += comp_reward_rand
        self.comp_reward_robin += comp_reward_robin
        
        self.state = new_state
        self.step_count += 1
        if done:
            print(" ")
            row = [self.episode, self.comp_reward, self.comp_reward_rand, self.comp_reward_robin]
            append_list_as_row(self.comp_filename, row)
            print("Comp rewards are: ", row)
            self.comp_reward = 0
            self.comp_reward_rand = 0
            self.comp_reward_robin = 0
            self.episode += 1
            
            self.network_reset()
            print('network_size was: ', self.network_size)

        return reward, randReward, robinReward
    
    
    
    def get_placement_rewards_ratio(self,controller_paths_stored, action):
        
        intraDomain_rewards = []
        interController_rewards = []
        choice = 0
        for controller in controller_paths_stored:
            intra_sum = 0
            for path in controller_paths_stored[controller]["intraDomain"]:
                intra_sum += len(path)
            inter_sum = 0                 
            for path in controller_paths_stored[controller]["interController"]:
                destination_domain = path[-1][0]
                if action[destination_domain] == 1:
                    inter_sum += len(path)
            intraDomain_rewards.append(1/intra_sum)
            interController_rewards.append(1/inter_sum)
            choice += 1
        
        #intraDomain_rewards = [1/x for x in intraDomain_rewards]   
        #interController_rewards = [1/x for x in interController_rewards] 
        
        normalized_intraDomain_rewards = [x/max(intraDomain_rewards) 
                                        for x in intraDomain_rewards]
        normalized_interController_rewards = [x/max(interController_rewards) 
                                for x in interController_rewards]
        
        rewards_nonscale = [sum(x) for x in zip(intraDomain_rewards,interController_rewards)]
        rewards = [x - min(rewards_nonscale) for x in rewards_nonscale]
        
        choice = 0
        for controller in controller_paths_stored: 
               
            if controller == self.core_node[1]:
                place_reward = rewards[choice]
                
            val1 = self.bestInter_Intra[choice][0]
            val2 = self.bestInter_Intra[choice][1]   
            if normalized_interController_rewards[choice] == 1:
                val1 = self.bestInter_Intra[choice][0] + 1 
                #print(self.bestInter_Intra[choice][0])
                #print(self.bestInter_Intra[choice])
            if normalized_intraDomain_rewards[choice] == 1:
                val2 = self.bestInter_Intra[choice][1] + 1
            self.bestInter_Intra[choice] = [val1,val2]
                
            choice += 1
                
        best_loc = rewards.index(max(rewards))
        if best_loc != self.bestAssign_prev:
            self.bestAssign_count += 1
        self.bestAssign_prev = best_loc
        
        return  place_reward 
    
    def get_provided_reward(self, provided_memory):
        local_nodes_and_delays = self.Calculate_nodes_and_delays(self.local_network_memory)
        global_nodes_and_delays = self.Calculate_nodes_and_delays(provided_memory)
        if len(local_nodes_and_delays) > 0 and len(global_nodes_and_delays) > 0:
            local_best_option = max(local_nodes_and_delays, key=local_nodes_and_delays.get)
            global_best_option = max(global_nodes_and_delays, key=global_nodes_and_delays.get)
            local_best_val = local_nodes_and_delays[local_best_option]
            global_val = local_nodes_and_delays[global_best_option]
            comp_reward = -(local_best_val - global_val)
            if local_best_option == global_best_option:
                reward = 100
            else:
                reward = 0
        else:
            reward = -99999999999999999
        return reward, comp_reward
        
    def get_reward(self):
        local_nodes_and_delays = self.Calculate_nodes_and_delays(self.local_network_memory)
        global_nodes_and_delays = self.Calculate_nodes_and_delays(self.controller_memory)
        if len(local_nodes_and_delays) > 0 and len(global_nodes_and_delays) > 0:
            local_best_option = max(local_nodes_and_delays, key=local_nodes_and_delays.get)
            global_best_option = max(global_nodes_and_delays, key=global_nodes_and_delays.get)
            for subnet in range(self.sync_size):
                if local_best_option[0] == subnet:
                    self.bestcount[subnet] += 1
            local_best_val = local_nodes_and_delays[local_best_option]
            global_val = local_nodes_and_delays[global_best_option]
            comp_reward = -(local_best_val - global_val)
            if local_best_option == global_best_option:
                reward = 100
            else:
                reward = 0
        else:
            reward = -99999999999999999
        return reward, comp_reward

    def Calculate_nodes_and_delays(self,memory):
        c = self.linkCapacities
        x = self.demandTraffic
        nodes_delays = {}
        for subnet in memory:
            if self.networkDownStatus[subnet] == 0:
                for node in memory[subnet][0]:
                    delayCalc = 0
                    #if node != self.connection_to_subnet[subnet]:
                    for link in memory[subnet][0][node]:
                        b = memory[subnet][0][node][link]
                        if (b+x) >= c: 
                            local_delay = 10000000000000.0  
                        else:
                            local_delay = ( ((b+x)/c) / (1 + ((b+x)/c)) )
                        local_delay = 0
                            #delayCalc += local_delay 
                    compDelay = self.computeRate * (self.potentials[subnet][node] - memory[subnet][1][node])
                    delayCalc += compDelay 
                    nodes_delays[(subnet,node)] = delayCalc
            if self.networkDownStatus[subnet] != 0:
                for node in memory[subnet][0]:
                    nodes_delays[(subnet,node)] = 0
        return nodes_delays

    def get_dijkstra_for_subnetworks(self, G, entryNode):
        dijkstraDict = {}
        node1 = entryNode
        for node2 in G:
            dijkstraDict[(node1,node2)] = nx.dijkstra_path(G, source=node1, target=node2)
        return dijkstraDict
    
    def evolve_network(self, G, prob):
        for node1 in G.nodes:
            for node2 in G.nodes:
                if node1 != node2:
                    if prob > random.uniform(0,1):
                        if (node1,node2) in G.edges:
                            G.remove_edge(node1,node2)
                        else:
                            G.add_edge(node1,node2)
        return G

    def network_reset(self):
        self.state = np.zeros(self.state_size)
        print('best server contain count:', self.bestcount)
        print("controller assignments is: ", self.controller_assignments)
        print("controller changes are: ", self.controller_change_count)
        print("placement performance [inter,entra] is: ", self.bestInter_Intra)   
        print("changes to best assignments: ", self.bestAssign_count)
        for subnet in range(self.sync_size):
            self.bestcount[subnet] = 0
            self.bestInter_Intra = [[0]*2]*self.num_controllers
        self.bestAssign_count = 0
        self.step_count = 0
        self.controller_assignments = [0]*self.num_controllers
        self.controller_change_count = 0
        self.step_count = 0

    def visualize_graph(self):
        pos = nx.spring_layout(self.fullG)
        nx.draw(self.fullG, pos, with_labels=True, node_size=700, node_color="skyblue")
        #nx.draw_networkx_edge_labels(self.fullG, pos, edge_labels={(i, j): self.fullG[i][j]['weight'] for i, j in self.fullG.edges()}, label_pos=0.5)
        plt.show()
    
            
        
        
        
        
        




class NetworkState_LB_C():
    
    def __init__(self, state_size, num_controllers):
        self.state_size = state_size

        self.minSize = 2
        self.maxSize = 15
        self.innitating_densityKnob = 0.2
        
        self.changing_densityKnob = 0.02 #0.05 #0.001
        self.changingfull_densityKnob = 0.0001

        self.linkCapacities = 10 
        self.backGroundTrafficInnMean = 3
        self.backGroundTrafficInnSD = 1
        self.UserTrafficSD = 0.1
        self.backTrafficSD = 0.1 
        self.demandTraffic = abs(random.normalvariate(2,1))

        self.serverMaxPotential = 100
        self.serverMinPotential = 5
        self.serverChangeProb = 0.25 #1 #0.1
        self.computeRate = 1
        
        self._inter_domain_innitating_densityKnob = 0.2 
    
        self.control_node = {}
        self.size = {}
        self.density = {}
        self.distance_control = {}
        self.distance_dataPAvg = {}
        self.capacity = {}
        
        
        self.networks = {}
        self.network_paths = {}
        self.server_potential = {}
        self.connection_to_subnet = {}
        self.local_network_memory = {}
        self.controller_memory = {}
        self.backgroundTraffics = {}
        self.potentials = {}
        self.backgroundLoads = {}
        self.network_size = []
        self.bestcount = {}
        
        self.local_network_paths = {}
        
        
        self.Robin_controller_memory = {}
        self.Rand_controller_memory = {}
        
        
        self.robinState = np.zeros(state_size)
        self.randState = np.zeros(state_size)
        
        self.num_controllers = num_controllers
        self.store_controllers = []
        self.sync_size = self.state_size - num_controllers
        self.place_steps = 0
        self.controller_assignments = [0]*num_controllers
        self.controller_change_count = 0
        self.bestInter_Intra = [[0]*2]*num_controllers
        self.bestAssign_count = 0
        self.bestAssign_prev = -1
        
        self.step_count = 0
        self.place_change_step = 10
        
        self.comp_filename = 'compute_diff.csv'
        with open(self.comp_filename, 'w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Episode","adapt_DQN","Random","RRobin"])
        self.comp_reward = 0
        self.comp_reward_rand = 0
        self.comp_reward_robin = 0
        
        self.episode = 1
        

        #create intra-domian networks
        for subnet in range(self.sync_size):
            if subnet == 0:
                size = self.maxSize - 5 
            else:
                size = random.randint(self.minSize, self.maxSize)
            G = gnp_random_connected_graph(size, self.innitating_densityKnob)
            G = G.to_directed()

            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = random.randint(1, 10)





            self.networks[subnet] = G
            #entryNode = random.sample(G.nodes,1)[0]
            #self.connection_to_subnet[subnet] = entryNode
            #self.paths = self.get_dijkstra_for_subnetworks(G, entryNode)
            #self.network_paths[subnet] = tupled_paths(self.paths)
            self.network_size.append(size)

            #self.backgroundTraffics[subnet] = {}
            #self.potentials[subnet] = {}
            #self.backgroundLoads[subnet] = {}

            #self.server_potential[subnet] = self.create_server_potential(subnet,G,entryNode)
            #traffics = self.create_background_traffics(subnet,G,entryNode,self.network_paths[subnet])
            #abilities = self.create_background_loads(subnet,G,entryNode)
            #self.local_network_memory[subnet] = [traffics, abilities]
            #self.controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
            #self.bestcount[subnet] = 0
            
            #self.Robin_controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
            #self.Rand_controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
            
        #create inter domain network
        self.fullG = nx.empty_graph()
        self.domainG = gnp_random_connected_graph(self.sync_size, self._inter_domain_innitating_densityKnob) 
        
        for subnet in range(self.sync_size):
            for node in self.networks[subnet]:
                self.fullG.add_node((subnet,node))
            for edge in self.networks[subnet].edges:
                self.fullG.add_edge((subnet,edge[0]),(subnet,edge[1]))

        for edge in self.domainG.edges:
                random_node1 = random.sample(list(self.networks[edge[0]].nodes),1)[0]
                random_node2 = random.sample(list(self.networks[edge[1]].nodes),1)[0]
                self.fullG.add_edge((edge[0],random_node1),(edge[1],random_node2))
                self.fullG[(edge[0],random_node1)][(edge[1],random_node2)]['weight'] = random.randint(1, 10)
                

        self.fullG = self.fullG.to_directed()
        self.control_fullG = copy.deepcopy(self.fullG)
        self.robin_fullG = copy.deepcopy(self.fullG)
        self.rand_fullG = copy.deepcopy(self.fullG)
        
        #nx.draw(self.fullG, with_labels=True)
        #plt.show()
        
        for subnet in range(self.sync_size):
            G = self.networks[subnet]
            if subnet == 0:
                self.store_controllers = random.sample(list(G.nodes),3)
                self.core_node = (subnet,random.sample(list(G.nodes),1)[0]) 
                self.control_node[subnet] = self.core_node
            else:
                self.control_node[subnet] = (subnet,random.sample(list(G.nodes),1)[0])
                
            self.network_paths[subnet] = self.get_dijkstra_for_full_network(self.fullG, subnet, self.core_node)
            #self.controller_memory[subnet] = copy.deepcopy(self.local_network_paths[subnet])
            #self.Robin_controller_memory[subnet] = copy.deepcopy(self.local_network_paths[subnet])
            #self.Rand_controller_memory[subnet] = copy.deepcopy(self.local_network_paths[subnet])
            
            self.backgroundTraffics[subnet] = {}
            self.potentials[subnet] = {}
            self.backgroundLoads[subnet] = {}

            self.server_potential[subnet] = self.create_server_potential(subnet,G)
            
            traffics = self.create_background_traffics(subnet,self.network_paths[subnet])
            
            abilities = self.create_background_loads(subnet,G)
            self.local_network_memory[subnet] = [traffics, abilities]
            
            self.controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
            self.bestcount[subnet] = 0
            self.Robin_controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
            self.Rand_controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])            
            
            
            self.size[subnet] = self.networks[subnet].number_of_nodes()
            self.density[subnet] = self.get_network_density(self.networks[subnet])
            self.distance_control[subnet] = self.get_avgCP_dist(self.control_node[subnet], self.network_paths[subnet])
            self.distance_dataPAvg[subnet] = self.get_avgDataP_dist(self.network_paths[subnet])
            self.capacity[subnet] = random.randrange(self.serverMinPotential,self.serverMaxPotential) 
            
                   
    def get_dijkstra_for_full_network(self, G, subnet, entryNode):
        dijkstraDict = {}
        node1 = entryNode
        for node2 in G:
            if node2[0] == subnet:
                dijkstraDict[(node1,node2)] = nx.dijkstra_path(G, source=node1, target=node2)               
        return dijkstraDict     
            

    def create_background_traffics(self,subnet,network_paths):

        for node in network_paths:
            backgroundTraffic = {}  
            for hop in network_paths[node]:
                backgroundTraffic[hop] = random.normalvariate(self.backGroundTrafficInnMean, self.backGroundTrafficInnSD)
            index = node[1][1]
            self.backgroundTraffics[subnet][index] = backgroundTraffic    
        return self.backgroundTraffics[subnet]
        

    def generate_background_traffics(self,subnet,network_paths):
        for node in network_paths:          
            for hop in network_paths[node]:
                self.backgroundTraffics[subnet][node[1][1]][hop] = self.backgroundTraffics[subnet][node[1][1]][hop] + self.randomized_traffic_change("background") 
        return self.backgroundTraffics[subnet]          

    def create_server_potential(self,subnet,G):
        for node in G.nodes:
            self.potentials[subnet][node] = self.serverMaxPotential #random.randrange(self.serverMinPotential,self.serverMaxPotential)
        return self.potentials[subnet]

    def create_background_loads(self,subnet,G):
        for node in G.nodes:
            randomBackTraff = random.uniform(0.1, random.uniform(0.4,0.8))*self.potentials[subnet][node]
            self.backgroundLoads[subnet][node] = randomBackTraff
        return self.backgroundLoads[subnet]

    def generate_background_loads(self,subnet,G):
        for node in G.nodes:
            if self.serverChangeProb > random.uniform(0,1):
                randomBackTraff = random.uniform(0.1, random.uniform(0.4,0.8))*self.potentials[subnet][node]
                self.backgroundLoads[subnet][node] = randomBackTraff
        return self.backgroundLoads[subnet]   
    

    def randomized_traffic_change(self,flag):
        if flag == "background":
            return random.normalvariate(0,self.backTrafficSD)
        else:
            return random.normalvariate(0,self.UserTrafficSD)     
    

    def get_network_density(self,G):
        num_edges = len(G.edges())
        return  float(num_edges) / float(G.number_of_nodes())
    
    def get_avgDataP_dist(self, localNets):
        sum = 0.0
        size = 0.0
        for entry in localNets:
            sum += len(localNets[entry])
            size += 1
        return sum/size
                    
    def get_avgCP_dist(self, entryP, localNets):   
        for entry in localNets:
            if entry[-1] == entryP:
                return float(len(entry))
            

    def update_full_graph(self, G, actions):  #update intra-domain edges from networks 
        
        for subnet in range(self.sync_size):
            if actions[subnet] == 1: # and subnet != 0:
                edges_to_remove = []
                for edge in G.edges:
                    node1 = edge[0][0]
                    node2 = edge[1][0]
                    if node1 == node2 and node1 == subnet:
                        edges_to_remove.append(edge)
                G.remove_edges_from(edges_to_remove)
               
                for edge in self.networks[subnet].edges:
                    G.add_edge((subnet,edge[0]),(subnet,edge[1]))            
                    G.add_edge((subnet,edge[1]),(subnet,edge[0])) 
            
        return G 
    
    
    def get_dijkstra_for_full_network(self, G, subnet, entryNode):
        dijkstraDict = {}
        node1 = entryNode
        for node2 in G:
            if node2[0] == subnet:
                dijkstraDict[(node1,node2)] = nx.dijkstra_path(G, source=node1, target=node2)               
        return dijkstraDict     
            
            
    def evolve_full_network(self):    #evolve interdomain edges
        
        for node in self.domainG.node:
            for edge in self.domainG.edges:
                if self.changingfull_densityKnob > random.uniform(0,1):

                    if node == edge[0]:
                        random_node1 = random.sample(list(self.networks[edge[0]].nodes),1)[0]
                        random_node2 = random.sample(list(self.networks[edge[1]].nodes),1)[0]
                        self.fullG.add_edge((edge[0],random_node1),(edge[1],random_node2))
                        self.fullG.add_edge((edge[1],random_node2),(edge[0],random_node1))

                        count = 0
                        for edgeFG in self.fullG.edges:    
                            if edgeFG[0][0] == edge[0] and edgeFG[1][0] == edge[1] and count == 0:
                                self.fullG.remove_edge((edge[0],edgeFG[0][1]),(edge[1],edgeFG[1][1]))
                                self.fullG.remove_edge((edge[1],edgeFG[1][1]),(edge[0],edgeFG[0][1]))
                                count += 1                
    '''        
    def add_controller(self):
        
        self.state_size += 1
        subnet = self.state_size - 1
        
        size = random.randint(self.minSize, self.maxSize)
        #size = random.randint(self.maxSize-4, self.maxSize)
        G = gnp_random_connected_graph(size, self.innitating_densityKnob)
        G = G.to_directed()
        self.networks[subnet] = G
        entryNode = random.sample(G.nodes,1)[0]
        self.connection_to_subnet[subnet] = entryNode
        self.paths = self.get_dijkstra_for_subnetworks(G, entryNode)
        self.network_paths[subnet] = tupled_paths(self.paths)
        self.network_size.append(size)

        self.backgroundTraffics[subnet] = {}
        self.potentials[subnet] = {}
        self.backgroundLoads[subnet] = {}

        self.server_potential[subnet] = self.create_server_potential(subnet,G,entryNode)
        traffics = self.create_background_traffics(subnet,G,entryNode,self.network_paths[subnet])
        abilities = self.create_background_loads(subnet,G,entryNode)
        self.local_network_memory[subnet] = [traffics, abilities]
        self.controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
        self.bestcount[subnet] = 0
        
        self.Robin_controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
        self.Rand_controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
    '''
     
     
    def update_controllers(self,action,new_state,done, randaction, robinaction, networkDownStatus):
        
        self.networkDownStatus = networkDownStatus

        for subnet in range(self.sync_size):
            G = self.networks[subnet]
            newTraffics = self.generate_background_traffics(subnet,self.network_paths[subnet])
            newLoads = self.generate_background_loads(subnet,G)
            self.local_network_memory[subnet] = [newTraffics,newLoads]
        
        for subnet in range(self.sync_size):    
            G = copy.deepcopy(self.networks[subnet])
            entryNode = random.sample(list(G.nodes),1)[0]
            Gtemp = copy.deepcopy(self.evolve_network(G, self.changing_densityKnob))
            if nx.is_strongly_connected(Gtemp):
                #newDijkstra = self.get_dijkstra_for_subnetworks(G, entryNode)
                self.networks[subnet] = copy.deepcopy(Gtemp)
                #self.network_change_count[subnet] += 1
            else:
                pass
            
        self.fullG = self.update_full_graph(self.fullG, actions = [1]*self.sync_size)
        self.control_fullG = self.update_full_graph(self.control_fullG, actions = action[:self.sync_size])
        self.rand_fullG = self.update_full_graph(self.rand_fullG, actions = randaction[:self.sync_size])
        self.robin_fullG = self.update_full_graph(self.robin_fullG, actions = robinaction[:self.sync_size])

        try:
            for subnet in range(self.sync_size):   #update global picture if controller syncronizes
                if action[subnet] == 1:
                    self.controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
                if randaction[subnet] == 1:
                    self.Rand_controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
                if robinaction[subnet] == 1:
                    self.Robin_controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
        except:
            pass
        
        if  self.step_count % self.place_change_step == 0:       
            self.place_steps += 1
            for entry in range(self.num_controllers):  #get core node / controller here
                if action[entry+self.sync_size] == 1:
                    selected_node = (0,self.store_controllers[entry])
                    if selected_node != self.core_node:
                        self.place_steps = 0
                        self.controller_change_count += 1
                    new_core_node = (0,self.store_controllers[entry])
                    self.controller_assignments[entry] += 1
                    break
            #time_start = time.time() 
            self.core_node = new_core_node
            
        
        controller_paths_list = {}
        for controller in self.store_controllers:
            controller_paths_list[controller] = nx.single_source_shortest_path(self.fullG, (0,controller))
        
        controller_paths_stored = {}
        for controller in self.store_controllers:
            controller_paths_stored[controller] = {}
            controller_paths_stored[controller]["intraDomain"] = []        
            controller_paths_stored[controller]["interController"] = []
            for entry in controller_paths_list[controller]:
                if entry[0] == 0:
                    path = controller_paths_list[controller][entry]   
                    controller_paths_stored[controller]["intraDomain"].append(path)
                else:
                    subnet = entry[0]
                    subnet_controller = self.control_node[subnet]
                    if entry == subnet_controller:   
                        path = controller_paths_list[controller][entry]    
                        controller_paths_stored[controller]["interController"].append(path)       
                        
        reward_placements_ratio = self.get_placement_rewards_ratio(controller_paths_stored, action)
        reward_placements_ratio_rand = self.get_placement_rewards_ratio(controller_paths_stored, randaction)
        reward_placements_ratio_robin = self.get_placement_rewards_ratio(controller_paths_stored, robinaction)
        
        syn_reward, comp_reward = self.get_reward()
        syn_reward_rand, comp_reward_rand = self.get_provided_reward(self.Rand_controller_memory)
        syn_reward_robin, comp_reward_robin  = self.get_provided_reward(self.Robin_controller_memory)
        reward = [syn_reward ,
                         reward_placements_ratio,self.place_steps]    
        randReward = [syn_reward_rand,
                      reward_placements_ratio_rand,self.place_steps] 
        robinReward = [syn_reward_robin,
                    reward_placements_ratio_robin,self.place_steps]
        
        self.comp_reward += comp_reward
        self.comp_reward_rand += comp_reward_rand
        self.comp_reward_robin += comp_reward_robin
        
        self.state = new_state
        self.step_count += 1
        if done:
            print(" ")
            row = [self.episode, self.comp_reward, self.comp_reward_rand, self.comp_reward_robin]
            append_list_as_row(self.comp_filename, row)
            print("Comp rewards are: ", row)
            self.comp_reward = 0
            self.comp_reward_rand = 0
            self.comp_reward_robin = 0
            self.episode += 1
            
            self.network_reset()
            print('network_size was: ', self.network_size)

        return reward, randReward, robinReward
    
    
    
    def get_placement_rewards_ratio(self,controller_paths_stored, action):
        
        intraDomain_rewards = []
        interController_rewards = []
        choice = 0
        for controller in controller_paths_stored:
            intra_sum = 0
            for path in controller_paths_stored[controller]["intraDomain"]:
                intra_sum += len(path)
            inter_sum = 0                 
            for path in controller_paths_stored[controller]["interController"]:
                destination_domain = path[-1][0]
                if action[destination_domain] == 1:
                    inter_sum += len(path)
            intraDomain_rewards.append(1/intra_sum)
            interController_rewards.append(1/inter_sum)
            choice += 1
        
        #intraDomain_rewards = [1/x for x in intraDomain_rewards]   
        #interController_rewards = [1/x for x in interController_rewards] 
        
        normalized_intraDomain_rewards = [x/max(intraDomain_rewards) 
                                        for x in intraDomain_rewards]
        normalized_interController_rewards = [x/max(interController_rewards) 
                                for x in interController_rewards]
        
        rewards_nonscale = [sum(x) for x in zip(intraDomain_rewards,interController_rewards)]
        rewards = [x - min(rewards_nonscale) for x in rewards_nonscale]
        
        choice = 0
        for controller in controller_paths_stored: 
               
            if controller == self.core_node[1]:
                place_reward = rewards[choice]
                
            val1 = self.bestInter_Intra[choice][0]
            val2 = self.bestInter_Intra[choice][1]   
            if normalized_interController_rewards[choice] == 1:
                val1 = self.bestInter_Intra[choice][0] + 1 
                #print(self.bestInter_Intra[choice][0])
                #print(self.bestInter_Intra[choice])
            if normalized_intraDomain_rewards[choice] == 1:
                val2 = self.bestInter_Intra[choice][1] + 1
            self.bestInter_Intra[choice] = [val1,val2]
                
            choice += 1
                
        best_loc = rewards.index(max(rewards))
        if best_loc != self.bestAssign_prev:
            self.bestAssign_count += 1
        self.bestAssign_prev = best_loc
        
        return  place_reward 
    
    def get_provided_reward(self, provided_memory):
        local_nodes_and_delays = self.Calculate_nodes_and_delays(self.local_network_memory)
        global_nodes_and_delays = self.Calculate_nodes_and_delays(provided_memory)
        if len(local_nodes_and_delays) > 0 and len(global_nodes_and_delays) > 0:
            local_best_option = max(local_nodes_and_delays, key=local_nodes_and_delays.get)
            global_best_option = max(global_nodes_and_delays, key=global_nodes_and_delays.get)
            local_best_val = local_nodes_and_delays[local_best_option]
            global_val = local_nodes_and_delays[global_best_option]
            comp_reward = -(local_best_val - global_val)
            if local_best_option == global_best_option:
                reward = 100
            else:
                reward = 0
        else:
            reward = -99999999999999999
        return reward, comp_reward
        
    def get_reward(self):
        local_nodes_and_delays = self.Calculate_nodes_and_delays(self.local_network_memory)
        global_nodes_and_delays = self.Calculate_nodes_and_delays(self.controller_memory)
        if len(local_nodes_and_delays) > 0 and len(global_nodes_and_delays) > 0:
            local_best_option = max(local_nodes_and_delays, key=local_nodes_and_delays.get)
            global_best_option = max(global_nodes_and_delays, key=global_nodes_and_delays.get)
            for subnet in range(self.sync_size):
                if local_best_option[0] == subnet:
                    self.bestcount[subnet] += 1
            local_best_val = local_nodes_and_delays[local_best_option]
            global_val = local_nodes_and_delays[global_best_option]
            comp_reward = -(local_best_val - global_val)
            if local_best_option == global_best_option:
                reward = 100
            else:
                reward = 0
        else:
            reward = -99999999999999999
        return reward, comp_reward

    def Calculate_nodes_and_delays(self,memory):
        c = self.linkCapacities
        x = self.demandTraffic
        nodes_delays = {}
        for subnet in memory:
            if self.networkDownStatus[subnet] == 0:
                for node in memory[subnet][0]:
                    delayCalc = 0
                    #if node != self.connection_to_subnet[subnet]:
                    for link in memory[subnet][0][node]:
                        b = memory[subnet][0][node][link]
                        if (b+x) >= c: 
                            local_delay = 10000000000000.0  
                        else:
                            local_delay = ( ((b+x)/c) / (1 + ((b+x)/c)) )
                        local_delay = 0
                            #delayCalc += local_delay 
                    compDelay = self.computeRate * (self.potentials[subnet][node] - memory[subnet][1][node])
                    delayCalc += compDelay 
                    nodes_delays[(subnet,node)] = delayCalc
            if self.networkDownStatus[subnet] != 0:
                for node in memory[subnet][0]:
                    nodes_delays[(subnet,node)] = 0
        return nodes_delays

    def get_dijkstra_for_subnetworks(self, G, entryNode):
        dijkstraDict = {}
        node1 = entryNode
        for node2 in G:
            dijkstraDict[(node1,node2)] = nx.dijkstra_path(G, source=node1, target=node2)
        return dijkstraDict
    
    def evolve_network(self, G, prob):
        for node1 in G.nodes:
            for node2 in G.nodes:
                if node1 != node2:
                    if prob > random.uniform(0,1):
                        if (node1,node2) in G.edges:
                            G.remove_edge(node1,node2)
                        else:
                            G.add_edge(node1,node2)
        return G

    def network_reset(self):
        self.state = np.zeros(self.state_size)
        print('best server contain count:', self.bestcount)
        print("controller assignments is: ", self.controller_assignments)
        print("controller changes are: ", self.controller_change_count)
        print("placement performance [inter,entra] is: ", self.bestInter_Intra)   
        print("changes to best assignments: ", self.bestAssign_count)
        for subnet in range(self.sync_size):
            self.bestcount[subnet] = 0
            self.bestInter_Intra = [[0]*2]*self.num_controllers
        self.bestAssign_count = 0
        self.step_count = 0
        self.controller_assignments = [0]*self.num_controllers
        self.controller_change_count = 0
        self.step_count = 0

    def visualize_graph(self):
        pos = nx.spring_layout(self.fullG)
        nx.draw(self.fullG, pos, with_labels=True, node_size=700, node_color="skyblue")
        #nx.draw_networkx_edge_labels(self.fullG, pos, edge_labels={(i, j): self.fullG[i][j]['weight'] for i, j in self.fullG.edges()}, label_pos=0.5)
        plt.show()
    











