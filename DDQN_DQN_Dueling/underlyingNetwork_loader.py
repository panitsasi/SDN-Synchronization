
from operator import sub
import networkx as nx
from itertools import combinations, groupby
import numpy as np
import random
import copy


class NetworkState_SP():
    
    def __init__(self, state_size):
        self.state_size = state_size
        self.minSize = 2
        self.maxSize = 15
        self.innitating_densityKnob = 0.2
        self.changing_densityKnob = 0.001

        self.networks = {}
        self.local_network_paths = {}
        self.connection_to_subnet = {}
        self.controller_memory = {}
        self.network_change_count = {}
        self.failure_count = {}
        self.impactful_change_count = {}
        
        self.Robin_controller_memory = {}
        self.Rand_controller_memory = {}

        for subnet in range(self.state_size):
            size = random.randint(self.minSize, self.maxSize)
            G = gnp_random_connected_graph(size, self.innitating_densityKnob)
            G = G.to_directed()
            self.networks[subnet] = G
            entryNode = random.sample(G.nodes,1)[0]
            self.connection_to_subnet[subnet] = entryNode
            self.local_network_paths[subnet] = self.get_dijkstra_for_subnetworks(G, entryNode)
            self.controller_memory[subnet] = copy.deepcopy(self.local_network_paths[subnet])
            self.network_change_count[subnet] = 0
            self.failure_count[subnet] = 0
            self.impactful_change_count[subnet] = 0
            
            self.Robin_controller_memory[subnet] = copy.deepcopy(self.local_network_paths[subnet])
            self.Rand_controller_memory[subnet] = copy.deepcopy(self.local_network_paths[subnet])

    def update_controllers(self,action,new_state,done,randaction, robinaction, networkDownStatus):
        
        self.networkDownStatus = networkDownStatus

        for subnet in range(self.state_size):   #update all topologies in their local memory
            G = self.networks[subnet]
            entryNode = self.connection_to_subnet[subnet]

            try:
                G = self.evolve_network(G, self.changing_densityKnob)
                newDijkstra = self.get_dijkstra_for_subnetworks(G, entryNode)
                self.networks[subnet] = G
                self.local_network_paths[subnet] = newDijkstra
                self.network_change_count[subnet] += 1
            except:
                pass


        for subnet in range(self.state_size):   #update global picture if controller syncronizes
            if action[subnet] == 1:
                for entry in self.controller_memory[subnet]:
                    control_mem = self.controller_memory[subnet]
                    local_mem = self.local_network_paths[subnet]
                    if control_mem[entry] != local_mem[entry]:
                        self.impactful_change_count[subnet] += 1
                self.controller_memory[subnet] = copy.deepcopy(self.local_network_paths[subnet])
            
            if randaction[subnet] == 1:
                self.Rand_controller_memory[subnet] = copy.deepcopy(self.local_network_paths[subnet])
            if robinaction[subnet] == 1:
                self.Robin_controller_memory[subnet] = copy.deepcopy(self.local_network_paths[subnet])       

        failureVector = self.failureVector()
        
        randReward = self.get_provided_failureVector(self.Rand_controller_memory) 
        robinReward = self.get_provided_failureVector(self.Robin_controller_memory)
        
        self.state = new_state

        if done:
            print(" ")
            print("subNetwork change count (over all plays) is: ", self.network_change_count)
            print("failures count (over 1 play): ", self.failure_count)
            print("impactful change count (over 1 play) is: ", self.impactful_change_count)
            self.network_reset()

        return failureVector, randReward, robinReward
    
    def failureVector(self):
        failures = []
        for subnet in range(self.state_size):
            entryNode = self.connection_to_subnet[subnet]
            local_paths = self.local_network_paths[subnet]
            global_paths = self.controller_memory[subnet]
            for node in self.networks[subnet]:
                if local_paths[(entryNode,node)] == global_paths[(entryNode,node)] and self.networkDownStatus[subnet]==0:
                    failures.append(1)
                else:
                    failures.append(0)
                    self.failure_count[subnet] += 1
        return failures
    
    def get_provided_failureVector(self, memory):
        failures = []
        for subnet in range(self.state_size):
            entryNode = self.connection_to_subnet[subnet]
            local_paths = self.local_network_paths[subnet]
            global_paths = memory[subnet]
            for node in self.networks[subnet]:
                if local_paths[(entryNode,node)] == global_paths[(entryNode,node)] and self.networkDownStatus[subnet]==0:
                    failures.append(1)
                else:
                    failures.append(0)
                    self.failure_count[subnet] += 1
        return failures

    def get_dijkstra_for_subnetworks(self, G, entryNode):
        dijkstraDict = {}
        node1 = entryNode
        for node2 in G:
            dijkstraDict[(node1,node2)] = nx.dijkstra_path(G, source=node1, target=node2)
        return dijkstraDict

    def evolve_network(self, G, prob):
        for node1 in G.nodes:
            for node2 in G.nodes:
                if prob < random.uniform(0,1):
                    if (node1,node2) in G.edges:
                        G.remove_edge(node1,node2)
                    else:
                        G.add_edge(node1,node2)
        return G

    def network_reset(self):
        for subnet in range(self.state_size):
            self.failure_count[subnet] = 0
            self.impactful_change_count[subnet] = 0
        self.state = np.zeros(self.state_size)






def gnp_random_connected_graph(n, p):
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









class NetworkState_LB_C():
    
    def __init__(self, state_size):
        self.state_size = state_size

        self.minSize = 2
        self.maxSize = 15
        self.innitating_densityKnob = 0.2

        self.linkCapacities = 10 
        self.backGroundTrafficInnMean = 3
        self.backGroundTrafficInnSD = 1
        self.UserTrafficSD = 0.1
        self.backTrafficSD = 0.1 
        self.demandTraffic = abs(random.normalvariate(2,1))

        self.serverMaxPotential = 100
        self.serverMinPotential = 5
        self.serverChangeProb = 1 #0.2 #0.1
        self.computeRate = 1
    

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
        
        self.Robin_controller_memory = {}
        self.Rand_controller_memory = {}
        

        for subnet in range(self.state_size):
            size = random.randint(self.minSize, self.maxSize)
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

            
    def create_background_traffics(self,subnet,G,entryNode,network_paths):
        for node in G.nodes: 
            backgroundTraffic = {}          
            for hop in network_paths[(entryNode,node)]:
                backgroundTraffic[hop] = random.normalvariate(self.backGroundTrafficInnMean, self.backGroundTrafficInnSD)
            self.backgroundTraffics[subnet][node] = backgroundTraffic
        return self.backgroundTraffics[subnet]

    def generate_background_traffics(self,subnet,G,entryNode,network_paths):
        for node in G.nodes:          
            for hop in network_paths[(entryNode,node)]:
                self.backgroundTraffics[subnet][node][hop] = self.backgroundTraffics[subnet][node][hop] + self.randomized_traffic_change("background") 
        return self.backgroundTraffics[subnet]          

    def create_server_potential(self,subnet,G,entryNode):
        for node in G.nodes:
            if node == entryNode:
                self.potentials[subnet][node]  = self.serverMaxPotential
            else:
                self.potentials[subnet][node] = self.serverMaxPotential #random.randrange(self.serverMinPotential,self.serverMaxPotential)
        return self.potentials[subnet]

    def create_background_loads(self,subnet,G,entryNode):
        for node in G.nodes:
            randomBackTraff = random.uniform(0.1, random.uniform(0.4,0.8))*self.potentials[subnet][node]
            self.backgroundLoads[subnet][node] = randomBackTraff
        return self.backgroundLoads[subnet]

    def generate_background_loads(self,subnet,G,entryNode):
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
     
     
    def update_controllers(self,action,new_state,done, randaction, robinaction, networkDownStatus):
        
        self.networkDownStatus = networkDownStatus

        for subnet in range(self.state_size):
            G = self.networks[subnet]
            entryNode = self.connection_to_subnet[subnet]
            newTraffics = self.generate_background_traffics(subnet,G,entryNode,self.network_paths[subnet])
            newLoads = self.generate_background_loads(subnet,G,entryNode)
            self.local_network_memory[subnet] = [newTraffics,newLoads]

        for subnet in range(self.state_size):   #update global picture if controller syncronizes
            if action[subnet] == 1:
                self.controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
            if randaction[subnet] == 1:
                self.Rand_controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])
            if robinaction[subnet] == 1:
                self.Robin_controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])


        reward= self.get_reward()   
        randReward = self.get_provided_reward(self.Rand_controller_memory) 
        robinReward = self.get_provided_reward(self.Robin_controller_memory)
        
        self.state = new_state
        if done:
            print(" ")
            self.network_reset()
            print('network_size was: ', self.network_size)

        return reward, randReward, robinReward
    
    def get_provided_reward(self, provided_memory):
        local_nodes_and_delays = self.Calculate_nodes_and_delays(self.local_network_memory)
        global_nodes_and_delays = self.Calculate_nodes_and_delays(provided_memory)
        if len(local_nodes_and_delays) > 0 and len(global_nodes_and_delays) > 0:
            local_best_option = max(local_nodes_and_delays, key=local_nodes_and_delays.get)
            global_best_option = max(global_nodes_and_delays, key=global_nodes_and_delays.get)
            local_best_val = local_nodes_and_delays[local_best_option]
            global_val = local_nodes_and_delays[global_best_option]
            #reward = local_best_val - global_val
            if local_best_option == global_best_option:
                reward = 100
            else:
                reward = 0
        else:
            reward = 0
        return reward
        
    def get_reward(self):
        local_nodes_and_delays = self.Calculate_nodes_and_delays(self.local_network_memory)
        global_nodes_and_delays = self.Calculate_nodes_and_delays(self.controller_memory)
        if len(local_nodes_and_delays) > 0 and len(global_nodes_and_delays) > 0:
            local_best_option = max(local_nodes_and_delays, key=local_nodes_and_delays.get)
            global_best_option = max(global_nodes_and_delays, key=global_nodes_and_delays.get)
            for subnet in range(self.state_size):
                if local_best_option[0] == subnet:
                    self.bestcount[subnet] += 1
            local_best_val = local_nodes_and_delays[local_best_option]
            global_val = local_nodes_and_delays[global_best_option]
            #reward = local_best_val - global_val
            if local_best_option == global_best_option:
                reward = 100
            else:
                reward = 0
        else:
            reward = 0
        return reward

    def Calculate_nodes_and_delays(self,memory):
        c = self.linkCapacities
        x = self.demandTraffic
        nodes_delays = {}
        for subnet in memory:
            if self.networkDownStatus[subnet] == 0:
                for node in memory[subnet][0]:
                    delayCalc = 0
                    if node != self.connection_to_subnet[subnet]:
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
                if prob < random.uniform(0,1):
                    if (node1,node2) in G.edges:
                        G.remove_edge(node1,node2)
                    else:
                        G.add_edge(node1,node2)
        return G

    def network_reset(self):
        self.state = np.zeros(self.state_size)
        print('best server contain count:', self.bestcount)
        for subnet in range(self.state_size):
            self.bestcount[subnet] = 0
            
        
        
        
        
        
























class NetworkState_LB():
    
    def __init__(self, state_size):
        self.state_size = state_size

        self.minSize = 2
        self.maxSize = 15
        self.innitating_densityKnob = 0.2

        self.linkCapacities = 10 
        self.backGroundTrafficInnMean = 3
        self.backGroundTrafficInnSD = 1
        self.UserTrafficSD = 0.1
        self.backTrafficSD = 0.1 
        self.demandTraffic = random.normalvariate(2,1)

        self.serverMaxPotential = 10
        self.serverMinPotential = 5
        self.changeChance = 0.25 #0.25
        self.computeRate = 1

        self.networks = {}
        self.network_paths = {}
        self.server_potential = {}
        self.connection_to_subnet = {}
        self.local_network_memory = {}
        self.controller_memory = {}
        self.backgroundTraffics = {}
        self.potentials = {}
        self.backgroundLoads = {}

        for subnet in range(self.state_size):
            size = random.randint(self.minSize, self.maxSize)
            G = gnp_random_connected_graph(size, self.innitating_densityKnob)
            G = G.to_directed()
            self.networks[subnet] = G
            entryNode = random.sample(G.nodes,1)[0]
            self.connection_to_subnet[subnet] = entryNode
            self.paths = self.get_dijkstra_for_subnetworks(G, entryNode)
            self.network_paths[subnet] = tupled_paths(self.paths)

            self.backgroundTraffics[subnet] = {}
            self.potentials[subnet] = {}
            self.backgroundLoads[subnet] = {}

            self.server_potential[subnet] = self.create_server_potential(subnet,G,entryNode)
            traffics = self.create_background_traffics(subnet,G,entryNode,self.network_paths[subnet])
            abilities = self.create_background_loads(subnet,G,entryNode)
            self.local_network_memory[subnet] = [traffics, abilities]
            self.controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])

            
    def create_background_traffics(self,subnet,G,entryNode,network_paths):
        for node in G.nodes: 
            backgroundTraffic = {}          
            for hop in network_paths[(entryNode,node)]:
                backgroundTraffic[hop] = random.normalvariate(self.backGroundTrafficInnMean, self.backGroundTrafficInnSD)
            self.backgroundTraffics[subnet][node] = backgroundTraffic
        return self.backgroundTraffics[subnet]

    def generate_background_traffics(self,subnet,G,entryNode,network_paths):
        for node in G.nodes:          
            for hop in network_paths[(entryNode,node)]:
                self.backgroundTraffics[subnet][node][hop] = self.backgroundTraffics[subnet][node][hop] + self.randomized_traffic_change("background") 
        return self.backgroundTraffics[subnet]          

    def create_server_potential(self,subnet,G,entryNode):
        for node in G.nodes:
            if node == entryNode:
                self.potentials[subnet][node]  = self.serverMaxPotential
            else:
                self.potentials[subnet][node] = random.randrange(self.serverMinPotential,self.serverMaxPotential)
        return self.potentials[subnet]

    def create_background_loads(self,subnet,G,entryNode):
        for node in G.nodes:
            randomBackTraff = random.uniform(0, random.uniform(0.4,0.8)*self.potentials[subnet][node])
            self.backgroundLoads[subnet][node] = min(self.potentials[subnet][node],randomBackTraff)
        return self.backgroundLoads[subnet]

    def generate_background_loads(self,subnet,G,entryNode):
        for node in G.nodes:
            randomBackTraff = self.backgroundLoads[subnet][node] + random.normalvariate(0,self.serverLoadChange)
            self.backgroundLoads[subnet][node] = max(0.2*self.potentials[subnet][node], min(self.potentials[subnet][node],randomBackTraff))
        return self.backgroundLoads[subnet]


    def randomized_traffic_change(self,flag):
        if flag == "background":
            return random.normalvariate(0,self.backTrafficSD)
        else:
            return random.normalvariate(0,self.UserTrafficSD)
     

    def update_controllers(self,action,new_state,done):

        self.demandTraffic = self.demandTraffic + self.randomized_traffic_change("user")

        for subnet in range(self.state_size):
            G = self.networks[subnet]
            entryNode = self.connection_to_subnet[subnet]
            newTraffics = self.generate_background_traffics(subnet,G,entryNode,self.network_paths[subnet])
            newLoads = self.generate_background_loads(subnet,G,entryNode)
            self.local_network_memory[subnet] = [newTraffics,newLoads]

        for subnet in range(self.state_size):   #update global picture if controller syncronizes
            if action[subnet] == 1:
                self.controller_memory[subnet] = copy.deepcopy(self.local_network_memory[subnet])

        reward = self.get_reward()   
        self.state = new_state
        if done:
            print(" ")
            self.network_reset()

        return reward
        
    def get_reward(self):
        local_nodes_and_delays = self.Calculate_nodes_and_delays(self.local_network_memory)
        global_nodes_and_delays = self.Calculate_nodes_and_delays(self.controller_memory)
        local_best_option = min(local_nodes_and_delays, key=local_nodes_and_delays.get)
        global_best_option = min(global_nodes_and_delays, key=global_nodes_and_delays.get)
        local_best_val = local_nodes_and_delays[local_best_option]
        global_val = local_nodes_and_delays[global_best_option]
        reward = local_best_val - global_val
        return reward

    def Calculate_nodes_and_delays(self,memory):
        c = self.linkCapacities
        x = self.demandTraffic
        nodes_delays = {}
        for subnet in memory:
            for node in memory[subnet][0]:
                if node != self.connection_to_subnet[subnet]:
                    delayCalc = 0
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
                if prob < random.uniform(0,1):
                    if (node1,node2) in G.edges:
                        G.remove_edge(node1,node2)
                    else:
                        G.add_edge(node1,node2)
        return G

    def network_reset(self):
        self.state = np.zeros(self.state_size)