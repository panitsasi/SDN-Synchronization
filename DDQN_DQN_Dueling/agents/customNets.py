import torch
from torch import nn
import math

class changeNN(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_layers, device):
        super(changeNN,self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_layers
        self.device = device
        
        self.layers = nn.ModuleList()
        
        
        for layer in range(len(self.hidden_dims)+1):
            if layer == 0:
                self.layers.append(nn.Linear(self.input_dim, self.hidden_dims[layer]))
                self.layers.append(nn.ReLU())
            elif layer == len(self.hidden_dims):
                self.layers.append(nn.Linear(self.hidden_dims[layer-1], self.output_dim))
            else:
                self.layers.append(nn.Linear(self.hidden_dims[layer-1], self.hidden_dims[layer]))
                self.layers.append(nn.ReLU())
        
        #self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(0.1)
        
    def forward(self,x):
        
        for layer in self.layers:
            x = layer(x)
            #print(x.size())
            
        return x
    
    def addInputNodes(self, old_status_vec, new_status_vec, budget, weightNormalization = False, SimilarityConsideration = False):
        
        
        #increase hidden layers? TBD....
        
        
        #save old weights and biases
        weightMat = []
        #print (self.layers)
        for layer in self.modules():
            if isinstance(layer, nn.Linear): 
                weightMat.append(layer.weight)
        #add new nodes and biases
        
        #normalize TBD...
        newEntries = len(new_status_vec) - len(old_status_vec)
        
        n = len(new_status_vec)
        o = len(old_status_vec)
        b = budget
        newActionEntries =  int ( math.factorial(n) / (math.factorial(n-b) * math.factorial(b)) - \
                                math.factorial(o) / (math.factorial(o-b) * math.factorial(b)) )
        #print("new actions num: ", newActionEntries)
        
        old_input_dim = self.input_dim
        old_output_dim = self.output_dim
        self.input_dim = self.input_dim + newEntries
        self.output_dim = self.output_dim + newActionEntries
        
        
        if weightNormalization == True:
            
            weightMat[0] = weightMat[0] * old_input_dim  / self.input_dim
            weightMat[-1] = weightMat[-1] * old_output_dim  / self.output_dim
                       
            avgNodeWeight = torch.mean(weightMat[0], 1, True)
            avgActionWeight = torch.mean(weightMat[-1], 0, True)
            NewNodeWeights = avgNodeWeight  #.unsqueeze(dim=-1)
            NewActionWeights = avgActionWeight  #.unsqueeze(dim=0)    
        elif SimilarityConsideration == True:
            pass
        else:
            NewNodeWeights = weightMat[0][:,0].unsqueeze(dim=-1)
            NewActionWeights = weightMat[-1][0,:].unsqueeze(dim=0)
        
        
        
        #print("new node weight  is: ", NewNodeWeights)
        #print("new action weight  is: ", NewActionWeights)

        #print("old input weights  are: ", weightMat[0])
        #print("old action weights are: ", weightMat[-1])
        
        #weights and bias for new entries:
        for entry in range(newEntries):     
            weightMat[0] = torch.cat([weightMat[0],NewNodeWeights],dim=1)
                
        for entry in range(newActionEntries):
            weightMat[-1] = torch.cat([weightMat[-1],NewActionWeights],dim=0)
                
        #print("new input weights  are: ", weightMat[0])
        #print("new action weights  are: ", weightMat[-1])
        
        #add/remove
        self.layers = nn.ModuleList()
        for layer in range(len(self.hidden_dims)+1):
            if layer == 0:
                self.layers.append(nn.Linear(self.input_dim, self.hidden_dims[layer]))
                self.layers.append(nn.ReLU())
            elif layer == len(self.hidden_dims):
                self.layers.append(nn.Linear(self.hidden_dims[layer-1], self.output_dim))
            else:
                self.layers.append(nn.Linear(self.hidden_dims[layer-1], self.hidden_dims[layer]))
                self.layers.append(nn.ReLU())
        
        #print (self.layers)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                try:
                    layer.weight.data = weightMat[0] 
                    weightMat.pop(0)
                except:
                    pass    