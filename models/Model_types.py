import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelA(nn.Module):
    def __init__(self, no_output_chan):
        super().__init__()
        print(f"I am model A")        
        self.conv1 = nn.Conv2d(1,no_output_chan,5)    
        self.conv2 = nn.Conv2d(no_output_chan,16,5)
        self.fc1 = nn.Linear(16*22*5,120) 
        #have to follow input, x.shape before flattern: 
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,35)
        
    def forward(self,x):
        #print(f"{x.shape=}")
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
      
        x = torch.flatten(x,1)
    
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
    
class ModelB(nn.Module):
    def __init__(self, no_output_chan):
        super().__init__()
        print(f"I am model B")
        self.conv1 = nn.Conv2d(1,no_output_chan,5)    
        self.conv2 = nn.Conv2d(no_output_chan,16,5)
        self.fc1 = nn.Linear(16*22*5,120) 
        #have to follow input, x.shape before flattern: 
        self.fc2 = nn.Linear(120,100)
        self.fc3 = nn.Linear(100,50)
        self.fc4 = nn.Linear(50,35)       

        
    def forward(self,x):
        #print(f"{x.shape=}")
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
      
        x = torch.flatten(x,1)
    
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x