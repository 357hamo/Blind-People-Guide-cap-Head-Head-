from torch import nn

class NeuralNetwork(nn.Module):

    
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__() # return temporary object of the super class to allow to access the all methods
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size,num_classes)
        self.relu = nn.ReLU()
        
        
    def forward_propagation(self,x):
        output = self.l1(x)
        output = self.relu(output)
        output = self.l2(output)
        output = self.relu(output)
        output = self.l3(output)
        output = self.relu(output)
        output = self.l4(output)
        # No activations functions and softmax because we compute cross entropy loss for back propagation 
        return output

        