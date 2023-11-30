import json
from nlp import tokenize, stem, bag_of_word
import numpy as np
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch import nn
from model import NeuralNetwork


with open('intents.json','r') as f:
    intents = json.load(f)

# print(intents)
tags = []
all_words = []
xy =[]
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)          # list of tags
    # print(intent)
    for pattern in intent['patterns']:
        tokenized_words = tokenize(pattern)
        all_words.extend(tokenized_words)
        xy.append((tokenized_words,tag))
len(all_words)
# print(all_words)
# print(tags)
# print(xy)
ignoring_words = ['?', '.', ',', ';']
all_words_ignoring = [token for token in all_words if token not in ignoring_words]
all_words_ignoring_stemming = stem(all_words_ignoring)
# print(len(all_words_ignoring))
# print(all_words_ignoring_stemming)
tags = sorted(set(tags))
all_words_ignoring_stemming = sorted(set(all_words_ignoring_stemming))
X_train =[]
y_train = []
for (pattern,tag) in xy:
    bag = bag_of_word(pattern,all_words_ignoring_stemming)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)
# print(X_train)
# print(y_train)
X_train = np.array(X_train) # array of bag of words for patterns
y_train = np.array(y_train) # array of index for tags 

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    def __getitem__(self,index): # to access dataset with index
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.n_samples
# Hyperparameters
batch_size = 16
input_size = len(all_words_ignoring_stemming)
hidden_size = 16
output_size = len(tags)
learning_rate = 0.01
num_epoches = 300
dataset = ChatDataset()
# print(dataset.__getitem__(4))
# list(train_loader)
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True, num_workers=0) 
device = torch.device('cpu')
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
# print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
for epoch in range(num_epoches):
    for (vectors_words,labels) in train_loader:
        vectors_words = vectors_words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        output = model.forward_propagation(vectors_words) # to make forward propagation
        loss = criterion(output,labels) # to calculate loss function
        # make back probagation to update the weights 
        optimizer.zero_grad()
        loss.backward() # to make derivatives of the loss
        optimizer.step() # to update the weights
    if (epoch+1) % 100 == 0:
        print(f"epoch {epoch+1}/{num_epoches}, loss={loss.item():.4f} accuracy={(1-loss.item())*100:0.2f}")
print(f"The final Loss is {loss.item():.4f} and accuracy is {(1-loss.item())*100:0.2f}")
data = {
         'model_state' : model.state_dict(),
         'input_size' : input_size,
         'hidden_size' : hidden_size,
         'output_size' : output_size,
         'all_words' : all_words_ignoring_stemming,
         'tags' : tags
        }
file = "data.pth"
torch.save(data,file)
print(f"The Training completed and stored in {file}.")
