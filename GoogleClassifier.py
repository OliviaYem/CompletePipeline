#Code snippet to load GoogLeNet Classifier

import torch
from torchvision import models
import torch.nn as nn

#Set up data
classNames = ['0','1','2','3','4']

# Set up model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = '/home/oliviayem/CompletePipeline/Classifiers/Google.pt'
net = models.googlenet(pretrained=False)
numFeatures = net.fc.in_features
net.fc = nn.Linear(numFeatures,len(classNames))
net.load_state_dict(torch.load(path))
net = net.to(device)