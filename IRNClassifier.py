#Code snippet to load InceptionResNet Classifier

import torch
from torchvision import models
import torch.nn as nn

#Set up data
classNames = ['0','1','2','3','4']

# Set up model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = '/home/oliviayem/CompletePipeline/Classifiers/IRN.pt'
irn = models.resnet18(pretrained=False)
numFeatures = irn.fc.in_features
irn.fc = nn.Linear(numFeatures,len(classNames))
irn.load_state_dict(torch.load(path))
irn = irn.to(device)